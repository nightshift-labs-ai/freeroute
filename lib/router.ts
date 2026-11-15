// Routing: which free model serves each request, and where to go when a provider refuses.
//
// The goal is the score first (every request served validly, the quality column high) and, among equal
// scores, the fewest provider calls (the tie-break). A refused call costs one call. So each request goes
// to the candidate with the lowest expected cost: the chance its provider refuses (from the posterior over
// its hidden budget, failure schedule and outage, lib/posterior.ts) plus a price on the quality it gives up
// against the request's best free rating. A hard floor keeps the quality column at or above its target.
//
// The router is a pure function of the brief, the catalog and the provider answers seen so far, so a
// replayed trial makes the same choices. It never reads a clock or a random source.

import { meets, type Brief, type ManifestEntry, type ProviderInfo, type Requires } from './brief.ts';
import { isFreeIn, type Catalog } from './catalog.ts';
import { Posterior } from './posterior.ts';

export interface Candidate {
  provider: ProviderInfo;
  model: string;
  rating: number;
  /** the quality given up against the request's best free rating, times its weight (rating milli) */
  loss: number;
  pOk: number;
  cost: number;
}

/** What the request itself demands of the provider's wire, beyond the model's capabilities. */
export interface WireNeeds {
  /** an OpenAI JSON-mode request: only the OpenAI and Gemini wires carry response_format */
  jsonMode: boolean;
  /** the client named no output cap: an Anthropic-wire call would have to invent one */
  noMaxTokens: boolean;
}

export interface Outcome {
  provider: string;
  model: string;
  ok: boolean;
  status: number;
  code: string;
}

export interface Params {
  /** calls a unit of quality loss is worth: cost = P(refusal) + mu * loss / 1000 */
  mu: number;
  /** the share of the quality bound the plan may give up in all (the quality column floors at 100 - 100*share) */
  lossShare: number;
  /** quality loss held back per unit of weight still to come, so late requests keep room to fail over */
  reservePerWeight: number;
}

export const DEFAULT_PARAMS: Params = { mu: 0.5, lossShare: 0.017, reservePerWeight: 6 };

export class Router {
  readonly brief: Brief;
  readonly catalog: Catalog;
  readonly params: Params;
  readonly posterior: Posterior;
  /** providers or provider/model pairs that answered something we cannot use (bad key, unknown model, broken body) */
  readonly broken = new Set<string>();
  /** quality given up so far by served requests (rating milli times weight) */
  lossSoFar = 0;
  /** the quality bound: each request's best free rating times its weight */
  readonly bound: number;
  readonly #served = new Set<string>();

  constructor(brief: Brief, catalog: Catalog, params: Params = DEFAULT_PARAMS) {
    this.brief = brief;
    this.catalog = catalog;
    this.params = params;
    this.posterior = new Posterior(brief.providers.map((p) => p.id));
    let bound = 0;
    for (const m of brief.manifest) bound += m.weight * (brief.best.get(m.id) ?? this.bestRating(m.requires, { jsonMode: false, noMaxTokens: false }));
    this.bound = bound;
  }

  /** Every free model that meets the request and its wire needs, unranked. */
  eligible(requires: Requires, needs: WireNeeds): { provider: ProviderInfo; model: string; rating: number }[] {
    const out: { provider: ProviderInfo; model: string; rating: number }[] = [];
    for (const p of this.brief.providers) {
      if (needs.jsonMode && p.wire === 'anthropic') continue;
      for (const r of this.brief.ratings) {
        if (r.provider !== p.id || !r.free) continue;
        if (!isFreeIn(this.catalog, p.id, r.model)) continue;
        if (!meets(r, requires)) continue;
        out.push({ provider: p, model: r.model, rating: r.rating_milli });
      }
    }
    return out;
  }

  bestRating(requires: Requires, needs: WireNeeds): number {
    let best = 0;
    for (const c of this.eligible(requires, needs)) best = Math.max(best, c.rating);
    return best;
  }

  order(id: string): number {
    return this.brief.providers.findIndex((p) => p.id === id);
  }

  /** Weight of the manifest requests after `entry` that have not been served. */
  #weightAfter(entry: ManifestEntry | null): number {
    if (entry === null) return 0;
    const at = this.brief.manifest.findIndex((m) => m.id === entry.id);
    let w = 0;
    for (let i = at + 1; i < this.brief.manifest.length; i++) w += this.brief.manifest[i].weight;
    return w;
  }

  /**
   * The candidates for one request, cheapest first; `tried` holds the providers this request has already
   * called. Each provider appears once, with its best model for the request.
   */
  plan(entry: ManifestEntry | null, requires: Requires, needs: WireNeeds, tried: Set<string>): Candidate[] {
    const weight = entry?.weight ?? 1;
    const best = entry !== null && this.brief.best.has(entry.id) ? (this.brief.best.get(entry.id) as number) : this.bestRating(requires, needs);
    const limit = this.params.lossShare * this.bound - this.params.reservePerWeight * this.#weightAfter(entry);
    const byProvider = new Map<string, Candidate>();
    for (const e of this.eligible(requires, needs)) {
      const id = e.provider.id;
      if (this.broken.has(id) || this.broken.has(`${id}/${e.model}`)) continue;
      // A provider this request already called is not excluded: a refusal on its failure schedule leaves
      // the next call free to succeed, and the posterior knows how likely that is.
      void tried;
      const loss = Math.max(0, best - e.rating) * weight;
      const pOk = this.posterior.pNext(id);
      // An Anthropic-wire call must name max_tokens: with none from the client the echo would differ.
      const wirePenalty = needs.noMaxTokens && e.provider.wire === 'anthropic' ? 10 : 0;
      const overLimit = this.lossSoFar + loss > limit ? 5 + (this.lossSoFar + loss - limit) / 100 : 0;
      const cost = (1 - pOk) + (this.params.mu * loss) / 1000 + wirePenalty + overLimit;
      const prev = byProvider.get(id);
      const c: Candidate = { provider: e.provider, model: e.model, rating: e.rating, loss, pOk, cost };
      if (prev === undefined || c.rating > prev.rating) byProvider.set(id, c);
    }
    return [...byProvider.values()].sort((a, b) => a.cost - b.cost || b.rating - a.rating || this.order(a.provider.id) - this.order(b.provider.id));
  }

  record(o: Outcome): void {
    if (o.ok) {
      this.posterior.observe(o.provider, true);
    } else if ((o.status === 402 || o.status === 429) && o.code !== 'concurrent_requests') {
      this.posterior.observe(o.provider, false);
    } else if (o.status === 400 || o.status === 404) {
      this.broken.add(`${o.provider}/${o.model}`);
    } else if (o.status === 401 || o.status === 403) {
      this.broken.add(o.provider);
    } else if (o.status === 0 || o.status >= 500) {
      // No answer, or the provider broke: it may still be working on our call, so never call it again.
      this.broken.add(o.provider);
    } else if (!(o.status >= 200 && o.status < 300)) {
      this.broken.add(`${o.provider}/${o.model}`);
    } else {
      // A 2xx we could not read: the call counted at the provider.
      this.posterior.observe(o.provider, true);
      this.broken.add(`${o.provider}/${o.model}`);
    }
  }

  /** A request was answered by this candidate: its quality loss counts against the floor. */
  served(entry: ManifestEntry | null, c: Candidate): void {
    const key = entry?.id ?? `#${this.#served.size}`;
    if (this.#served.has(key)) return;
    this.#served.add(key);
    this.lossSoFar += c.loss;
  }
}
