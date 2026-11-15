// Routing: which free model serves each request, and where to go when a provider refuses.
//
// The router is a pure function of the brief, the catalog and the provider answers seen so far, so a
// replayed trial makes the same choices. It never reads a clock or a random source.

import { meets, type Brief, type ManifestEntry, type ProviderInfo, type Requires, type Wire } from './brief.ts';
import { isFreeIn, type Catalog } from './catalog.ts';

export interface Candidate {
  provider: ProviderInfo;
  model: string;
  rating: number;
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
  /** the provider's call number for this call when it reached the budget (1 for its first) */
  ok: boolean;
  status: number;
  code: string;
}

export class Router {
  readonly brief: Brief;
  readonly catalog: Catalog;
  /** calls each provider has counted (successes and refusals that reached it) */
  calls = new Map<string, number>();
  /** refusals (402/429 on its schedule, outage or budget) each provider answered */
  refusals = new Map<string, number>();
  successes = new Map<string, number>();
  /** providers that answered something we cannot use (bad key, a model it does not serve, a broken body) */
  broken = new Set<string>();

  constructor(brief: Brief, catalog: Catalog) {
    this.brief = brief;
    this.catalog = catalog;
  }

  /** Every free model that meets the request, best rating first. */
  eligible(requires: Requires, needs: WireNeeds): Candidate[] {
    const out: Candidate[] = [];
    for (const p of this.brief.providers) {
      if (needs.jsonMode && p.wire === 'anthropic') continue;
      for (const r of this.brief.ratings) {
        if (r.provider !== p.id || !r.free) continue;
        if (!isFreeIn(this.catalog, p.id, r.model)) continue;
        if (!meets(r, requires)) continue;
        out.push({ provider: p, model: r.model, rating: r.rating_milli });
      }
    }
    const wirePenalty = (w: Wire): number => (needs.noMaxTokens && w === 'anthropic' ? 1 : 0);
    out.sort((a, b) => wirePenalty(a.provider.wire) - wirePenalty(b.provider.wire) || b.rating - a.rating || this.order(a.provider.id) - this.order(b.provider.id));
    return out;
  }

  order(id: string): number {
    return this.brief.providers.findIndex((p) => p.id === id);
  }

  /**
   * The candidates for one request in the order to try them; `tried` holds the providers this request
   * has already called.
   */
  plan(_entry: ManifestEntry | null, requires: Requires, needs: WireNeeds, tried: Set<string>): Candidate[] {
    const all = this.eligible(requires, needs).filter((c) => !tried.has(c.provider.id) && !this.broken.has(c.provider.id));
    const healthy = all.filter((c) => (this.refusals.get(c.provider.id) ?? 0) === 0);
    const hurt = all.filter((c) => (this.refusals.get(c.provider.id) ?? 0) > 0);
    return [...healthy, ...hurt];
  }

  record(o: Outcome): void {
    const bump = (m: Map<string, number>): void => {
      m.set(o.provider, (m.get(o.provider) ?? 0) + 1);
    };
    if (o.ok) {
      bump(this.calls);
      bump(this.successes);
    } else if ((o.status === 402 || o.status === 429) && o.code !== 'concurrent_requests') {
      bump(this.calls);
      bump(this.refusals);
    } else if (o.status === 401 || o.status === 400 || o.status === 404) {
      this.broken.add(o.provider);
    } else {
      bump(this.refusals);
    }
  }
}
