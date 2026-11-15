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
import { fails, fastPOk, Posterior, Rng, type Hyp } from './posterior.ts';
import { createHash } from 'node:crypto';

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
  /** the request carries something only its own wire passes through intact: keep it on that wire if any provider can */
  sameWire?: 'openai' | 'anthropic' | null;
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
  /** rollout lookahead: candidates compared per decision (0 or 1 turns it off) */
  rolloutK: number;
  /** rollout lookahead: posterior draws per decision */
  rolloutM: number;
  /** rollout objective: calls + objMu * loss / 1000, plus penalties for an unserved request or a quality floor broken */
  objMu: number;
  /** 0: greedy (plan order, or rollouts when rolloutK > 1); 1: the slot-assignment plan */
  planner: number;
  /** future call slots per provider the assignment plan considers */
  slotT: number;
  /** quality loss the assignment plan leaves unplanned under the floor, for the fallbacks refusals force */
  planMargin: number;
  /** slot costs: 0 the chained chance a slot is refused; 1 the exact expected extra refusals of each success */
  slotModel: number;
  /** slotModel 1: refusals a success that can never come is charged (what finding out costs) */
  deadCost: number;
}

export const DEFAULT_PARAMS: Params = { mu: 0.5, lossShare: 0.0185, reservePerWeight: 6, rolloutK: 0, rolloutM: 32, objMu: 0.5, planner: 1, slotT: 12, planMargin: 0, slotModel: 0, deadCost: 1.5 };

/** Min-cost assignment of every row to a distinct column (rows <= columns): the Hungarian method with potentials. */
export function hungarian(cost: number[][]): number[] {
  const n = cost.length;
  if (n === 0) return [];
  const m = cost[0].length;
  const INF = Number.POSITIVE_INFINITY;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(m + 1).fill(0);
  const p = new Array<number>(m + 1).fill(0);
  const way = new Array<number>(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(m + 1).fill(INF);
    const used = new Array<boolean>(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  const out = new Array<number>(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j] !== 0) out[p[j] - 1] = j - 1;
  return out;
}

/** The quality column's floor we hold: 98 of 100 keeps the score at its 99 ceiling. */
const QUALITY_FLOOR_SHARE = 0.0195;
const UNSERVED_PENALTY = 25;
const FLOOR_PENALTY = 25;

interface Slot {
  q: number;
  rating: number;
  loss: number;
}

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
  /** per manifest request: each provider's best slot for it (provider index, rating, loss) */
  readonly #slots: Slot[][];
  readonly #weightsAfter: number[];
  readonly #seed: string;
  #decisions = 0;
  /** test hook: a provider to prefer whenever it can serve (empty in a trial) */
  prefer = '';

  constructor(brief: Brief, catalog: Catalog, params: Params = DEFAULT_PARAMS, seed = '') {
    this.brief = brief;
    this.catalog = catalog;
    this.params = params;
    this.#seed = seed;
    this.posterior = new Posterior(brief.providers.map((p) => p.id));
    let bound = 0;
    for (const m of brief.manifest) bound += m.weight * (brief.best.get(m.id) ?? this.bestRating(m.requires, { jsonMode: false, noMaxTokens: false }));
    this.bound = bound;
    this.#slots = brief.manifest.map((m) => {
      const needs = { jsonMode: m.api === 'openai' && m.requires.json === true, noMaxTokens: false };
      const best = brief.best.get(m.id) ?? this.bestRating(m.requires, needs);
      const byQ = new Map<number, Slot>();
      for (const e of this.eligible(m.requires, needs)) {
        const q = this.order(e.provider.id);
        const slot = { q, rating: e.rating, loss: Math.max(0, best - e.rating) * m.weight };
        const prev = byQ.get(q);
        if (prev === undefined || slot.rating > prev.rating) byQ.set(q, slot);
      }
      return [...byQ.values()].sort((a, b) => a.q - b.q);
    });
    this.#weightsAfter = brief.manifest.map((_, i) => brief.manifest.slice(i + 1).reduce((n, m) => n + m.weight, 0));
  }

  /** Every free model that meets the request and its wire needs, unranked. */
  eligible(requires: Requires, needs: WireNeeds): { provider: ProviderInfo; model: string; rating: number }[] {
    const all = this.#eligible(requires, needs);
    if (needs.sameWire === undefined || needs.sameWire === null) return all;
    const same = all.filter((e) => e.provider.wire === needs.sameWire);
    return same.length > 0 ? same : all;
  }

  #eligible(requires: Requires, needs: WireNeeds): { provider: ProviderInfo; model: string; rating: number }[] {
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

  /**
   * The candidate to call next for this request. The greedy order (plan) decides, unless the rollout
   * lookahead is on: then the top candidates are each played out to the end of the workload over the same
   * posterior draws of every provider's hidden tier, the rest of the workload served greedily, and the
   * candidate with the lowest expected calls (plus a price on quality, and heavy penalties for an unserved
   * request or the quality floor broken) wins. The draws come from a generator seeded by the trial's seed
   * and the decision's number, so the same answers always lead to the same choice.
   */
  choose(entry: ManifestEntry | null, requires: Requires, needs: WireNeeds, tried: Set<string>): Candidate | undefined {
    const cands = this.plan(entry, requires, needs, tried);
    this.#decisions += 1;
    const at = entry === null ? -1 : this.brief.manifest.findIndex((m) => m.id === entry.id);
    // A test hook (FREEROUTE_PREFER, never set in a trial): send every request a provider can serve to it.
    if (this.prefer !== '') {
      const p = cands.find((c) => c.provider.id === this.prefer && c.pOk > 0.05);
      if (p !== undefined) return p;
    }
    if (this.params.planner === 1 && at !== -1 && cands.length > 1) return this.#assign(at, cands) ?? cands[0];
    const K = Math.min(this.params.rolloutK, cands.length);
    if (K <= 1 || at === -1) return cands[0];
    const top = cands.slice(0, K);
    const digest = createHash('sha256').update(`freeroute/rollout/${this.#seed}/${this.#decisions}`).digest();
    const rng = new Rng(digest.readUInt32BE(0));
    const totals = top.map(() => 0);
    const slotsNow: Slot[] = cands.map((c) => ({ q: this.order(c.provider.id), rating: c.rating, loss: c.loss }));
    for (let m = 0; m < this.params.rolloutM; m++) {
      const hidden = this.posterior.sample(rng);
      top.forEach((c, i) => {
        totals[i] += this.#rollout(hidden, at, slotsNow, this.order(c.provider.id));
      });
    }
    let bestI = 0;
    for (let i = 1; i < top.length; i++) if (totals[i] < totals[bestI] - 1e-9) bestI = i;
    return top[bestI];
  }

  /**
   * The certainty-equivalent plan: every request still to serve (this one first) is assigned one future
   * call slot of one provider (its t-th next call, which succeeds with the posterior's chance given the
   * calls before it succeed), at the least total cost: the chance each slot is refused plus a price on the
   * quality each request gives up. The price rises (a Lagrange multiplier, found by bisection) until the
   * whole plan keeps the quality floor. This request goes to the provider its row was given. Scarce
   * capabilities are respected for free: a later vision request keeps a vision provider's sure first call.
   */
  #assign(at: number, cands: Candidate[]): Candidate | null {
    const n = this.posterior.ids.length;
    const T = this.params.slotT;
    const slotCost: number[][] = [];
    const roles = this.params.slotModel === 1 ? this.posterior.roleWeights() : [];
    for (let q = 0; q < n; q++) {
      const id = this.posterior.ids[q];
      const row: number[] = [];
      if (this.params.slotModel === 1) {
        // Slot t costs the extra refusals the t-th success from this provider is expected to bring.
        const e = this.posterior.expectedRefusals(id, T, this.params.deadCost, roles);
        for (let t = 0; t < T; t++) row.push(Math.max(0, e[t] - (t > 0 ? e[t - 1] : 0)));
      } else {
        let prev = 1;
        for (let t = 1; t <= T; t++) {
          const all = this.posterior.predict(id, new Array<boolean>(t).fill(true));
          const cond = prev > 0 ? all / prev : 0;
          row.push(1 - Math.min(1, Math.max(0, cond)));
          prev = all;
        }
      }
      slotCost.push(row);
    }
    const rows: Slot[][] = [cands.map((c) => ({ q: this.order(c.provider.id), rating: c.rating, loss: c.loss }))];
    for (let j = at + 1; j < this.brief.manifest.length; j++) rows.push(this.#slots[j].filter((s) => !this.broken.has(this.posterior.ids[s.q])));
    const limit = this.params.lossShare * this.bound - this.params.planMargin;
    const solve = (mu: number): { first: number; loss: number } => {
      const m = n * T;
      const BIG = 1e6;
      const cost: number[][] = rows.map((slots, i) => {
        const r = new Array<number>(m).fill(BIG);
        // This request makes its provider's very next call: row 0 may take only a first slot.
        for (const s of slots) for (let t = 0; t < (i === 0 ? 1 : T); t++) r[s.q * T + t] = slotCost[s.q][t] + (mu * s.loss) / 1000;
        return r;
      });
      const assign = hungarian(cost);
      let loss = 0;
      rows.forEach((slots, i) => {
        const q = Math.floor(assign[i] / T);
        const s = slots.find((x) => x.q === q);
        loss += s === undefined ? 0 : s.loss;
      });
      return { first: Math.floor(assign[0] / T), loss };
    };
    let plan = solve(this.params.mu);
    if (this.lossSoFar + plan.loss > limit) {
      // The price of quality is capped: past it a refusal-prone provider would win on quality alone.
      const MU_CAP = 12;
      let lo = this.params.mu;
      let hi = MU_CAP;
      let fit = solve(hi);
      if (this.lossSoFar + fit.loss > limit) return cands.find((c) => this.order(c.provider.id) === fit.first) ?? null;
      for (let k = 0; k < 10; k++) {
        const mid = (lo + hi) / 2;
        const p = solve(mid);
        if (this.lossSoFar + p.loss > limit) lo = mid;
        else {
          hi = mid;
          fit = p;
        }
      }
      plan = fit;
    }
    return cands.find((c) => this.order(c.provider.id) === plan.first) ?? null;
  }

  /** Play the workload out from request `at` on one draw of the hidden tiers, `first` called first. */
  #rollout(hidden: Hyp[], at: number, slotsNow: Slot[], first: number): number {
    const n = this.posterior.ids.length;
    const counts: number[] = [];
    const keys: string[] = [];
    const ranges: [number, number][] = [];
    for (let q = 0; q < n; q++) {
      const obs = this.posterior.obs[q];
      counts.push(obs.length);
      keys.push(obs.map((x) => (x ? '1' : '0')).join(''));
      ranges.push(this.posterior.range(q));
    }
    let loss = this.lossSoFar;
    let calls = 0;
    let penalty = 0;
    const mu = this.params.mu / 1000;
    const total = this.params.lossShare * this.bound;
    for (let j = at; j < this.brief.manifest.length; j++) {
      const slots = j === at ? slotsNow : this.#slots[j];
      const limit = total - this.params.reservePerWeight * this.#weightsAfter[j];
      let served = false;
      for (let attempt = 0; attempt < 16 && slots.length > 0; attempt++) {
        let pick: Slot | null = null;
        if (j === at && attempt === 0) {
          pick = slots.find((s) => s.q === first) ?? null;
        }
        if (pick === null) {
          let bestCost = Infinity;
          for (const s of slots) {
            const [lo, hi] = ranges[s.q];
            const p = fastPOk(lo, hi, keys[s.q]);
            const over = loss + s.loss > limit ? 5 + (loss + s.loss - limit) / 100 : 0;
            const cost = 1 - p + mu * s.loss + over;
            if (cost < bestCost - 1e-12) {
              bestCost = cost;
              pick = s;
            }
          }
        }
        if (pick === null) break;
        const k = counts[pick.q] + 1;
        counts[pick.q] = k;
        calls += 1;
        const refused = fails(hidden[pick.q], k);
        keys[pick.q] += refused ? '0' : '1';
        if (!refused) {
          loss += pick.loss;
          served = true;
          break;
        }
      }
      if (!served) penalty += UNSERVED_PENALTY;
    }
    if (loss > QUALITY_FLOOR_SHARE * this.bound) penalty += FLOOR_PENALTY;
    return calls + (this.params.objMu * loss) / 1000 + penalty;
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
