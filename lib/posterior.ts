// What a provider's free tier will do next: a Bayesian posterior over its hidden budget, its failure
// schedule and an outage, from the answers its calls have given so far.
//
// The kit's env (program/env/server.mjs, program/generator/draw.mjs, both public) draws each provider's
// free tier the same way every trial: a budget B in the provider's [min, max] calls; one or two failing
// call numbers drawn from 2..B-1; one provider (uniform over the ten) has call 2 forced into its
// schedule; one provider (uniform, independent) goes out from a call drawn in 2..B-1 on. A call fails when
// its number is over B, at or past the outage, or on the schedule; a refused call still counts. So a
// provider's first call always succeeds, and every refusal tells us something about the rest.
//
// The posterior is exact: per provider we enumerate every (B, schedule, outage) hypothesis consistent with
// the call outcomes seen, and couple the providers through "exactly one designated, exactly one outage".
// It is a pure function of the outcomes: no clock, no randomness.

/** Each known provider's budget range, from the kit's public env table; others get GENERIC_BUDGET. */
export const KNOWN_BUDGETS: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
export const GENERIC_BUDGET: [number, number] = [4, 12];

interface Hyp {
  B: number;
  /** failing call numbers as a bit mask (bit k = call k) */
  F: number;
  /** first outage call, 0 for none */
  O: number;
  w: number;
}

function fails(h: Hyp, k: number): boolean {
  return k > h.B || (h.O !== 0 && k >= h.O) || (h.F & (2 ** k)) !== 0;
}

function hypotheses(lo: number, hi: number, designated: boolean, outage: boolean): Hyp[] {
  const out: Hyp[] = [];
  const nB = hi - lo + 1;
  for (let B = lo; B <= hi; B++) {
    const pool: number[] = [];
    for (let k = 2; k <= B - 1; k++) pool.push(k);
    const n = pool.length;
    const subsets: number[][] = [];
    if (n >= 1) for (const a of pool) subsets.push([a]);
    const singles = subsets.length;
    if (n >= 2) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) subsets.push([pool[i], pool[j]]);
    const pairs = subsets.length - singles;
    for (let s = 0; s < subsets.length; s++) {
      let set = subsets[s];
      const isSingle = s < singles;
      // count 1 or 2 with probability 1/2 each (a pair needs room for two numbers)
      const pCount = n >= 2 ? 0.5 : isSingle ? 1 : 0;
      const pSet = pCount / (isSingle ? singles : pairs);
      if (designated && !set.includes(2)) set = [2, ...set.slice(1)];
      let F = 0;
      for (const k of set) F |= 2 ** k;
      const w = pSet / nB;
      if (outage) {
        for (let O = 2; O <= B - 1; O++) out.push({ B, F, O, w: w / (B - 2) });
      } else {
        out.push({ B, F, O: 0, w });
      }
    }
  }
  return out;
}

/** The four hypothesis sets of one provider: [designated][outage]. */
type HypSets = [[Hyp[], Hyp[]], [Hyp[], Hyp[]]];

function hypSets(lo: number, hi: number): HypSets {
  return [
    [hypotheses(lo, hi, false, false), hypotheses(lo, hi, false, true)],
    [hypotheses(lo, hi, true, false), hypotheses(lo, hi, true, true)],
  ];
}

/** Total prior weight of the hypotheses consistent with `obs` (obs[i] = call i+1 succeeded). */
function weight(hs: Hyp[], obs: boolean[]): number {
  let total = 0;
  outer: for (const h of hs) {
    for (let i = 0; i < obs.length; i++) if (fails(h, i + 1) === obs[i]) continue outer;
    total += h.w;
  }
  return total;
}

const SETS = new Map<string, HypSets>();
/** Memo of a provider's four likelihoods by its budget range and outcome string: a pure function. */
const LIKELIHOOD = new Map<string, number[]>();

function setsFor(lo: number, hi: number): HypSets {
  const key = `${lo}-${hi}`;
  let s = SETS.get(key);
  if (s === undefined) {
    s = hypSets(lo, hi);
    SETS.set(key, s);
  }
  return s;
}

/** [L(d=0,o=0), L(d=0,o=1), L(d=1,o=0), L(d=1,o=1)] of an outcome sequence for a provider with budget range [lo, hi]. */
function likelihoods(lo: number, hi: number, obs: boolean[]): number[] {
  const key = `${lo}-${hi}|${obs.map((x) => (x ? '1' : '0')).join('')}`;
  let v = LIKELIHOOD.get(key);
  if (v === undefined) {
    const s = setsFor(lo, hi);
    v = [weight(s[0][0], obs), weight(s[0][1], obs), weight(s[1][0], obs), weight(s[1][1], obs)];
    LIKELIHOOD.set(key, v);
  }
  return v;
}

export class Posterior {
  readonly ids: string[];
  readonly #ranges: [number, number][];
  /** counted call outcomes per provider, in call order */
  readonly obs: boolean[][];
  #cache: Map<string, number> = new Map();

  constructor(ids: string[]) {
    this.ids = ids;
    this.#ranges = ids.map((id) => KNOWN_BUDGETS[id] ?? GENERIC_BUDGET);
    this.obs = ids.map(() => []);
  }

  index(id: string): number {
    return this.ids.indexOf(id);
  }

  /** Record one counted call (a success, or a refusal that spent the provider's budget). */
  observe(id: string, ok: boolean): void {
    const i = this.index(id);
    if (i === -1) return;
    this.obs[i].push(ok);
    this.#cache.clear();
  }

  calls(id: string): number {
    const i = this.index(id);
    return i === -1 ? 0 : this.obs[i].length;
  }

  #L(q: number, extra: boolean[] = []): number[] {
    const [lo, hi] = this.#ranges[q];
    return likelihoods(lo, hi, extra.length === 0 ? this.obs[q] : [...this.obs[q], ...extra]);
  }

  /** P(the next `outcomes.length` calls to provider `id` go exactly as `outcomes`), given everything seen. */
  predict(id: string, outcomes: boolean[]): number {
    const key = `${id}|${outcomes.map((x) => (x ? 1 : 0)).join('')}`;
    const hit = this.#cache.get(key);
    if (hit !== undefined) return hit;
    const p = this.index(id);
    if (p === -1) return 0.5;
    const n = this.ids.length;
    // L[q][2d+o]: weight of provider q's outcomes under (designated d, outage o)
    const L = this.ids.map((_, q) => this.#L(q));
    const Lnext = this.#L(p, outcomes);
    // Every provider plain, then the joint weight of (designated d, outage o) by likelihood ratios.
    let num = 0;
    let den = 0;
    let base = 1;
    for (let q = 0; q < n; q++) base *= L[q][0];
    for (let d = 0; d < n; d++) {
      for (let o = 0; o < n; o++) {
        let w: number;
        if (base > 0) {
          w = base;
          if (d === o) w = (w / L[d][0]) * L[d][3];
          else w = (((w / L[d][0]) * L[d][2]) / L[o][0]) * L[o][1];
        } else {
          w = 1;
          for (let q = 0; q < n; q++) w *= L[q][(q === d ? 2 : 0) + (q === o ? 1 : 0)];
        }
        if (!(w > 0)) continue;
        const role = (p === d ? 2 : 0) + (p === o ? 1 : 0);
        den += w;
        if (L[p][role] > 0) num += (w * Lnext[role]) / L[p][role];
      }
    }
    const value = den === 0 ? 0 : num / den;
    this.#cache.set(key, value);
    return value;
  }

  /** P(the next call to provider `id` succeeds). */
  pNext(id: string): number {
    return this.predict(id, [true]);
  }
}
