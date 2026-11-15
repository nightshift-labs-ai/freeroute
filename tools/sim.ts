// Monte-Carlo simulator of a trial's provider tiers, to tune the router offline.
//
// It draws provider budgets, failure schedules and the outage exactly as the kit's generator does
// (program/generator/draw.mjs: drawInstance, with its HMAC-SHA256 stream), refuses calls exactly as the
// env does (program/env/server.mjs: Providers.handle), and runs lib/router.ts over the 20-request manifest
// of a practice brief, wires assumed perfect. It reports the mean provider calls, the quality column and
// the score over many drawn instances.
//
//   node tools/sim.ts [--n 2000] [--mu 4] [--share 0.017] [--reserve 6] [--practice] [--sweep]

import { createHash, createHmac } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { parseBrief, type Brief } from '../lib/brief.ts';
import { buildCatalog } from '../lib/catalog.ts';
import { DEFAULT_PARAMS, Router, type Params } from '../lib/router.ts';

const KIT = '/home/agent/makefun-kit/agg01/x_s222/free-model-aggregator-01-kit';

class Stream {
  key: Buffer;
  label: string;
  block = 0;
  buf = Buffer.alloc(0);
  pos = 0;
  constructor(seedHex: string, label: string) {
    this.key = Buffer.from(seedHex, 'hex');
    this.label = label;
  }
  byte(): number {
    if (this.pos === this.buf.length) {
      this.buf = createHmac('sha256', this.key).update(`${this.label}/${this.block}`).digest();
      this.block += 1;
      this.pos = 0;
    }
    const b = this.buf[this.pos];
    this.pos += 1;
    return b;
  }
  uint32(): number {
    return ((this.byte() << 24) >>> 0) + (this.byte() << 16) + (this.byte() << 8) + this.byte();
  }
  int(lo: number, hi: number): number {
    const n = hi - lo + 1;
    const limit = Math.floor(2 ** 32 / n) * n;
    for (;;) {
      const w = this.uint32();
      if (w < limit) return lo + (w % n);
    }
  }
  pick<T>(xs: T[]): T {
    return xs[this.int(0, xs.length - 1)];
  }
}

const TABLE: { id: string; lo: number; hi: number; status: number }[] = [
  { id: 'groq', lo: 9, hi: 12, status: 429 },
  { id: 'cerebras', lo: 5, hi: 6, status: 429 },
  { id: 'gemini', lo: 6, hi: 8, status: 429 },
  { id: 'zai', lo: 4, hi: 5, status: 429 },
  { id: 'openrouter', lo: 10, hi: 12, status: 429 },
  { id: 'sambanova', lo: 4, hi: 5, status: 429 },
  { id: 'pollinations', lo: 10, hi: 14, status: 402 },
  { id: 'cloudflare', lo: 7, hi: 8, status: 429 },
  { id: 'ovh', lo: 5, hi: 6, status: 429 },
  { id: 'kilol', lo: 12, hi: 16, status: 429 },
];

export interface TierDraw {
  id: string;
  budget: number;
  fail_calls: number[];
  outage_from: number | null;
  status: number;
}

export function drawTiers(seed: string): TierDraw[] {
  const ids = TABLE.map((p) => p.id);
  const secondCall = new Stream(seed, 'failover/second-call').pick(ids);
  const outage = new Stream(seed, 'outage').pick(ids);
  const tiers = TABLE.map((p) => {
    const budget = new Stream(seed, `provider/${p.id}/budget`).int(p.lo, p.hi);
    const count = new Stream(seed, `provider/${p.id}/fails/count`).int(1, 2);
    const st = new Stream(seed, `provider/${p.id}/fails`);
    const picked = new Set<number>();
    while (picked.size < count) picked.add(st.int(2, budget - 1));
    return { id: p.id, budget, fail_calls: [...picked].sort((a, b) => a - b), outage_from: null as number | null, status: p.status };
  });
  const d = tiers.find((t) => t.id === secondCall)!;
  if (!d.fail_calls.includes(2)) d.fail_calls = [2, ...d.fail_calls.slice(1)].sort((a, b) => a - b);
  const down = tiers.find((t) => t.id === outage)!;
  down.outage_from = new Stream(seed, `provider/${outage}/outage`).int(2, down.budget - 1);
  return tiers;
}

export interface SimResult {
  calls: number;
  valid: number;
  quality: number;
  failover: number;
  score: number;
  loss: number;
  trace: string[];
}

export function simulate(brief: Brief, tiers: TierDraw[], params: Params, trace = false): SimResult {
  const catalog = buildCatalog(brief, brief.providers.map(() => null));
  const router = new Router(brief, catalog, params);
  const counts = new Map<string, number>();
  let calls = 0;
  let valid = 0;
  let earned = 0;
  let bound = 0;
  let foReq = 0;
  let foOk = 0;
  const lines: string[] = [];
  for (const entry of brief.manifest) {
    const best = brief.best.get(entry.id) ?? 0;
    bound += best * entry.weight;
    const needs = { jsonMode: entry.api === 'openai' && entry.requires.json === true, noMaxTokens: false };
    const tried = new Set<string>();
    let first = true;
    let firstFailed = false;
    let servedBy = '';
    for (let attempt = 0; attempt < 24; attempt++) {
      const plan = router.plan(entry, entry.requires, needs, tried);
      const c = plan[0];
      if (c === undefined) break;
      tried.add(c.provider.id);
      const t = tiers.find((x) => x.id === c.provider.id)!;
      const n = (counts.get(t.id) ?? 0) + 1;
      counts.set(t.id, n);
      calls += 1;
      const refused = n > t.budget || (t.outage_from !== null && n >= t.outage_from) || t.fail_calls.includes(n);
      if (first) firstFailed = refused;
      first = false;
      router.record({ provider: t.id, model: c.model, ok: !refused, status: refused ? t.status : 200, code: refused ? 'x' : '' });
      if (trace) lines.push(`${entry.id} -> ${t.id}#${n} ${c.model} p=${c.pOk.toFixed(2)} loss=${c.loss} ${refused ? 'REFUSED' : 'ok'}`);
      if (!refused) {
        router.served(entry, c);
        valid += 1;
        earned += Math.min(c.rating, best) * entry.weight;
        servedBy = c.provider.id;
        break;
      }
    }
    if (firstFailed) {
      foReq += 1;
      if (servedBy !== '') foOk += 1;
    }
  }
  const quality = Math.floor((100 * earned) / bound);
  const served = Math.floor((100 * valid) / brief.manifest.length);
  const failover = foReq === 0 ? (valid > 0 ? 100 : 0) : Math.floor((100 * foOk) / foReq);
  const score = Math.round((60 * quality + 15 * served + 10 * 100 + 10 * failover + 5 * 100) / 100);
  return { calls, valid, quality, failover, score, loss: bound - earned, trace: lines };
}

function seeds(n: number, salt = 'sim'): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(createHash('sha256').update(`${salt}/${i}`).digest('hex'));
  return out;
}

export function evaluate(brief: Brief, params: Params, n: number, salt = 'sim'): { calls: number; score: number; q98: number; s99: number; minQ: number; maxCalls: number; meanLoss: number } {
  let calls = 0;
  let score = 0;
  let q98 = 0;
  let s99 = 0;
  let minQ = 100;
  let maxCalls = 0;
  let loss = 0;
  for (const seed of seeds(n, salt)) {
    const r = simulate(brief, drawTiers(seed), params);
    calls += r.calls;
    score += r.score;
    if (r.quality >= 98) q98 += 1;
    if (r.score >= 99) s99 += 1;
    minQ = Math.min(minQ, r.quality);
    maxCalls = Math.max(maxCalls, r.calls);
    loss += r.loss;
  }
  return { calls: calls / n, score: score / n, q98: q98 / n, s99: s99 / n, minQ, maxCalls, meanLoss: loss / n };
}

const isMain = process.argv[1] !== undefined && import.meta.url.endsWith(process.argv[1].split('/').pop() ?? '');
if (isMain) {
  const args = process.argv.slice(2);
  const opt = (name: string, dflt: number): number => {
    const i = args.indexOf(name);
    return i !== -1 ? Number(args[i + 1]) : dflt;
  };
  const brief = parseBrief(readFileSync(`${KIT}/practice/trial-0/input/prompt.txt`, 'utf8'));
  const params: Params = { mu: opt('--mu', DEFAULT_PARAMS.mu), lossShare: opt('--share', DEFAULT_PARAMS.lossShare), reservePerWeight: opt('--reserve', DEFAULT_PARAMS.reservePerWeight) };
  const n = opt('--n', 2000);
  if (args.includes('--practice')) {
    const yaml = readFileSync(`${KIT}/rules/instance.yaml`, 'utf8');
    const practiceSeeds = [...yaml.matchAll(/seed:\s*"([0-9a-f]{64})"/g)].map((m) => m[1]);
    practiceSeeds.forEach((seed, t) => {
      const tiers = drawTiers(seed);
      const r = simulate(brief, tiers, params, true);
      process.stdout.write(`practice ${t}: calls ${r.calls} quality ${r.quality} score ${r.score}\n  ${r.trace.join('\n  ')}\n`);
    });
  } else if (args.includes('--sweep')) {
    for (const mu of [0, 1, 2, 4, 8, 16]) {
      for (const reserve of [0, 6, 12]) {
        const p = { ...params, mu, reservePerWeight: reserve };
        const e = evaluate(brief, p, n);
        process.stdout.write(`mu ${mu} reserve ${reserve}: calls ${e.calls.toFixed(3)} score ${e.score.toFixed(3)} s99 ${(100 * e.s99).toFixed(1)}% q98 ${(100 * e.q98).toFixed(1)}% minQ ${e.minQ} maxCalls ${e.maxCalls} loss ${e.meanLoss.toFixed(0)}\n`);
      }
    }
  } else {
    const e = evaluate(brief, params, n);
    process.stdout.write(`${JSON.stringify(params)}: calls ${e.calls.toFixed(3)} score ${e.score.toFixed(3)} s99 ${(100 * e.s99).toFixed(1)}% q98 ${(100 * e.q98).toFixed(1)}% minQ ${e.minQ} maxCalls ${e.maxCalls} loss ${e.meanLoss.toFixed(0)}\n`);
  }
}
