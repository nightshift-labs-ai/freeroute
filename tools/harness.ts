// Local trial harness: runs the kit's env (program/env/server.mjs) seeded with a practice instance, runs
// this repository's main.ts against it the way a trial does, then runs the kit's scorer on the env's
// state export and prints the score columns. Docker is not needed: the env, the server and the scorer run
// as plain node22 processes, the env on loopback as the kit's process driver runs it.
//
//   node tools/harness.ts [--trial 0|1|2|all] [--twice] [--kit <dir>] [--quiet] [--keep]
//
// --twice runs each trial a second time and compares the two state exports byte for byte (the replay
// check: the env is deterministic, so a second live pass must leave the same export).

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const NODE = process.execPath;
const args = process.argv.slice(2);
const flag = (name: string): boolean => args.includes(name);
const opt = (name: string, dflt: string): string => {
  const i = args.indexOf(name);
  return i !== -1 && i + 1 < args.length ? args[i + 1] : dflt;
};
const KIT = resolve(opt('--kit', '/home/agent/makefun-kit/agg01/x_s222/free-model-aggregator-01-kit'));
const REPO = resolve(new URL('..', import.meta.url).pathname);
const QUIET = flag('--quiet');
const KEEP = flag('--keep');
const TRIAL_NOW = '2026-11-13T22:56:12Z';

function seedOf(trial: number): string {
  const text = readFileSync(join(KIT, 'rules/instance.yaml'), 'utf8');
  const seeds = [...text.matchAll(/seed:\s*"([0-9a-f]{64})"/g)].map((m) => m[1]);
  return seeds[trial] ?? '0'.repeat(64);
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

interface Run {
  scores: Record<string, number>;
  state: string;
  list: string;
  serverLog: string;
  exitCode: number | null;
  wallMs: number;
}

async function runTrial(trial: number, envPort: number): Promise<Run> {
  const dir = mkdtempSync(join(tmpdir(), `fr-trial${trial}-`));
  const stateDir = join(dir, 'state');
  const dataDir = join(KIT, `practice/trial-${trial}/input/env`);
  const prompt = join(KIT, `practice/trial-${trial}/input/prompt.txt`);
  const reference = join(KIT, `practice/trial-${trial}/reference`);
  const env = spawn(NODE, [join(KIT, 'program/env/server.mjs')], {
    env: { ...process.env, MAKEFUN_STATE_DIR: stateDir, MAKEFUN_ENV_DATA_DIR: dataDir, MAKEFUN_ENV_PORT: String(envPort), MAKEFUN_ENV_HOST: '127.0.0.1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let envLog = '';
  env.stdout.on('data', (d) => (envLog += d));
  env.stderr.on('data', (d) => (envLog += d));
  for (let i = 0; i < 100 && !existsSync(join(stateDir, 'ready')); i++) await sleep(50);
  if (!existsSync(join(stateDir, 'ready'))) throw new Error(`the env did not start:\n${envLog}`);
  const envUrl = `http://127.0.0.1:${envPort}`;
  const trialEnv = {
    PATH: process.env.PATH ?? '',
    HOME: process.env.HOME ?? '',
    MAKEFUN_ENV_URL: envUrl,
    MAKEFUN_PROMPT_FILE: prompt,
    MAKEFUN_TRIAL_SEED: seedOf(trial),
    MAKEFUN_TRIAL_NOW: TRIAL_NOW,
  };
  // --list first, as a checker would: it must print what the server registers.
  const list = await new Promise<string>((res) => {
    const p = spawn(NODE, [join(REPO, 'main.ts'), '--list'], { env: trialEnv, cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    p.stdout.on('data', (d) => (out += d));
    p.on('exit', () => res(out));
  });
  const started = performance.now();
  const server = spawn(NODE, [join(REPO, 'main.ts')], { env: trialEnv, cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
  let serverLog = '';
  server.stdout.on('data', (d) => (serverLog += d));
  server.stderr.on('data', (d) => (serverLog += d));
  const exitCode = await new Promise<number | null>((res) => {
    const timer = setTimeout(() => {
      serverLog += '\n[harness] the 180 s wall ran out: killing the server\n';
      server.kill('SIGKILL');
    }, 180000);
    server.on('exit', (code) => {
      clearTimeout(timer);
      res(code);
    });
  });
  const wallMs = performance.now() - started;
  env.kill('SIGTERM');
  await new Promise((r) => env.on('exit', r));
  const exportFile = join(stateDir, 'export/state.json');
  const state = existsSync(exportFile) ? readFileSync(exportFile, 'utf8') : '';
  const scoresDir = join(dir, 'scores');
  const scorerOut = await new Promise<string>((res) => {
    const p = spawn(NODE, [join(KIT, 'program/scorer/score.mjs')], {
      env: { ...process.env, MAKEFUN_OUTPUT_DIR: join(stateDir, 'export'), MAKEFUN_REFERENCE_DIR: reference, MAKEFUN_SCORES_DIR: scoresDir },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    p.stdout.on('data', (d) => (out += d));
    p.stderr.on('data', (d) => (out += d));
    p.on('exit', () => res(out));
  });
  let scores: Record<string, number> = {};
  try {
    scores = JSON.parse(readFileSync(join(scoresDir, 'scores.json'), 'utf8'));
  } catch {
    process.stdout.write(`scorer: ${scorerOut}\n`);
  }
  if (!KEEP) rmSync(dir, { recursive: true, force: true });
  else process.stdout.write(`kept ${dir}\n`);
  return { scores, state, list, serverLog, exitCode, wallMs };
}

function report(trial: number, run: Run): void {
  const s = run.scores;
  process.stdout.write(
    `trial-${trial}: score ${s.score} | quality ${s.quality} served ${s.served} wire ${s.wire} failover ${s.failover} catalog ${s.catalog} | calls ${s.upstream_calls} working ${s.working} | exit ${run.exitCode} in ${(run.wallMs / 1000).toFixed(1)} s\n`,
  );
  let state: Record<string, unknown> = {};
  try {
    state = JSON.parse(run.state);
  } catch {
    process.stdout.write('  no state export\n');
    return;
  }
  const catalog = JSON.stringify(state.catalog);
  let listed = '';
  try {
    listed = JSON.stringify(JSON.parse(run.list));
  } catch {
    listed = '(not JSON)';
  }
  process.stdout.write(`  --list matches the registered catalog: ${listed === catalog ? 'yes' : 'NO'}; catalog_route ${JSON.stringify(state.catalog_route)}\n`);
  const requests = Array.isArray(state.requests) ? (state.requests as Record<string, unknown>[]) : [];
  for (const r of requests) {
    const checks = (r.checks as { name: string; pass: boolean; detail: string }[]) ?? [];
    const failed = checks.filter((c) => !c.pass).map((c) => `${c.name}: ${c.detail}`);
    const failures = (r.failures as { provider: string; status: number; code: string }[]) ?? [];
    if (!QUIET || !r.valid || failures.length > 0)
      process.stdout.write(
        `  ${r.id} ${r.valid ? 'ok ' : 'BAD'} ${String(r.model ?? '-').padEnd(40)} ${String(r.rating_milli).padStart(4)} calls ${r.upstream_calls}${failures.length > 0 ? ` first-fail ${failures[0].provider} ${failures[0].status} ${failures[0].code}` : ''}${failed.length > 0 ? `\n      ${failed.join('\n      ')}` : ''}\n`,
      );
  }
}

const which = opt('--trial', 'all');
const trials = which === 'all' ? [0, 1, 2] : [Number(which)];
let port = 28080;
let allSame = true;
for (const t of trials) {
  const run = await runTrial(t, port++);
  report(t, run);
  if (flag('--log')) process.stdout.write(run.serverLog);
  if (flag('--twice')) {
    const again = await runTrial(t, port++);
    const same = again.state === run.state && again.state !== '';
    allSame &&= same;
    process.stdout.write(`  replay: the second pass left ${same ? 'the same' : 'a DIFFERENT'} state export (score ${again.scores.score}, calls ${again.scores.upstream_calls})\n`);
  }
}
if (flag('--twice')) process.stdout.write(`replay check: ${allSame ? 'every export matched' : 'MISMATCH'}\n`);
