// freeroute: a free AI model aggregator for make.fun's free-model-aggregator-01.
//
// One server, two client wires (OpenAI Chat Completions and Anthropic Messages, JSON and SSE), ten free
// provider tiers behind it on three wires (OpenAI, Anthropic, Gemini). At start it reads the brief, probes
// each provider's catalog route in its own wire, builds the catalog, listens, registers with the env and
// serves the client's requests one at a time, each on the best free model that meets it and still answers.
// It exits 0 once GET /work says done.
//
//   node main.ts          run the server (a trial sets MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED)
//   node main.ts --list   print the catalog it registers, byte for byte, and exit 0
//
// Determinism: every choice is a function of the brief, the requests and the providers' answers. No wall
// clock and no random source reaches anything we send; timers only wait.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { authHeaders, dialBase, isRecord, readBrief, type ClientApi, type Json, type ManifestEntry, type Requires } from './lib/brief.ts';
import { freeModelIds, loadCatalog, type Catalog } from './lib/catalog.ts';
import { Router, type Candidate, type WireNeeds } from './lib/router.ts';
import { clientError, clientReply, readAnswer, readClient, upstreamCall, type Reply } from './lib/wire.ts';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const POLL_MS = 150;
/** One provider call's longest wait, and the whole request's (the env's client waits 30 s). */
const UPSTREAM_TIMEOUT_MS = 12000;
const REQUEST_BUDGET_MS = 26000;
const MAX_BODY_BYTES = 16 * 1024 * 1024;
/** The port we try first; the next ones in turn if it is taken (never 8080, the env's own). */
const PREFERRED_PORT = 18787;

function log(message: string): void {
  process.stderr.write(`freeroute: ${message}\n`);
}

/** The trial's fixed now as Unix seconds (MAKEFUN_TRIAL_NOW), for a completion's `created`; 0 without one. */
function trialCreated(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW ?? '';
  if (/^\d+$/.test(raw)) {
    const n = Number(raw);
    return Math.floor(n > 1e11 ? n / 1000 : n);
  }
  const t = Date.parse(raw);
  return Number.isFinite(t) ? Math.floor(t / 1000) : 0;
}

const brief = readBrief(PROMPT_FILE);
for (const e of brief.errors) log(e);
let catalog: Catalog = { providers: [] };
let catalogText = '{"providers":[]}';
let router: Router;
const CREATED = trialCreated();

// ---------------------------------------------------------------------------------------------------
// The client's requests

/** How many inference requests have arrived: the client sends the manifest in order, one at a time. */
let arrivals = 0;
/** One request at a time reaches the providers, whatever the client does. */
let queue: Promise<void> = Promise.resolve();

async function readBody(req: IncomingMessage): Promise<{ ok: true; value: unknown } | { ok: false }> {
  const parts: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const b = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += b.length;
    if (size > MAX_BODY_BYTES) return { ok: false };
    parts.push(b);
  }
  try {
    return { ok: true, value: JSON.parse(Buffer.concat(parts).toString('utf8')) };
  } catch {
    return { ok: false };
  }
}

function send(res: ServerResponse, reply: Reply): void {
  const body = Buffer.from(reply.body, 'utf8');
  const headers: Record<string, string> = { 'content-type': reply.contentType, 'content-length': String(body.length) };
  if (reply.contentType.startsWith('text/event-stream')) {
    headers['cache-control'] = 'no-cache';
    headers.connection = 'keep-alive';
  }
  res.writeHead(reply.status, headers);
  res.end(body);
}

/**
 * The manifest row of the request the client is sending now: GET /work names it while it is in flight;
 * else the arrival order does. A row whose API or streaming disagrees with the request itself is not it.
 */
async function currentEntry(index: number, api: ClientApi, stream: boolean): Promise<ManifestEntry | null> {
  let id: string | null = null;
  if (ENV_URL !== '') {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(3000) });
      const work: unknown = await res.json();
      if (isRecord(work) && work.done !== true && typeof work.next === 'string') id = work.next;
    } catch {
      id = null;
    }
  }
  const fits = (m: ManifestEntry | undefined): m is ManifestEntry => m !== undefined && m.api === api && m.stream === stream;
  const byId = id === null ? undefined : brief.manifest.find((m) => m.id === id);
  if (fits(byId)) return byId;
  const byOrder = brief.manifest[index];
  if (fits(byOrder)) return byOrder;
  return byId ?? byOrder ?? null;
}

/** What the body itself asks of a model, beyond the manifest's word. */
function bodyRequires(api: ClientApi, body: Json): { requires: Requires; needs: WireNeeds } {
  const n = readClient(api, body);
  const requires: Requires = {};
  if (n.tools.length > 0) requires.tools = true;
  if (n.messages.some((m) => m.parts.some((p) => p.kind === 'image'))) requires.vision = true;
  if (n.jsonMode !== null) requires.json = true;
  // A conservative floor on the context the prompt needs (three characters a token, the tool schemas
  // included, plus the output cap): only a prompt far past the smallest window narrows the choice.
  let chars = n.system.length;
  for (const m of n.messages) {
    for (const p of m.parts) chars += p.kind === 'text' ? p.text.length : 0;
    for (const c of m.toolCalls) chars += c.name.length + c.args.length;
  }
  for (const t of n.tools) chars += t.name.length + (t.description?.length ?? 0) + JSON.stringify(t.parameters ?? {}).length;
  const estimate = Math.ceil(chars / 3) + (n.maxTokens ?? 0);
  if (estimate > 16384) requires.min_context = estimate;
  // What only the client's own wire carries through intact (each wire's parser reads these differently):
  // a turn that ends on a tool result, a developer message beside a system one, an Anthropic server tool.
  const last = n.messages[n.messages.length - 1];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const lastRaw = messages[messages.length - 1];
  const endsOnToolResult =
    last?.role === 'tool' || (api === 'anthropic' && isRecord(lastRaw) && Array.isArray(lastRaw.content) && lastRaw.content.some((b) => isRecord(b) && b.type === 'tool_result'));
  const developerAndSystem = api === 'openai' && messages.some((m) => isRecord(m) && m.role === 'developer') && messages.some((m) => isRecord(m) && m.role === 'system');
  const serverTools = api === 'anthropic' && Array.isArray(body.tools) && body.tools.some((t) => isRecord(t) && typeof t.type === 'string' && t.type !== 'custom');
  const sameWire = endsOnToolResult || developerAndSystem || serverTools ? api : null;
  return { requires, needs: { jsonMode: api === 'openai' && n.jsonMode !== null, noMaxTokens: n.maxTokens === null, sameWire } };
}

function mergeRequires(a: Requires, b: Requires): Requires {
  const out: Requires = { ...a };
  if (b.tools === true) out.tools = true;
  if (b.vision === true) out.vision = true;
  if (b.json === true) out.json = true;
  if (b.min_context !== undefined) out.min_context = Math.max(out.min_context ?? 0, b.min_context);
  return out;
}

interface CallResult {
  status: number;
  code: string;
  body: unknown;
  error: string | null;
}

function errorCode(body: unknown): string {
  if (!isRecord(body)) return '';
  const e = body.error;
  if (isRecord(e)) {
    if (typeof e.code === 'string') return e.code;
    if (typeof e.status === 'string') return e.status;
    if (typeof e.type === 'string') return e.type;
  }
  return '';
}

function errorMessage(body: unknown): string {
  if (!isRecord(body)) return '';
  const e = body.error;
  if (isRecord(e) && typeof e.message === 'string') return e.message;
  if (typeof body.message === 'string') return body.message;
  return '';
}

/** One provider call. The env's answer, parsed; status 0 when no answer came back in `timeoutMs`. */
async function callProvider(c: Candidate, path: string, body: Json, api: ClientApi, headersIn: IncomingMessage['headers'], timeoutMs: number): Promise<CallResult> {
  const p = c.provider;
  const url = `${dialBase(ENV_URL, p.path)}${path}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: 'application/json', ...authHeaders(p) };
  if (p.wire === 'anthropic') {
    const v = api === 'anthropic' && typeof headersIn['anthropic-version'] === 'string' ? headersIn['anthropic-version'] : '2023-06-01';
    headers['anthropic-version'] = v;
    const beta = headersIn['anthropic-beta'];
    if (api === 'anthropic' && typeof beta === 'string' && beta !== '') headers['anthropic-beta'] = beta;
  }
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs) });
    const text = await res.text();
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
    return { status: res.status, code: res.status >= 200 && res.status < 300 ? '' : errorCode(parsed), body: parsed, error: null };
  } catch (err) {
    return { status: 0, code: 'network', body: null, error: (err as Error).message };
  }
}

async function serveInference(api: ClientApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const parsed = await readBody(req);
  const index = arrivals;
  arrivals += 1;
  if (!parsed.ok || !isRecord(parsed.value)) {
    send(res, clientError(api, 400, 'invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const body = parsed.value;
  const stream = body.stream === true;
  const entry = await currentEntry(index, api, stream);
  const fromBody = bodyRequires(api, body);
  const requires = mergeRequires(entry?.requires ?? {}, fromBody.requires);
  const tag = entry?.id ?? `#${index + 1}`;
  // Providers to pass over for this request only: one still busy with a call (concurrent_requests).
  const skip = new Set<string>();
  // Providers that refused the request itself (a 4xx about the body, not the model): two of them end it.
  const rejectedBody = new Set<string>();
  // One deadline for the whole request, inside the client's own 30 s wait; and once the client has gone,
  // no further provider call is made for it (its calls would be counted against the next request).
  const deadline = performance.now() + REQUEST_BUDGET_MS;
  let clientGone = false;
  res.on('close', () => {
    if (!res.writableEnded) clientGone = true;
  });
  let lastStatus = 503;
  let lastCode = 'no_provider';
  for (let attempt = 0; attempt < 24; attempt++) {
    const left = deadline - performance.now();
    if (clientGone || left < 500) break;
    const c = router.choose(entry, requires, fromBody.needs, skip);
    if (c === undefined) break;
    const call = upstreamCall(api, body, c.provider.wire, c.model);
    const result = await callProvider(c, call.path, call.body, api, req.headers, Math.min(UPSTREAM_TIMEOUT_MS, left));
    const ok = result.status >= 200 && result.status < 300;
    const answer = ok ? readAnswer(c.provider.wire, result.body, c.model, `call_${tag.replace(/[^A-Za-z0-9_-]/g, '')}_${attempt}`) : null;
    const message = ok ? '' : errorMessage(result.body);
    router.record({ provider: c.provider.id, model: c.model, ok: ok && answer !== null, status: result.status, code: result.code, message });
    if (result.status === 429 && result.code === 'concurrent_requests') skip.add(c.provider.id);
    if ((result.status === 400 || result.status === 413 || result.status === 422) && !Router.modelLevel(result.code, message)) {
      rejectedBody.add(c.provider.id);
      if (rejectedBody.size >= 2) {
        log(`${tag}: two providers refused the request body itself (${message}); answering 400`);
        send(res, clientError(api, 400, result.code || 'invalid_request_error', message || 'the request was refused by the providers'));
        return;
      }
      skip.add(c.provider.id);
    }
    log(`${tag} ${api}${stream ? ' stream' : ''} -> ${c.provider.id}/${c.model}: ${result.status}${result.code === '' ? '' : ` ${result.code}`}`);
    if (ok && answer !== null) {
      router.served(entry, c);
      // The provider's own id when it speaks the client's wire; else one in the client's id shape, made
      // from the request's manifest id (never a clock or a random draw).
      const prefix = api === 'openai' ? 'chatcmpl-' : 'msg_';
      const ownId = c.provider.wire === api && answer.id !== null ? answer.id : null;
      const ids = { id: ownId ?? `${prefix}freeroute-${tag.replace(/[^A-Za-z0-9_-]/g, '')}`, created: CREATED };
      send(res, clientReply(api, stream, c.provider.wire, result.body, answer, ids));
      return;
    }
    if (ok) {
      lastStatus = 502;
      lastCode = 'bad_upstream_answer';
    } else {
      lastStatus = result.status === 0 ? 504 : result.status;
      lastCode = result.code || 'upstream_error';
    }
  }
  const status = lastStatus === 402 || lastStatus === 429 ? lastStatus : lastStatus >= 500 ? lastStatus : 503;
  send(res, clientError(api, status, lastCode, 'every free provider that could serve this request refused it'));
}

function modelList(wire: ClientApi): Json {
  const models = freeModelIds(catalog);
  if (wire === 'openai') {
    return { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: CREATED, owned_by: m.provider })) };
  }
  const createdAt = new Date(CREATED * 1000).toISOString().replace('.000Z', 'Z');
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: createdAt })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** The wire a request's auth header names: x-api-key is Anthropic, Authorization: Bearer is OpenAI. */
function wireOfHeaders(h: IncomingMessage['headers']): ClientApi {
  if (typeof h['x-api-key'] === 'string') return 'anthropic';
  if (typeof h.authorization === 'string') return 'openai';
  if (typeof h['anthropic-version'] === 'string') return 'anthropic';
  return 'openai';
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const path = url.pathname.replace(/\/+$/, '') || '/';
  const method = req.method ?? 'GET';
  if (path === '/v1/chat/completions' || path === '/chat/completions') {
    if (method !== 'POST') return send(res, clientError('openai', 405, 'method_not_allowed', `${path} takes POST`));
    return serveQueued('openai', req, res);
  }
  if (path === '/v1/messages' || path === '/messages') {
    if (method !== 'POST') return send(res, clientError('anthropic', 405, 'method_not_allowed', `${path} takes POST`));
    return serveQueued('anthropic', req, res);
  }
  if (path === '/v1/models' || path === '/models') {
    const wire = wireOfHeaders(req.headers);
    for await (const _ of req) void _;
    return send(res, { status: 200, contentType: 'application/json', body: JSON.stringify(modelList(wire)) });
  }
  if (path.startsWith('/v1/models/')) {
    const wire = wireOfHeaders(req.headers);
    const id = decodeURIComponent(path.slice('/v1/models/'.length));
    const m = freeModelIds(catalog).find((x) => x.id === id);
    for await (const _ of req) void _;
    if (m === undefined) return send(res, clientError(wire, 404, 'model_not_found', `no free model ${id}`));
    const list = modelList(wire);
    const item = (list.data as Json[]).find((x) => x.id === id);
    return send(res, { status: 200, contentType: 'application/json', body: JSON.stringify(item) });
  }
  for await (const _ of req) void _;
  const wire = path.startsWith('/v1/messages') ? 'anthropic' : wireOfHeaders(req.headers);
  return send(res, clientError(wire, 404, 'not_found', `no route ${method} ${path}`));
}

function serveQueued(api: ClientApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const run = queue.then(() => serveInference(api, req, res));
  queue = run.catch(() => undefined);
  return run;
}

// ---------------------------------------------------------------------------------------------------
// Start, register, poll

function listen(server: ReturnType<typeof createServer>, port: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const onError = (err: Error): void => {
      server.removeListener('listening', onListening);
      reject(err);
    };
    const onListening = (): void => {
      server.removeListener('error', onError);
      const a = server.address();
      resolve(a !== null && typeof a === 'object' ? a.port : port);
    };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port, '0.0.0.0');
  });
}

async function listenSomewhere(server: ReturnType<typeof createServer>): Promise<number> {
  for (let i = 0; i < 20; i++) {
    try {
      return await listen(server, PREFERRED_PORT + i);
    } catch {
      // taken: try the next one
    }
  }
  return listen(server, 0);
}

async function register(port: number): Promise<void> {
  const body = `{"protocol_version":${PROTOCOL_VERSION},"port":${port},"catalog":${catalogText}}`;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, { method: 'POST', headers: { 'content-type': 'application/json' }, body, signal: AbortSignal.timeout(10000) });
      const text = await res.text();
      log(`register: HTTP ${res.status} ${text.trim()}`);
      if (res.ok) return;
      if (res.status >= 400 && res.status < 500) return;
    } catch (err) {
      log(`register failed: ${(err as Error).message}`);
    }
    await new Promise((r) => setTimeout(r, 300));
  }
}

function poll(): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(3000) })
      .then((res) => res.json())
      .then((work: unknown) => {
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          log('the env says the workload is done: exiting');
          // Let the last answer's socket flush, then leave.
          queue.finally(() => setTimeout(() => process.exit(0), 50));
        }
      })
      .catch(() => undefined)
      .finally(() => {
        busy = false;
      });
  }, POLL_MS);
}

async function main(): Promise<void> {
  catalog = await loadCatalog(ENV_URL, brief);
  catalogText = JSON.stringify(catalog);
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(`${catalogText}\n`, () => resolve()));
    process.exit(0);
  }
  router = new Router(brief, catalog, undefined, process.env.MAKEFUN_TRIAL_SEED ?? '');
  router.prefer = process.env.FREEROUTE_PREFER ?? '';
  if (ENV_URL === '') {
    log('MAKEFUN_ENV_URL is not set: nothing to serve');
    process.exit(0);
  }
  const server = createServer((req, res) => {
    handle(req, res).catch((err: unknown) => {
      log(`handler failed: ${(err as Error).stack ?? String(err)}`);
      if (!res.headersSent) send(res, clientError((req.url ?? '').startsWith('/v1/messages') ? 'anthropic' : 'openai', 500, 'internal_error', 'the aggregator failed to answer'));
    });
  });
  server.keepAliveTimeout = 5000;
  const port = await listenSomewhere(server);
  log(`listening on 0.0.0.0:${port}; ${catalog.providers.length} providers, ${freeModelIds(catalog).length} free model ids`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll();
}

main().catch((err: unknown) => {
  log((err as Error).stack ?? String(err));
  process.exit(1);
});
