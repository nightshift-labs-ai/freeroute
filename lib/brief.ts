// The trial brief: the provider directory, the quality table and the workload manifest.
//
// The brief ($MAKEFUN_PROMPT_FILE) is prose with three JSON documents in it, each tagged by its "schema".
// We pull every JSON object that carries a string "schema" out of the text and keep the three we know.
// Nothing here reads a clock or a random source: the brief alone decides what comes out.

import { readFileSync } from 'node:fs';

export type Json = Record<string, unknown>;
export type Wire = 'openai' | 'anthropic' | 'gemini';
export type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type ClientApi = 'openai' | 'anthropic';

export interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  /** The provider's own base path (e.g. /p/groq): what the catalog reports; join it to the env to dial. */
  path: string;
}

export interface ModelRating {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestEntry {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requires;
}

export interface Brief {
  providers: ProviderInfo[];
  ratings: ModelRating[];
  best: Map<string, number>;
  manifest: ManifestEntry[];
  errors: string[];
}

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** The index of the `}` that closes the object opening at `start`, or -1 (string contents skipped). */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`, in order of appearance. */
export function schemaDocuments(text: string): Json[] {
  const out: Json[] = [];
  let at = text.indexOf('{');
  while (at !== -1) {
    const end = objectEnd(text, at);
    if (end === -1) break;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text.slice(at, end + 1));
    } catch {
      parsed = undefined;
    }
    if (isRecord(parsed) && typeof parsed.schema === 'string') {
      out.push(parsed);
      at = text.indexOf('{', end + 1);
    } else {
      at = text.indexOf('{', at + 1);
    }
  }
  return out;
}

const str = (v: unknown, dflt = ''): string => (typeof v === 'string' ? v : dflt);
const bool = (v: unknown): boolean => v === true;
const intOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): Auth {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' ? v : 'none';
}

function requiresOf(v: unknown): Requires {
  const out: Requires = {};
  if (!isRecord(v)) return out;
  if (v.tools === true) out.tools = true;
  if (v.vision === true) out.vision = true;
  if (v.json === true) out.json = true;
  if (typeof v.min_context === 'number' && Number.isFinite(v.min_context)) out.min_context = v.min_context;
  return out;
}

export function parseBrief(text: string): Brief {
  const brief: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], errors: [] };
  for (const doc of schemaDocuments(text)) {
    const schema = str(doc.schema);
    if (schema.endsWith('/providers/v1') && Array.isArray(doc.providers)) {
      for (const p of doc.providers) {
        if (!isRecord(p) || typeof p.id !== 'string' || p.id === '') continue;
        const wire = wireOf(p.wire);
        if (wire === null) {
          brief.errors.push(`provider ${p.id} names no known wire`);
          continue;
        }
        if (brief.providers.some((q) => q.id === p.id)) continue;
        brief.providers.push({
          id: p.id,
          name: str(p.name, p.id),
          wire,
          auth: authOf(p.auth),
          key_env: str(p.key_env),
          key: str(p.key),
          path: str(p.path, str(p.base_url, `/p/${p.id}`)),
        });
      }
    } else if (schema.endsWith('/ratings/v1') && Array.isArray(doc.ratings)) {
      for (const r of doc.ratings) {
        if (!isRecord(r) || typeof r.model !== 'string' || typeof r.provider !== 'string') continue;
        brief.ratings.push({
          model: r.model,
          provider: r.provider,
          rating_milli: intOrNull(r.rating_milli) ?? 0,
          context_length: intOrNull(r.context_length),
          tools: bool(r.tools),
          vision: bool(r.vision),
          json: bool(r.json),
          free: r.free !== false,
        });
      }
      if (Array.isArray(doc.best)) {
        for (const b of doc.best) {
          if (isRecord(b) && typeof b.id === 'string' && typeof b.rating_milli === 'number') brief.best.set(b.id, b.rating_milli);
        }
      }
    } else if (schema.endsWith('/workload/v1') && Array.isArray(doc.manifest)) {
      for (const m of doc.manifest) {
        if (!isRecord(m) || typeof m.id !== 'string') continue;
        brief.manifest.push({
          id: m.id,
          api: m.api === 'anthropic' ? 'anthropic' : 'openai',
          stream: m.stream === true,
          weight: typeof m.weight === 'number' && m.weight > 0 ? m.weight : 1,
          requires: requiresOf(m.requires),
        });
      }
    }
  }
  if (brief.providers.length === 0) brief.errors.push('the brief holds no provider directory');
  return brief;
}

export function readBrief(file: string): Brief {
  if (file === '') return parseBrief('');
  try {
    return parseBrief(readFileSync(file, 'utf8'));
  } catch (err) {
    const brief = parseBrief('');
    brief.errors.push(`cannot read the brief ${file}: ${(err as Error).message}`);
    return brief;
  }
}

/** Does a model with these capabilities meet the request's requirements? */
export function meets(model: { tools: boolean; vision: boolean; json: boolean; context_length: number | null }, requires: Requires): boolean {
  if (requires.tools === true && !model.tools) return false;
  if (requires.vision === true && !model.vision) return false;
  if (requires.json === true && !model.json) return false;
  if (requires.min_context !== undefined && (model.context_length === null || model.context_length < requires.min_context)) return false;
  return true;
}

/** The headers that carry a provider's key, by its auth scheme (the wire never decides the header). */
export function authHeaders(p: ProviderInfo): Record<string, string> {
  if (p.auth === 'none' || p.key === '') return {};
  if (p.auth === 'bearer') return { authorization: `Bearer ${p.key}` };
  if (p.auth === 'x-api-key') return { 'x-api-key': p.key };
  return { 'x-goog-api-key': p.key };
}

/** Join a provider's base path to the env's origin (an absolute base URL is used as it is). */
export function dialBase(envUrl: string, path: string): string {
  const base = path.trim();
  if (/^https?:\/\//i.test(base)) return base.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  return `${origin}${base.startsWith('/') ? '' : '/'}${base.replace(/\/+$/, '')}`;
}
