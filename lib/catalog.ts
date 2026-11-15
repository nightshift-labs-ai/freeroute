// The catalog: every provider, its base path, key variable and wire, and each model its own catalog
// route lists with the right free flag. The providers' listings are the truth; the brief's quality table
// (free models only) fills in when a listing cannot be read.

import { authHeaders, dialBase, isRecord, type Brief, type Json, type ProviderInfo } from './brief.ts';

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface ListedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
}

const PROBE_TIMEOUT_MS = 5000;

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? Math.floor(v) : null);

/** The models one catalog answer lists, in the provider's own order, read in its wire's shape. */
export function listedModels(wire: string, body: unknown): ListedModel[] | null {
  if (!isRecord(body)) return null;
  const out: ListedModel[] = [];
  if (wire === 'gemini') {
    if (!Array.isArray(body.models)) return null;
    for (const m of body.models) {
      if (!isRecord(m)) continue;
      const name = typeof m.name === 'string' ? m.name : typeof m.id === 'string' ? m.id : '';
      const id = name.startsWith('models/') ? name.slice('models/'.length) : name;
      if (id === '') continue;
      out.push({ id, free: typeof m.free === 'boolean' ? m.free : null, context_length: num(m.context_length) ?? num(m.inputTokenLimit) });
    }
    return out;
  }
  if (!Array.isArray(body.data)) return null;
  for (const m of body.data) {
    if (!isRecord(m) || typeof m.id !== 'string' || m.id === '') continue;
    const ctx = num(m.context_length) ?? num(m.context_window) ?? num(m.max_input_tokens) ?? (isRecord(m.top_provider) ? num(m.top_provider.context_length) : null);
    out.push({ id: m.id, free: typeof m.free === 'boolean' ? m.free : null, context_length: ctx });
  }
  return out;
}

/** GET the provider's catalog route in its own wire; null when it cannot be read. */
export async function probeProvider(envUrl: string, p: ProviderInfo): Promise<ListedModel[] | null> {
  if (envUrl === '') return null;
  const base = dialBase(envUrl, p.path);
  const url = p.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(p) };
  if (p.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const res = await fetch(url, { headers, signal: controller.signal });
    const text = await res.text();
    if (res.status < 200 || res.status >= 300) return null;
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    return listedModels(p.wire, body);
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The one catalog document: providers in the brief's order, each provider's models in its listing's
 * order. A model is free when its listing says so; a listing that carries no free flag defers to the
 * brief's quality table, which holds free models only.
 */
export function buildCatalog(brief: Brief, listings: (ListedModel[] | null)[]): Catalog {
  const providers: CatalogProvider[] = [];
  brief.providers.forEach((p, i) => {
    const rows = brief.ratings.filter((r) => r.provider === p.id);
    const models: CatalogModel[] = [];
    const seen = new Set<string>();
    const listing = listings[i] ?? null;
    if (listing !== null) {
      for (const m of listing) {
        if (seen.has(m.id)) continue;
        seen.add(m.id);
        const row = rows.find((r) => r.model === m.id);
        const free = m.free !== null ? m.free : row !== undefined && row.free;
        models.push({ id: m.id, free, context_length: m.context_length ?? row?.context_length ?? null });
      }
    } else {
      for (const row of rows) {
        if (seen.has(row.model)) continue;
        seen.add(row.model);
        models.push({ id: row.model, free: row.free, context_length: row.context_length });
      }
    }
    providers.push({ id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models });
  });
  return { providers };
}

export async function loadCatalog(envUrl: string, brief: Brief): Promise<Catalog> {
  const listings: (ListedModel[] | null)[] = [];
  // One probe at a time, in the brief's order: the same answers always build the same document.
  for (const p of brief.providers) listings.push(await probeProvider(envUrl, p));
  return buildCatalog(brief, listings);
}

/** Every free model id the catalog lists, first provider first, each id once. */
export function freeModelIds(catalog: Catalog): { id: string; provider: string }[] {
  const out: { id: string; provider: string }[] = [];
  const seen = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (!m.free || seen.has(m.id)) continue;
      seen.add(m.id);
      out.push({ id: m.id, provider: p.id });
    }
  }
  return out;
}

export function isFreeIn(catalog: Catalog, provider: string, model: string): boolean {
  const p = catalog.providers.find((x) => x.id === provider);
  return p !== undefined && p.models.some((m) => m.id === model && m.free);
}

export type { Json };
