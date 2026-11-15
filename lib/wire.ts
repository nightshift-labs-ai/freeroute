// Wire translation between the OpenAI Chat Completions, Anthropic Messages and Gemini generateContent APIs.
//
// A client request is read into one neutral form (system text, messages, tools, limits, JSON mode), then
// written out in the provider's wire. A provider answer is read into one neutral answer (text, tool calls,
// stop reason, usage), then written back in the client's wire, as JSON or as server-sent events.
//
// The provider echoes what it parsed and the client compares that with what it sent, so the translation
// keeps every field the parsers read: the system text (joined with "\n" across segments, as each wire's
// parser joins it), the last user turn's text and image bytes, the tool names in order, max_tokens, the
// stop sequences and JSON mode. Nothing here reads a clock or a random source.

import { isRecord, type ClientApi, type Json, type Wire } from './brief.ts';

// ---------------------------------------------------------------------------------------------------
// The neutral request

export interface TextPart {
  kind: 'text';
  text: string;
}
export interface ImagePart {
  kind: 'image';
  /** base64 bytes and their media type, for an inline image */
  data: string | null;
  mediaType: string | null;
  /** a remote image's URL */
  url: string | null;
}
export type Part = TextPart | ImagePart;

export interface ToolCall {
  id: string;
  name: string;
  /** the arguments as a JSON text */
  args: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
  /** true when the client sent this turn's content as one plain string */
  plain: boolean;
  toolCalls: ToolCall[];
  toolCallId: string | null;
  isError: boolean;
}

export interface ToolDef {
  name: string;
  description: string | null;
  parameters: Json | null;
}

export type ToolChoice = null | 'auto' | 'none' | 'required' | { name: string };

export interface JsonMode {
  type: 'json_object' | 'json_schema';
  schema: Json | null;
  name: string | null;
}

export interface Neutral {
  system: string;
  hasSystem: boolean;
  messages: Message[];
  tools: ToolDef[];
  toolChoice: ToolChoice;
  maxTokens: number | null;
  stops: string[] | null;
  temperature: number | null;
  topP: number | null;
  topK: number | null;
  jsonMode: JsonMode | null;
  parallelToolCalls: boolean | null;
}

const finite = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function stringList(v: unknown): string[] | null {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return null;
}

/** The text blocks of a content value joined with "\n" (a string is its own text). */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}

/** Split a data: URL into its media type and base64 payload; null when it is not a base64 data URL. */
function dataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^,]*?);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  // The media type alone: parameters such as ;name=... are not part of it.
  const mediaType = m[1].split(';')[0].trim();
  return { mediaType: mediaType === '' ? 'application/octet-stream' : mediaType, data: m[2] };
}

function imageFromUrl(url: string): ImagePart {
  const d = dataUrl(url);
  if (d !== null) return { kind: 'image', data: d.data, mediaType: d.mediaType, url: null };
  return { kind: 'image', data: null, mediaType: null, url };
}

function newMessage(role: Message['role']): Message {
  return { role, parts: [], plain: false, toolCalls: [], toolCallId: null, isError: false };
}

function argsText(v: unknown): string {
  if (typeof v === 'string') return v;
  if (v === undefined) return '{}';
  return JSON.stringify(v);
}

function openAIParts(content: unknown, msg: Message): void {
  if (typeof content === 'string') {
    msg.parts.push({ kind: 'text', text: content });
    msg.plain = true;
    return;
  }
  if (!Array.isArray(content)) return;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') msg.parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image_url') {
      const url = isRecord(block.image_url) ? block.image_url.url : block.image_url;
      if (typeof url === 'string' && url !== '') msg.parts.push(imageFromUrl(url));
    }
  }
}

/** Read an OpenAI Chat Completions request. */
export function fromOpenAI(body: Json): Neutral {
  const n: Neutral = {
    system: '',
    hasSystem: false,
    messages: [],
    tools: [],
    toolChoice: null,
    maxTokens: finite(body.max_tokens) ?? finite(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: finite(body.temperature),
    topP: finite(body.top_p),
    topK: null,
    jsonMode: null,
    parallelToolCalls: typeof body.parallel_tool_calls === 'boolean' ? body.parallel_tool_calls : null,
  };
  const systemParts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system' || m.role === 'developer') {
      const text = blocksText(m.content);
      if (text !== '') systemParts.push(text);
    } else if (m.role === 'user') {
      const msg = newMessage('user');
      openAIParts(m.content, msg);
      n.messages.push(msg);
    } else if (m.role === 'assistant') {
      const msg = newMessage('assistant');
      openAIParts(m.content, msg);
      if (Array.isArray(m.tool_calls)) {
        for (const c of m.tool_calls) {
          if (!isRecord(c) || !isRecord(c.function) || typeof c.function.name !== 'string') continue;
          msg.toolCalls.push({ id: typeof c.id === 'string' ? c.id : `call_${msg.toolCalls.length}`, name: c.function.name, args: argsText(c.function.arguments) });
        }
      } else if (isRecord(m.function_call) && typeof m.function_call.name === 'string') {
        msg.toolCalls.push({ id: 'call_0', name: m.function_call.name, args: argsText(m.function_call.arguments) });
      }
      n.messages.push(msg);
    } else if (m.role === 'tool' || m.role === 'function') {
      const msg = newMessage('tool');
      msg.parts.push({ kind: 'text', text: blocksText(m.content) });
      msg.plain = typeof m.content === 'string';
      msg.toolCallId = typeof m.tool_call_id === 'string' ? m.tool_call_id : typeof m.name === 'string' ? m.name : null;
      n.messages.push(msg);
    }
  }
  n.system = systemParts.join('\n');
  n.hasSystem = systemParts.length > 0;
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') continue;
      n.tools.push({
        name: t.function.name,
        description: typeof t.function.description === 'string' ? t.function.description : null,
        parameters: isRecord(t.function.parameters) ? t.function.parameters : null,
      });
    }
  } else if (Array.isArray(body.functions)) {
    for (const f of body.functions) {
      if (!isRecord(f) || typeof f.name !== 'string') continue;
      n.tools.push({ name: f.name, description: typeof f.description === 'string' ? f.description : null, parameters: isRecord(f.parameters) ? f.parameters : null });
    }
  }
  const tc = body.tool_choice ?? body.function_call;
  if (tc === 'auto' || tc === 'none' || tc === 'required') n.toolChoice = tc;
  else if (isRecord(tc)) {
    const name = isRecord(tc.function) ? tc.function.name : tc.name;
    if (typeof name === 'string') n.toolChoice = { name };
  }
  if (isRecord(body.response_format)) {
    const rf = body.response_format;
    if (rf.type === 'json_object') n.jsonMode = { type: 'json_object', schema: null, name: null };
    else if (rf.type === 'json_schema') {
      const js = isRecord(rf.json_schema) ? rf.json_schema : {};
      n.jsonMode = { type: 'json_schema', schema: isRecord(js.schema) ? js.schema : null, name: typeof js.name === 'string' ? js.name : null };
    }
  }
  return n;
}

/** Read an Anthropic Messages request. */
export function fromAnthropic(body: Json): Neutral {
  const system = blocksText(body.system);
  const n: Neutral = {
    system,
    hasSystem: system !== '',
    messages: [],
    tools: [],
    toolChoice: null,
    maxTokens: finite(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: finite(body.temperature),
    topP: finite(body.top_p),
    topK: finite(body.top_k),
    jsonMode: null,
    parallelToolCalls: null,
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = m.role === 'assistant' ? 'assistant' : m.role === 'user' ? 'user' : null;
    if (role === null) continue;
    if (typeof m.content === 'string') {
      const msg = newMessage(role);
      msg.parts.push({ kind: 'text', text: m.content });
      msg.plain = true;
      n.messages.push(msg);
      continue;
    }
    if (!Array.isArray(m.content)) {
      n.messages.push(newMessage(role));
      continue;
    }
    let msg = newMessage(role);
    const flush = (): void => {
      if (msg.parts.length > 0 || msg.toolCalls.length > 0) n.messages.push(msg);
      msg = newMessage(role);
    };
    let pushedAny = false;
    for (const block of m.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') msg.parts.push({ kind: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          msg.parts.push({ kind: 'image', data: src.data, mediaType: typeof src.media_type === 'string' ? src.media_type : 'image/png', url: null });
        } else if (src.type === 'url' && typeof src.url === 'string') {
          msg.parts.push(imageFromUrl(src.url));
        }
      } else if (block.type === 'tool_use' && role === 'assistant' && typeof block.name === 'string') {
        msg.toolCalls.push({ id: typeof block.id === 'string' ? block.id : `toolu_${msg.toolCalls.length}`, name: block.name, args: argsText(block.input ?? {}) });
      } else if (block.type === 'tool_result' && role === 'user') {
        // A tool result is its own turn on the OpenAI and Gemini wires.
        flush();
        const t = newMessage('tool');
        t.parts.push({ kind: 'text', text: blocksText(block.content) });
        t.plain = typeof block.content === 'string';
        t.toolCallId = typeof block.tool_use_id === 'string' ? block.tool_use_id : null;
        t.isError = block.is_error === true;
        n.messages.push(t);
        pushedAny = true;
      }
    }
    if (msg.parts.length > 0 || msg.toolCalls.length > 0 || !pushedAny) n.messages.push(msg);
  }
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || typeof t.name !== 'string') continue;
      if (typeof t.type === 'string' && t.type !== 'custom') continue; // a server tool has no schema to carry
      n.tools.push({ name: t.name, description: typeof t.description === 'string' ? t.description : null, parameters: isRecord(t.input_schema) ? t.input_schema : null });
    }
  }
  if (isRecord(body.tool_choice)) {
    const tc = body.tool_choice;
    if (tc.type === 'auto') n.toolChoice = 'auto';
    else if (tc.type === 'any') n.toolChoice = 'required';
    else if (tc.type === 'none') n.toolChoice = 'none';
    else if (tc.type === 'tool' && typeof tc.name === 'string') n.toolChoice = { name: tc.name };
    if (tc.disable_parallel_tool_use === true) n.parallelToolCalls = false;
  }
  return n;
}

export function readClient(api: ClientApi, body: Json): Neutral {
  return api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
}

// ---------------------------------------------------------------------------------------------------
// Writing the provider request

function imageDataUrl(p: ImagePart): string {
  if (p.data !== null) return `data:${p.mediaType ?? 'image/png'};base64,${p.data}`;
  return p.url ?? '';
}

function textOf(parts: Part[]): string {
  return parts.filter((p): p is TextPart => p.kind === 'text').map((p) => p.text).join('');
}

function parseArgs(args: string): unknown {
  try {
    const v: unknown = JSON.parse(args);
    return isRecord(v) ? v : { value: v };
  } catch {
    return {};
  }
}

export function toOpenAI(n: Neutral, model: string): Json {
  const messages: Json[] = [];
  if (n.hasSystem) messages.push({ role: 'system', content: n.system });
  for (const m of n.messages) {
    if (m.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: m.toolCallId ?? '', content: textOf(m.parts) });
    } else if (m.role === 'user') {
      if (m.plain && m.parts.length === 1 && m.parts[0].kind === 'text') {
        messages.push({ role: 'user', content: m.parts[0].text });
      } else {
        messages.push({
          role: 'user',
          content: m.parts.map((p) => (p.kind === 'text' ? { type: 'text', text: p.text } : { type: 'image_url', image_url: { url: imageDataUrl(p) } })),
        });
      }
    } else {
      const text = textOf(m.parts);
      const out: Json = { role: 'assistant', content: text === '' && m.toolCalls.length > 0 ? null : text };
      if (m.toolCalls.length > 0) out.tool_calls = m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
      messages.push(out);
    }
  }
  const body: Json = { model, messages };
  if (n.maxTokens !== null) body.max_tokens = n.maxTokens;
  if (n.temperature !== null) body.temperature = n.temperature;
  if (n.topP !== null) body.top_p = n.topP;
  if (n.stops !== null && n.stops.length > 0) body.stop = n.stops;
  if (n.tools.length > 0) {
    body.tools = n.tools.map((t) => {
      const fn: Json = { name: t.name };
      if (t.description !== null) fn.description = t.description;
      fn.parameters = t.parameters ?? { type: 'object', properties: {} };
      return { type: 'function', function: fn };
    });
    if (n.toolChoice !== null) body.tool_choice = typeof n.toolChoice === 'string' ? n.toolChoice : { type: 'function', function: { name: n.toolChoice.name } };
    if (n.parallelToolCalls !== null) body.parallel_tool_calls = n.parallelToolCalls;
  }
  if (n.jsonMode !== null) {
    body.response_format =
      n.jsonMode.type === 'json_object'
        ? { type: 'json_object' }
        : { type: 'json_schema', json_schema: { name: n.jsonMode.name ?? 'response', schema: n.jsonMode.schema ?? { type: 'object' } } };
  }
  return body;
}

/** The output cap an Anthropic call must carry when the client named none. */
export const ANTHROPIC_DEFAULT_MAX_TOKENS = 4096;

export function toAnthropic(n: Neutral, model: string): Json {
  const messages: Json[] = [];
  const push = (role: 'user' | 'assistant', blocks: Json[]): void => {
    const last = messages[messages.length - 1];
    // Consecutive tool results share one user turn, as the Messages API expects.
    if (last !== undefined && last.role === role && Array.isArray(last.content) && blocks.every((b) => b.type === 'tool_result') && (last.content as Json[]).every((b) => b.type === 'tool_result')) {
      (last.content as Json[]).push(...blocks);
      return;
    }
    messages.push({ role, content: blocks });
  };
  for (const m of n.messages) {
    if (m.role === 'tool') {
      const block: Json = { type: 'tool_result', tool_use_id: m.toolCallId ?? '', content: textOf(m.parts) };
      if (m.isError) block.is_error = true;
      push('user', [block]);
      continue;
    }
    if (m.role === 'user' && m.plain && m.parts.length === 1 && m.parts[0].kind === 'text') {
      messages.push({ role: 'user', content: m.parts[0].text });
      continue;
    }
    const blocks: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') {
        if (m.role === 'assistant' && p.text === '') continue;
        blocks.push({ type: 'text', text: p.text });
      } else if (p.data !== null) {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: p.mediaType ?? 'image/png', data: p.data } });
      } else if (p.url !== null) {
        blocks.push({ type: 'image', source: { type: 'url', url: p.url } });
      }
    }
    for (const c of m.toolCalls) blocks.push({ type: 'tool_use', id: c.id, name: c.name, input: parseArgs(c.args) });
    if (m.role === 'assistant' && blocks.length === 0) continue;
    messages.push({ role: m.role, content: blocks });
  }
  const body: Json = { model, max_tokens: n.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS };
  if (n.hasSystem) body.system = n.system;
  body.messages = messages;
  if (n.temperature !== null) body.temperature = Math.min(1, Math.max(0, n.temperature));
  if (n.topP !== null) body.top_p = n.topP;
  if (n.topK !== null) body.top_k = n.topK;
  if (n.stops !== null && n.stops.length > 0) body.stop_sequences = n.stops;
  if (n.tools.length > 0) {
    body.tools = n.tools.map((t) => {
      const out: Json = { name: t.name };
      if (t.description !== null) out.description = t.description;
      out.input_schema = t.parameters ?? { type: 'object', properties: {} };
      return out;
    });
    if (n.toolChoice !== null || n.parallelToolCalls === false) {
      const tc: Json =
        n.toolChoice === 'required' ? { type: 'any' } : n.toolChoice === 'none' ? { type: 'none' } : n.toolChoice !== null && typeof n.toolChoice === 'object' ? { type: 'tool', name: n.toolChoice.name } : { type: 'auto' };
      if (n.parallelToolCalls === false && tc.type !== 'none') tc.disable_parallel_tool_use = true;
      body.tool_choice = tc;
    }
  }
  return body;
}

export function toGemini(n: Neutral): Json {
  const contents: Json[] = [];
  const callNames = new Map<string, string>();
  for (const m of n.messages) {
    if (m.role === 'assistant') {
      const parts: Json[] = [];
      for (const p of m.parts) if (p.kind === 'text' && p.text !== '') parts.push({ text: p.text });
      for (const c of m.toolCalls) {
        callNames.set(c.id, c.name);
        parts.push({ functionCall: { name: c.name, args: parseArgs(c.args) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
    } else if (m.role === 'tool') {
      const name = (m.toolCallId !== null ? callNames.get(m.toolCallId) : undefined) ?? m.toolCallId ?? 'tool';
      const part = { functionResponse: { name, response: { content: textOf(m.parts) } } };
      const last = contents[contents.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.parts) && (last.parts as Json[]).every((x) => isRecord(x.functionResponse))) (last.parts as Json[]).push(part);
      else contents.push({ role: 'user', parts: [part] });
    } else {
      const parts: Json[] = [];
      for (const p of m.parts) {
        if (p.kind === 'text') parts.push({ text: p.text });
        else if (p.data !== null) parts.push({ inlineData: { mimeType: p.mediaType ?? 'image/png', data: p.data } });
        else if (p.url !== null) parts.push({ fileData: { mimeType: guessImageType(p.url), fileUri: p.url } });
      }
      contents.push({ role: 'user', parts });
    }
  }
  const body: Json = {};
  if (n.hasSystem) body.systemInstruction = { parts: [{ text: n.system }] };
  body.contents = contents;
  if (n.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: n.tools.map((t) => {
          const fn: Json = { name: t.name };
          if (t.description !== null) fn.description = t.description;
          if (t.parameters !== null) fn.parameters = geminiSchema(t.parameters);
          return fn;
        }),
      },
    ];
    if (n.toolChoice !== null) {
      const cfg: Json =
        n.toolChoice === 'auto' ? { mode: 'AUTO' } : n.toolChoice === 'none' ? { mode: 'NONE' } : n.toolChoice === 'required' ? { mode: 'ANY' } : { mode: 'ANY', allowedFunctionNames: [n.toolChoice.name] };
      body.toolConfig = { functionCallingConfig: cfg };
    }
  }
  const gen: Json = {};
  if (n.maxTokens !== null) gen.maxOutputTokens = n.maxTokens;
  if (n.temperature !== null) gen.temperature = n.temperature;
  if (n.topP !== null) gen.topP = n.topP;
  if (n.topK !== null) gen.topK = n.topK;
  if (n.stops !== null && n.stops.length > 0) gen.stopSequences = n.stops;
  if (n.jsonMode !== null) {
    gen.responseMimeType = 'application/json';
    if (n.jsonMode.type === 'json_schema' && n.jsonMode.schema !== null) gen.responseJsonSchema = n.jsonMode.schema;
  }
  if (Object.keys(gen).length > 0) body.generationConfig = gen;
  return body;
}

function guessImageType(url: string): string {
  const u = url.toLowerCase().split('?')[0];
  if (u.endsWith('.jpg') || u.endsWith('.jpeg')) return 'image/jpeg';
  if (u.endsWith('.webp')) return 'image/webp';
  if (u.endsWith('.gif')) return 'image/gif';
  return 'image/png';
}

/** Gemini's function schema is an OpenAPI subset: drop the JSON Schema keywords it refuses. */
function geminiSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(geminiSchema);
  if (!isRecord(schema)) return schema;
  const out: Json = {};
  for (const [k, v] of Object.entries(schema)) {
    if (k === 'additionalProperties' || k === '$schema' || k === 'strict') continue;
    out[k] = k === 'properties' && isRecord(v) ? Object.fromEntries(Object.entries(v).map(([pk, pv]) => [pk, geminiSchema(pv)])) : geminiSchema(v);
  }
  return out;
}

/** The provider call for one client request: the path under the provider's base and the JSON body. */
export function upstreamCall(client: ClientApi, body: Json, wire: Wire, model: string): { path: string; body: Json } {
  if (wire === client) {
    // The same wire: pass the request through untouched but for the model we picked. The answer is read
    // whole and re-streamed to the client when it asked for a stream.
    const out: Json = { ...body, model };
    delete out.stream;
    delete out.stream_options;
    return { path: wire === 'openai' ? '/v1/chat/completions' : '/v1/messages', body: out };
  }
  const n = readClient(client, body);
  if (wire === 'openai') return { path: '/v1/chat/completions', body: toOpenAI(n, model) };
  if (wire === 'anthropic') return { path: '/v1/messages', body: toAnthropic(n, model) };
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body: toGemini(n) };
}

// ---------------------------------------------------------------------------------------------------
// The neutral answer

export type Stop = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface Answer {
  id: string | null;
  model: string;
  created: number | null;
  text: string;
  toolCalls: ToolCall[];
  stop: Stop;
  stopSequence: string | null;
  usage: { input: number; output: number } | null;
}

const int = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function openAIStop(v: unknown): Stop {
  if (v === 'length') return 'length';
  if (v === 'tool_calls' || v === 'function_call') return 'tool_calls';
  if (v === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStop(v: unknown): Stop {
  if (v === 'max_tokens' || v === 'model_context_window_exceeded') return 'length';
  if (v === 'tool_use') return 'tool_calls';
  if (v === 'refusal') return 'content_filter';
  return 'stop';
}

function geminiStop(v: unknown, hasCalls: boolean): Stop {
  if (v === 'MAX_TOKENS') return 'length';
  if (v === 'SAFETY' || v === 'RECITATION' || v === 'BLOCKLIST' || v === 'PROHIBITED_CONTENT' || v === 'SPII' || v === 'IMAGE_SAFETY') return 'content_filter';
  return hasCalls ? 'tool_calls' : 'stop';
}

/** Read a provider's (non-streamed) answer in its wire; null when it is not one. */
export function readAnswer(wire: Wire, body: unknown, model: string): Answer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    if (!Array.isArray(body.choices) || body.choices.length === 0 || !isRecord(body.choices[0])) return null;
    const choice = body.choices[0];
    const message = isRecord(choice.message) ? choice.message : null;
    if (message === null) return null;
    const content = message.content;
    const text = typeof content === 'string' ? content : Array.isArray(content) ? content.filter((b) => isRecord(b) && typeof b.text === 'string').map((b) => (b as Json).text as string).join('') : '';
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(message.tool_calls)) {
      for (const c of message.tool_calls) {
        if (!isRecord(c) || !isRecord(c.function) || typeof c.function.name !== 'string') continue;
        toolCalls.push({ id: typeof c.id === 'string' ? c.id : `call_${toolCalls.length}`, name: c.function.name, args: argsText(c.function.arguments) });
      }
    } else if (isRecord(message.function_call) && typeof message.function_call.name === 'string') {
      // The legacy single function call.
      toolCalls.push({ id: 'call_0', name: message.function_call.name, args: argsText(message.function_call.arguments) });
    }
    // A refusal carries its own text and no content: pass it on as the answer, stopped by the filter.
    const refusal = typeof message.refusal === 'string' && message.refusal !== '' && text === '' ? message.refusal : null;
    const usage = isRecord(body.usage) ? { input: int(body.usage.prompt_tokens), output: int(body.usage.completion_tokens) } : null;
    return {
      id: typeof body.id === 'string' ? body.id : null,
      model: typeof body.model === 'string' ? body.model : model,
      created: typeof body.created === 'number' ? body.created : null,
      text: refusal ?? text,
      toolCalls,
      stop: refusal !== null ? 'content_filter' : openAIStop(choice.finish_reason),
      stopSequence: null,
      usage,
    };
  }
  if (wire === 'anthropic') {
    if (!Array.isArray(body.content)) return null;
    const texts: string[] = [];
    const toolCalls: ToolCall[] = [];
    for (const b of body.content) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') texts.push(b.text);
      else if (b.type === 'tool_use' && typeof b.name === 'string') toolCalls.push({ id: typeof b.id === 'string' ? b.id : `toolu_${toolCalls.length}`, name: b.name, args: argsText(b.input ?? {}) });
    }
    const u = isRecord(body.usage) ? body.usage : null;
    const usage = u === null ? null : { input: int(u.input_tokens) + int(u.cache_creation_input_tokens) + int(u.cache_read_input_tokens), output: int(u.output_tokens) };
    return {
      id: typeof body.id === 'string' ? body.id : null,
      model: typeof body.model === 'string' ? body.model : model,
      created: null,
      text: texts.join(''),
      toolCalls,
      stop: anthropicStop(body.stop_reason),
      stopSequence: typeof body.stop_sequence === 'string' ? body.stop_sequence : null,
      usage,
    };
  }
  if (!Array.isArray(body.candidates) || body.candidates.length === 0 || !isRecord(body.candidates[0])) return null;
  const cand = body.candidates[0];
  const parts = isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts : [];
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string' && p.thought !== true) texts.push(p.text);
    else if (isRecord(p.functionCall) && typeof p.functionCall.name === 'string') {
      toolCalls.push({ id: typeof p.functionCall.id === 'string' ? p.functionCall.id : `call_${toolCalls.length}`, name: p.functionCall.name, args: argsText(p.functionCall.args ?? {}) });
    }
  }
  const um = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  const usage = um === null ? null : { input: int(um.promptTokenCount), output: int(um.candidatesTokenCount) + int(um.thoughtsTokenCount) };
  return {
    id: typeof body.responseId === 'string' ? body.responseId : null,
    model: typeof body.modelVersion === 'string' ? body.modelVersion : model,
    created: null,
    text: texts.join(''),
    toolCalls,
    stop: geminiStop(cand.finishReason, toolCalls.length > 0),
    stopSequence: null,
    usage,
  };
}

// ---------------------------------------------------------------------------------------------------
// Writing the client's answer

export interface Reply {
  status: number;
  contentType: string;
  body: string;
}

const ANTHROPIC_STOP: Record<Stop, string> = { stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use', content_filter: 'refusal' };

/** Split text into stream deltas of up to `size` characters, never inside a surrogate pair. */
export function chunks(text: string, size = 24): string[] {
  const chars = Array.from(text);
  const out: string[] = [];
  for (let i = 0; i < chars.length; i += size) out.push(chars.slice(i, i + size).join(''));
  return out;
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export interface ReplyIds {
  /** an id for the completion or message when the provider gave none in the client's wire */
  id: string;
  /** the Unix seconds a completion carries when the provider gave none (the trial's fixed now) */
  created: number;
}

function anthropicToolId(id: string): string {
  return id.startsWith('toolu_') ? id : `toolu_${id.replace(/[^A-Za-z0-9_-]/g, '_')}`;
}

export function openAIJson(a: Answer, ids: ReplyIds): Json {
  const message: Json = { role: 'assistant', content: a.text === '' && a.toolCalls.length > 0 ? null : a.text };
  if (a.toolCalls.length > 0) message.tool_calls = a.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
  const out: Json = {
    id: ids.id,
    object: 'chat.completion',
    created: a.created ?? ids.created,
    model: a.model,
    choices: [{ index: 0, message, logprobs: null, finish_reason: a.stop }],
  };
  const usage = a.usage ?? { input: 0, output: 0 };
  out.usage = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
  return out;
}

/** The Anthropic stop_reason: a matched stop sequence is stop_sequence, as the Messages API reports it. */
function anthropicStopReason(a: Answer): string {
  return a.stop === 'stop' && a.stopSequence !== null ? 'stop_sequence' : ANTHROPIC_STOP[a.stop];
}

export function openAIStream(a: Answer, ids: ReplyIds): string {
  const created = a.created ?? ids.created;
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: ids.id,
    object: 'chat.completion.chunk',
    created,
    model: a.model,
    choices: [{ index: 0, delta, logprobs: null, finish_reason: finish }],
  });
  let out = sse(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of chunks(a.text)) out += sse(null, chunk({ content: piece }, null));
  a.toolCalls.forEach((c, i) => {
    out += sse(null, chunk({ tool_calls: [{ index: i, id: c.id, type: 'function', function: { name: c.name, arguments: '' } }] }, null));
    for (const piece of chunks(c.args, 64)) out += sse(null, chunk({ tool_calls: [{ index: i, function: { arguments: piece } }] }, null));
  });
  out += sse(null, chunk({}, a.stop));
  // The usage chunk: no choices, the counts the provider reported (what stream_options.include_usage asks for).
  const usage = a.usage ?? { input: 0, output: 0 };
  out += sse(null, { id: ids.id, object: 'chat.completion.chunk', created, model: a.model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } });
  out += 'data: [DONE]\n\n';
  return out;
}

export function anthropicJson(a: Answer, ids: ReplyIds): Json {
  const content: Json[] = [];
  if (a.text !== '' || a.toolCalls.length === 0) content.push({ type: 'text', text: a.text });
  for (const c of a.toolCalls) content.push({ type: 'tool_use', id: anthropicToolId(c.id), name: c.name, input: parseArgs(c.args) });
  const usage = a.usage ?? { input: 0, output: 0 };
  return {
    id: ids.id,
    type: 'message',
    role: 'assistant',
    model: a.model,
    content,
    stop_reason: anthropicStopReason(a),
    stop_sequence: a.stopSequence,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

export function anthropicStream(a: Answer, ids: ReplyIds): string {
  const usage = a.usage ?? { input: 0, output: 0 };
  let out = sse('message_start', {
    type: 'message_start',
    message: { id: ids.id, type: 'message', role: 'assistant', model: a.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
  });
  let index = 0;
  if (a.text !== '' || a.toolCalls.length === 0) {
    out += sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of chunks(a.text)) out += sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    out += sse('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const c of a.toolCalls) {
    out += sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: anthropicToolId(c.id), name: c.name, input: {} } });
    const json = JSON.stringify(parseArgs(c.args));
    for (const piece of chunks(json, 64)) out += sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: piece } });
    out += sse('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sse('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopReason(a), stop_sequence: a.stopSequence }, usage: { output_tokens: usage.output } });
  out += sse('message_stop', { type: 'message_stop' });
  return out;
}

/** The client's answer: the provider's own JSON when the wires match, else the translation. */
export function clientReply(client: ClientApi, stream: boolean, wire: Wire, raw: unknown, a: Answer, ids: ReplyIds): Reply {
  if (stream) {
    const body = client === 'openai' ? openAIStream(a, ids) : anthropicStream(a, ids);
    return { status: 200, contentType: 'text/event-stream; charset=utf-8', body };
  }
  if (wire === client && isRecord(raw)) return { status: 200, contentType: 'application/json', body: JSON.stringify(raw) };
  const body = client === 'openai' ? openAIJson(a, ids) : anthropicJson(a, ids);
  return { status: 200, contentType: 'application/json', body: JSON.stringify(body) };
}

export function clientError(client: ClientApi, status: number, code: string, message: string): Reply {
  if (client === 'openai') {
    const type = status === 429 ? 'rate_limit_error' : status >= 500 ? 'api_error' : status === 401 ? 'authentication_error' : 'invalid_request_error';
    return { status, contentType: 'application/json', body: JSON.stringify({ error: { message, type, param: null, code } }) };
  }
  const byStatus: Record<number, string> = {
    401: 'authentication_error',
    402: 'billing_error',
    403: 'permission_error',
    404: 'not_found_error',
    413: 'request_too_large',
    429: 'rate_limit_error',
    504: 'timeout_error',
    529: 'overloaded_error',
  };
  const type = byStatus[status] ?? (status >= 500 ? 'api_error' : 'invalid_request_error');
  return { status, contentType: 'application/json', body: JSON.stringify({ type: 'error', error: { type, message } }) };
}
