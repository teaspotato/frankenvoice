// Local proxy: keeps API keys out of the browser and avoids CORS. It forwards
// requests verbatim (adding credentials) and streams responses back untouched,
// so the browser records exactly what the provider sent. It also stores
// session bundles and the settings edited in the web UI (keys, base URLs).

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { chmod, mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, join, resolve } from 'node:path';
import { WebSocket, WebSocketServer } from 'ws';

try {
  process.loadEnvFile('.env');
} catch {
  // no .env: keys come from the shell environment
}

const PORT = Number(process.env.FV_PROXY_PORT ?? 8787);

/**
 * Every setting can come from three places, highest first: the web UI (saved
 * to SETTINGS_FILE), the environment / .env, or a built-in default. Secrets
 * are write-only over HTTP: the UI sees whether a key is set, never its value.
 */
interface SettingDef {
  name: string;
  group: string;
  label: string;
  secret?: boolean;
  default?: string;
  hint?: string;
}

const SETTINGS: SettingDef[] = [
  { name: 'DEEPGRAM_API_KEY', group: 'Deepgram (ASR, TTS)', label: 'API key', secret: true },
  { name: 'DEEPGRAM_BASE_URL', group: 'Deepgram (ASR, TTS)', label: 'Base URL', default: 'wss://api.deepgram.com', hint: 'ws(s) URL; REST calls use the same host over http(s)' },
  { name: 'ANTHROPIC_API_KEY', group: 'Anthropic (LLM)', label: 'API key', secret: true },
  { name: 'ANTHROPIC_BASE_URL', group: 'Anthropic (LLM)', label: 'Base URL', default: 'https://api.anthropic.com' },
  { name: 'OPENROUTER_API_KEY', group: 'OpenRouter (LLM)', label: 'API key', secret: true },
  { name: 'OPENROUTER_BASE_URL', group: 'OpenRouter (LLM)', label: 'Base URL', default: 'https://openrouter.ai/api/v1' },
  { name: 'OPENAI_API_KEY', group: 'OpenAI (LLM, TTS)', label: 'API key', secret: true },
  { name: 'OPENAI_BASE_URL', group: 'OpenAI (LLM, TTS)', label: 'Base URL', default: 'https://api.openai.com/v1' },
  { name: 'OPENAI_COMPAT_BASE_URL', group: 'Custom OpenAI-compatible (LLM, TTS)', label: 'Base URL', hint: 'e.g. http://localhost:11434/v1 (Ollama), vLLM, LM Studio' },
  { name: 'OPENAI_COMPAT_API_KEY', group: 'Custom OpenAI-compatible (LLM, TTS)', label: 'API key (optional)', secret: true },
  { name: 'CARTESIA_API_KEY', group: 'Cartesia (TTS)', label: 'API key', secret: true },
  { name: 'CARTESIA_BASE_URL', group: 'Cartesia (TTS)', label: 'Base URL', default: 'https://api.cartesia.ai' },
  { name: 'CARTESIA_VERSION', group: 'Cartesia (TTS)', label: 'API version', default: '2025-04-16' },
  { name: 'FV_SESSIONS_DIR', group: 'Storage', label: 'Sessions folder', default: 'sessions', hint: 'Relative to where the proxy runs' },
];
const SETTING_NAMES = new Set(SETTINGS.map((d) => d.name));
const SETTINGS_FILE = resolve(process.env.FV_SETTINGS_FILE ?? '.frankenvoice/settings.json');

let saved: Record<string, string> = {};
try {
  saved = JSON.parse(await readFile(SETTINGS_FILE, 'utf8')) as Record<string, string>;
} catch {
  // nothing saved from the UI yet
}

function setting(name: string): string | undefined {
  const def = SETTINGS.find((d) => d.name === name);
  return saved[name] || process.env[name] || def?.default || undefined;
}

function settingSource(name: string): 'ui' | 'env' | 'default' | 'unset' {
  if (saved[name]) return 'ui';
  if (process.env[name]) return 'env';
  return SETTINGS.find((d) => d.name === name)?.default ? 'default' : 'unset';
}

function describeSettings() {
  return SETTINGS.map((d) => {
    const value = setting(d.name);
    return {
      ...d,
      source: settingSource(d.name),
      set: !!value,
      // Secrets never leave the proxy; the last four characters help tell keys apart.
      value: d.secret ? null : (value ?? null),
      preview: d.secret && value ? `…${value.slice(-4)}` : null,
    };
  });
}

/** Apply {NAME: value} from the UI; null or '' removes the UI value (falls back to .env / default). */
async function updateSettings(patch: Record<string, unknown>) {
  const next = { ...saved };
  for (const [k, v] of Object.entries(patch)) {
    if (!SETTING_NAMES.has(k)) throw new Error(`unknown setting "${k}"`);
    if (v === null || v === '') delete next[k];
    else if (typeof v === 'string') next[k] = v.trim();
    else throw new Error(`setting "${k}" must be a string or null`);
  }
  await mkdir(dirname(SETTINGS_FILE), { recursive: true });
  await writeFile(SETTINGS_FILE, JSON.stringify(next, null, 2), { mode: 0o600 });
  await chmod(SETTINGS_FILE, 0o600);
  saved = next;
}

const sessionsDir = () => resolve(setting('FV_SESSIONS_DIR')!);

interface Upstream {
  baseVar: string;
  keyVar: string;
  /** Local OpenAI-compatible servers often need no key. */
  keyOptional?: boolean;
  headers(key: string | undefined): Record<string, string>;
}

const bearer = (key: string | undefined): Record<string, string> => (key ? { authorization: `Bearer ${key}` } : {});

const upstreams: Record<string, Upstream> = {
  anthropic: { baseVar: 'ANTHROPIC_BASE_URL', keyVar: 'ANTHROPIC_API_KEY', headers: (key) => ({ 'x-api-key': key! }) },
  cartesia: {
    baseVar: 'CARTESIA_BASE_URL',
    keyVar: 'CARTESIA_API_KEY',
    headers: (key) => ({ 'x-api-key': key!, 'cartesia-version': setting('CARTESIA_VERSION')! }),
  },
  deepgram: { baseVar: 'DEEPGRAM_BASE_URL', keyVar: 'DEEPGRAM_API_KEY', headers: (key) => ({ authorization: `Token ${key}` }) },
  // OpenAI-compatible endpoints, addressed as /api/openai/<name>/...
  'openai/openrouter': {
    baseVar: 'OPENROUTER_BASE_URL',
    keyVar: 'OPENROUTER_API_KEY',
    headers: (key) => ({ ...bearer(key), 'x-title': 'frankenvoice' }),
  },
  'openai/openai': { baseVar: 'OPENAI_BASE_URL', keyVar: 'OPENAI_API_KEY', headers: bearer },
  'openai/custom': { baseVar: 'OPENAI_COMPAT_BASE_URL', keyVar: 'OPENAI_COMPAT_API_KEY', keyOptional: true, headers: bearer },
};

const baseOf = (u: Upstream) => setting(u.baseVar);
const keyOf = (u: Upstream) => setting(u.keyVar);
const configured = (u: Upstream) => !!baseOf(u) && (!!keyOf(u) || !!u.keyOptional);

/**
 * The proxy holds credentials and can repoint upstreams, so only the local UI
 * may talk to it: reject other websites (Origin) and DNS rebinding (Host).
 */
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/;
function localOnly(req: IncomingMessage): boolean {
  if (!LOCAL_HOST.test(req.headers.host ?? '')) return false;
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    return LOCAL_HOST.test(new URL(origin).host);
  } catch {
    return false;
  }
}

const DROP_REQ = new Set(['host', 'connection', 'content-length', 'x-api-key', 'authorization', 'origin', 'referer', 'cookie', 'anthropic-dangerous-direct-browser-access']);
const DROP_RES = new Set(['content-encoding', 'content-length', 'transfer-encoding', 'connection']);

function json(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  return Buffer.concat(chunks);
}

async function forward(name: string, rest: string, req: IncomingMessage, res: ServerResponse) {
  const up = upstreams[name];
  if (!up) return json(res, 404, { error: `unknown upstream "${name}"` });
  const upBase = baseOf(up);
  const key = keyOf(up);
  if (!upBase) return json(res, 500, { error: `no base URL configured for "${name}" (${up.baseVar}; set it in Settings)` });
  if (!key && !up.keyOptional) return json(res, 500, { error: `${up.keyVar} is not set (set it in Settings)` });
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(req.headers)) if (!DROP_REQ.has(k) && typeof v === 'string') headers[k] = v;
  Object.assign(headers, up.headers(key));
  const body = req.method === 'GET' || req.method === 'HEAD' ? undefined : new Uint8Array(await readBody(req));
  const ac = new AbortController();
  res.on('close', () => ac.abort());
  // The Deepgram upstream is a ws(s) URL for streaming ASR; its REST endpoints (TTS) use http(s).
  const base = upBase.replace(/^ws(s?):/, 'http$1:');
  const upstream = await fetch(base + rest, { method: req.method, headers, body, signal: ac.signal });
  const outHeaders: Record<string, string> = {};
  upstream.headers.forEach((v, k) => {
    if (!DROP_RES.has(k)) outHeaders[k] = v;
  });
  res.writeHead(upstream.status, outHeaders);
  if (!upstream.body) return res.end();
  // Errors mid-stream (the browser aborting a reply it no longer wants, the upstream resetting) end
  // this response only. With a bare .pipe() they were unhandled and took the whole proxy down.
  await pipeline(Readable.fromWeb(upstream.body as import('node:stream/web').ReadableStream), res).catch((e: unknown) => {
    if (!ac.signal.aborted) console.warn(`[${name}] response stream failed: ${String(e)}`);
    res.destroy();
  });
}

/** Close codes 1005/1006/1015 are reserved and may not be sent in a close frame. */
function sendableCode(code: number): number {
  if (code < 1000 || code >= 5000 || code === 1004 || code === 1005 || code === 1006 || code === 1015) return code === 1005 ? 1000 : 1011;
  return code;
}

/** Session ids are file names: no path separators or reserved characters, no leading dot. */
const SAFE_ID = /^[^/\\:*?"<>|\x00-\x1f.\s][^/\\:*?"<>|\x00-\x1f]{0,99}$/u;

async function sessions(req: IncomingMessage, res: ServerResponse, id: string | undefined) {
  const SESSIONS_DIR = sessionsDir();
  await mkdir(SESSIONS_DIR, { recursive: true });
  if (!id && req.method === 'GET') {
    const files = (await readdir(SESSIONS_DIR)).filter((f) => f.endsWith('.fvs'));
    const list = await Promise.all(files.map(async (f) => ({ id: f.slice(0, -4), bytes: (await stat(join(SESSIONS_DIR, f))).size })));
    return json(res, 200, list.sort((a, b) => b.id.localeCompare(a.id)));
  }
  if (!id || !SAFE_ID.test(id)) return json(res, 400, { error: 'bad session id' });
  const file = join(SESSIONS_DIR, `${id}.fvs`);
  if (req.method === 'PUT') {
    await writeFile(file, await readBody(req));
    return json(res, 200, { ok: true, path: file });
  }
  if (req.method === 'GET') {
    res.writeHead(200, { 'content-type': 'application/zip' });
    return res.end(await readFile(file));
  }
  if (req.method === 'PATCH') {
    const to = String((JSON.parse((await readBody(req)).toString('utf8') || '{}') as { id?: unknown }).id ?? '').trim();
    if (!SAFE_ID.test(to)) return json(res, 400, { error: 'bad session name' });
    const dest = join(SESSIONS_DIR, `${to}.fvs`);
    if (to !== id && (await stat(dest).catch(() => null))) return json(res, 409, { error: `a session named ${to} already exists` });
    await rename(file, dest);
    return json(res, 200, { ok: true, path: dest });
  }
  if (req.method === 'DELETE') {
    await rm(file, { force: true });
    return json(res, 200, { ok: true });
  }
  json(res, 405, { error: 'method not allowed' });
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://local');
  const p = url.pathname;
  if (!localOnly(req)) return json(res, 403, { error: 'the proxy only accepts requests from the local UI' });
  const handle = async () => {
    if (p === '/api/settings') {
      if (req.method === 'PUT') {
        try {
          await updateSettings(JSON.parse((await readBody(req)).toString('utf8') || '{}') as Record<string, unknown>);
        } catch (e) {
          return json(res, 400, { error: String(e instanceof Error ? e.message : e) });
        }
      } else if (req.method !== 'GET') return json(res, 405, { error: 'method not allowed' });
      return json(res, 200, { file: SETTINGS_FILE, settings: describeSettings() });
    }
    if (p === '/api/health') {
      return json(res, 200, Object.fromEntries(Object.entries(upstreams).map(([k, u]) => [k, configured(u)])));
    }
    let m = /^\/api\/(anthropic|cartesia|deepgram|openai\/[\w-]+)(\/.*)$/.exec(p);
    if (m) return forward(m[1], m[2] + url.search, req, res);
    m = /^\/api\/sessions(?:\/([^/]+))?$/.exec(p);
    if (m) return sessions(req, res, m[1] && decodeURIComponent(m[1]));
    json(res, 404, { error: 'not found' });
  };
  handle().catch((e) => {
    if (!res.headersSent) json(res, 502, { error: String(e) });
    else res.destroy();
  });
});

// Deepgram streaming: browser <-> proxy <-> Deepgram, frames passed through unchanged.
const wss = new WebSocketServer({ noServer: true });
server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url ?? '/', 'http://local');
  const m = /^\/api\/deepgram(\/.*)$/.exec(url.pathname);
  if (!m || !localOnly(req)) return socket.destroy();
  wss.handleUpgrade(req, socket, head, (client) => {
    const up = upstreams.deepgram;
    if (!configured(up)) return client.close(1011, 'DEEPGRAM_API_KEY is not set (set it in Settings)');
    const remote = new WebSocket(baseOf(up) + m[1] + url.search, { headers: up.headers(keyOf(up)) });
    const queued: { data: Buffer; binary: boolean }[] = [];
    client.on('message', (data: Buffer, binary) => {
      if (remote.readyState === WebSocket.OPEN) remote.send(data, { binary });
      else queued.push({ data, binary });
    });
    remote.on('open', () => {
      for (const q of queued.splice(0)) remote.send(q.data, { binary: q.binary });
    });
    remote.on('message', (data: Buffer, binary) => client.readyState === WebSocket.OPEN && client.send(data, { binary }));
    remote.on('unexpected-response', (_r, res) => {
      client.close(1011, `deepgram HTTP ${res.statusCode}`.slice(0, 120));
    });
    // A close reason over 123 bytes throws; keep it short.
    remote.on('close', (code, reason) => client.readyState === WebSocket.OPEN && client.close(sendableCode(code), reason.subarray(0, 120)));
    remote.on('error', (e) => client.readyState === WebSocket.OPEN && client.close(1011, String(e).slice(0, 120)));
    client.on('close', () => remote.readyState <= WebSocket.OPEN && remote.close());
  });
});

// Last resort: one bad request must not stop the proxy, which would drop every connection of a live session.
process.on('uncaughtException', (e) => console.error('[proxy] uncaught error, still running:', e));

server.listen(PORT, '127.0.0.1', () => {
  const keys = Object.entries(upstreams)
    .map(([k, u]) => `${k}:${configured(u) ? 'ok' : 'off'}`)
    .join(' ');
  console.log(`frankenvoice proxy on http://127.0.0.1:${PORT}  [${keys}]  sessions -> ${sessionsDir()}  settings -> ${SETTINGS_FILE}`);
});
