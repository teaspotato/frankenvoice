// Local proxy: keeps API keys out of the browser and avoids CORS. It forwards
// requests verbatim (adding credentials) and streams responses back untouched,
// so the browser records exactly what the provider sent. It also stores
// session bundles and the settings edited in the web UI (keys, base URLs).

import { timingSafeEqual } from 'node:crypto';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { chmod, mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { WebSocket, WebSocketServer } from 'ws';

try {
  process.loadEnvFile('.env');
} catch {
  // no .env: keys come from the shell environment
}

const PORT = Number(process.env.FV_PROXY_PORT ?? 8787);
/**
 * Hosted mode (e.g. on the GPU box, behind a tunnel): the proxy also serves the built UI from
 * FV_STATIC_DIR, and instead of local-only access every request needs FV_HOST_TOKEN, given once as
 * `?token=` and kept in a cookie.
 */
const HOST_TOKEN = process.env.FV_HOST_TOKEN || '';
const STATIC_DIR = process.env.FV_STATIC_DIR ? resolve(process.env.FV_STATIC_DIR) : '';

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
  {
    name: 'GPUBOX_BASE_URL',
    group: 'GPU box (Qwen3-ASR, Kyutai TTS)',
    label: 'Base URL',
    hint: 'ws(s) URL of gpubox/server.py, e.g. wss://xyz.trycloudflare.com',
  },
  { name: 'GPUBOX_TOKEN', group: 'GPU box (Qwen3-ASR, Kyutai TTS)', label: 'Token', secret: true, hint: 'Printed by gpubox/setup.sh' },
  {
    name: 'OPENAI_COMPAT_BASE_URL',
    group: 'LLM (OpenAI-compatible)',
    label: 'Base URL',
    hint: 'e.g. http://127.0.0.1:8103/v1 (llama.cpp on the GPU box), or any local server: vLLM, Ollama, LM Studio',
  },
  { name: 'OPENAI_COMPAT_API_KEY', group: 'LLM (OpenAI-compatible)', label: 'API key (optional)', secret: true },
  { name: 'TYPESAFE_API_KEY', group: 'TypeSafe (opener, backchannels, interjection gate)', label: 'API key', secret: true },
  { name: 'TYPESAFE_BASE_URL', group: 'TypeSafe (opener, backchannels, interjection gate)', label: 'Base URL', default: 'https://api.typesafe.ai' },
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
  typesafe: { baseVar: 'TYPESAFE_BASE_URL', keyVar: 'TYPESAFE_API_KEY', headers: bearer },
  gpubox: { baseVar: 'GPUBOX_BASE_URL', keyVar: 'GPUBOX_TOKEN', headers: bearer },
  // The OpenAI-compatible chat endpoint (llama.cpp on the GPU box), addressed as /api/llm/...
  llm: { baseVar: 'OPENAI_COMPAT_BASE_URL', keyVar: 'OPENAI_COMPAT_API_KEY', keyOptional: true, headers: bearer },
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

const sameToken = (got: string) => {
  const a = Buffer.from(got);
  const b = Buffer.from(HOST_TOKEN);
  return a.length === b.length && timingSafeEqual(a, b);
};
const cookieToken = (req: IncomingMessage) => /(?:^|;\s*)fv_token=([^;]*)/.exec(req.headers.cookie ?? '')?.[1] ?? '';

/** Hosted mode: same-origin requests carrying the token cookie. Otherwise only the local UI. */
function allowed(req: IncomingMessage): boolean {
  if (!HOST_TOKEN) return localOnly(req);
  const origin = req.headers.origin;
  if (origin) {
    try {
      if (new URL(origin).host !== req.headers.host) return false;
    } catch {
      return false;
    }
  }
  return sameToken(decodeURIComponent(cookieToken(req)));
}

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.onnx': 'application/octet-stream',
};

/** The built UI; unknown paths get index.html. */
async function serveStatic(p: string, res: ServerResponse) {
  let file = resolve(STATIC_DIR, '.' + decodeURIComponent(p));
  if (file !== STATIC_DIR && !file.startsWith(STATIC_DIR + sep)) return json(res, 400, { error: 'bad path' });
  if (!(await stat(file).catch(() => null))?.isFile()) file = join(STATIC_DIR, 'index.html');
  res.writeHead(200, {
    'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
    'cache-control': file.includes(`${sep}assets${sep}`) ? 'public, max-age=31536000, immutable' : 'no-cache',
  });
  res.end(await readFile(file));
}

const DROP_REQ = new Set(['host', 'connection', 'content-length', 'x-api-key', 'authorization', 'origin', 'referer', 'cookie']);
// Added by a tunnel or reverse proxy in front of a hosted proxy; providers don't need the viewer's IP.
const DROP_REQ_PREFIX = /^(cf-|x-forwarded-|x-real-ip$|true-client-ip$|forwarded$|cdn-loop$|via$)/;
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
  for (const [k, v] of Object.entries(req.headers)) if (!DROP_REQ.has(k) && !DROP_REQ_PREFIX.test(k) && typeof v === 'string') headers[k] = v;
  Object.assign(headers, up.headers(key));
  const body = req.method === 'GET' || req.method === 'HEAD' ? undefined : new Uint8Array(await readBody(req));
  const ac = new AbortController();
  res.on('close', () => ac.abort());
  // The GPU box is a ws(s) URL for streaming; its REST endpoints use http(s).
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
  // Hosted mode: `?token=` on a page load sets the cookie, then the token is dropped from the URL.
  // Lax, not Strict: a Strict cookie isn't sent on the redirect when the link was opened from another site.
  const given = url.searchParams.get('token');
  if (HOST_TOKEN && given !== null && req.method === 'GET' && !p.startsWith('/api/')) {
    if (!sameToken(given)) return json(res, 403, { error: 'wrong token' });
    url.searchParams.delete('token');
    res.writeHead(302, {
      'set-cookie': `fv_token=${encodeURIComponent(given)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000`,
      location: url.pathname + url.search,
    });
    return res.end();
  }
  if (!allowed(req)) {
    return json(res, 403, { error: HOST_TOKEN ? 'open this page with ?token=<FV_HOST_TOKEN> first' : 'the proxy only accepts requests from the local UI' });
  }
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
    let m = /^\/api\/(gpubox|typesafe|llm)(\/.*)$/.exec(p);
    if (m) return forward(m[1], m[2] + url.search, req, res);
    m = /^\/api\/sessions(?:\/([^/]+))?$/.exec(p);
    if (m) return sessions(req, res, m[1] && decodeURIComponent(m[1]));
    if (STATIC_DIR && !p.startsWith('/api/') && (req.method === 'GET' || req.method === 'HEAD')) return serveStatic(p, res);
    json(res, 404, { error: 'not found' });
  };
  handle().catch((e) => {
    if (!res.headersSent) json(res, 502, { error: String(e) });
    else res.destroy();
  });
});

// Streaming (the GPU box): browser <-> proxy <-> upstream, frames passed through unchanged.
const wss = new WebSocketServer({ noServer: true });
server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url ?? '/', 'http://local');
  const m = /^\/api\/(gpubox)(\/.*)$/.exec(url.pathname);
  if (!m || !allowed(req)) return socket.destroy();
  const name = m[1];
  wss.handleUpgrade(req, socket, head, (client) => {
    const up = upstreams[name];
    if (!configured(up)) return client.close(1011, `${!baseOf(up) ? up.baseVar : up.keyVar} is not set (set it in Settings)`);
    const remote = new WebSocket(baseOf(up) + m[2] + url.search, { headers: up.headers(keyOf(up)) });
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
      client.close(1011, `${name} HTTP ${res.statusCode}`.slice(0, 120));
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
  const mode = HOST_TOKEN ? `  hosted (token)${STATIC_DIR ? `, UI from ${STATIC_DIR}` : ''}` : '';
  console.log(`frankenvoice proxy on http://127.0.0.1:${PORT}${mode}  [${keys}]  sessions -> ${sessionsDir()}  settings -> ${SETTINGS_FILE}`);
});
