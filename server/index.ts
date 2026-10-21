// The server: runs the pipeline (server/live.ts) next to the models and providers, so nothing
// detours through the browser; the page only captures the mic, plays the agent and shows the run.
// It also stores session bundles and the settings edited in the web UI (keys, base URLs), and in
// hosted mode serves the built UI.

import { timingSafeEqual } from 'node:crypto';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { chmod, mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { WebSocketServer } from 'ws';
import { HOST_MAX_MESSAGE, LiveHub, VISITOR_MAX_MESSAGE } from './live';
import type { UpstreamName, Upstreams } from './drivers';
import { checkGraph, type GraphSpec } from '../src/core/graph';
import { buildGraph, defaultConfig } from '../src/pipeline';
import { preloadTurnModel, type TurnModelRequest } from './drivers/turn-model';
import type { AcpConfig } from './drivers/acp';

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
  { name: 'TYPESAFE_API_KEY', group: 'TypeSafe (backchannels, end of turn, interjection gate)', label: 'API key', secret: true },
  { name: 'TYPESAFE_BASE_URL', group: 'TypeSafe (backchannels, end of turn, interjection gate)', label: 'Base URL', default: 'https://api.typesafe.ai' },
  {
    name: 'FV_ACP_URL',
    group: 'System 2 (ACP agent)',
    label: 'Agent endpoint (optional)',
    hint: 'ws(s) URL of an agent that speaks the Agent Client Protocol, one JSON-RPC message per frame. Set: workers connect here instead of starting the command below',
  },
  { name: 'FV_ACP_TOKEN', group: 'System 2 (ACP agent)', label: 'Endpoint token', secret: true, hint: 'Sent as a bearer token to the endpoint' },
  {
    name: 'FV_ACP_COMMAND',
    group: 'System 2 (ACP agent)',
    label: 'Agent command',
    default: 'npx -y @agentclientprotocol/claude-agent-acp@0.85.1',
    hint: 'Started once per worker when no endpoint is set; speaks the Agent Client Protocol on stdin/stdout. Gets the API key and base URL below',
  },
  { name: 'ANTHROPIC_API_KEY', group: 'System 2 (ACP agent)', label: 'API key', secret: true, hint: 'Passed to the agent command as ANTHROPIC_API_KEY' },
  { name: 'ANTHROPIC_BASE_URL', group: 'System 2 (ACP agent)', label: 'API base URL', hint: "Passed as ANTHROPIC_BASE_URL; empty uses the agent's default" },
  { name: 'FV_ACP_CWD', group: 'System 2 (ACP agent)', label: 'Working folder', default: '.frankenvoice/workers', hint: 'Where started agents run; relative to where the server runs' },
  { name: 'FV_SESSIONS_DIR', group: 'Storage', label: 'Sessions folder', default: 'sessions', hint: 'Relative to where the proxy runs' },
  {
    name: 'FV_MAX_SESSIONS',
    group: 'Users',
    label: 'Live sessions at once',
    default: '1',
    hint: 'More pages wait in a queue. One L4 gives one user the best latency; 2 users add about 1 s to each reply, 4 about 2 s',
  },
  {
    name: 'FV_MAX_SESSION_S',
    group: 'Users',
    label: 'Longest live session (s)',
    default: '180',
    hint: 'A live session is stopped (and saved) after this long, so a stuck page can\'t hold the box. 0 = no limit',
  },
  {
    name: 'FV_PUBLIC',
    group: 'Users',
    label: 'Public demo (1 = on)',
    default: '0',
    hint: 'Hosted mode: anyone with the link (no token) can talk to the agent. Visitors run the pipeline you started last and see no settings, recordings or replays; with the token you stay the host',
  },
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
/** Downloaded models (the turn model), next to the settings file so a redeployed app keeps them. */
const MODELS_DIR = resolve(process.env.FV_MODELS_DIR ?? join(dirname(SETTINGS_FILE), 'fv-models'));
/** Canned lines' audio, rendered once per GPU box and kept across sessions and redeploys. */
const CANNED_DIR = resolve(process.env.FV_TTS_CACHE_DIR ?? join(dirname(SETTINGS_FILE), 'fv-tts-cache'));

interface Upstream {
  baseVar: string;
  keyVar: string;
  /** Local OpenAI-compatible servers often need no key. */
  keyOptional?: boolean;
  headers(key: string | undefined): Record<string, string>;
}

const bearer = (key: string | undefined): Record<string, string> => (key ? { authorization: `Bearer ${key}` } : {});

const upstreams: Record<UpstreamName, Upstream> = {
  typesafe: { baseVar: 'TYPESAFE_BASE_URL', keyVar: 'TYPESAFE_API_KEY', headers: bearer },
  gpubox: { baseVar: 'GPUBOX_BASE_URL', keyVar: 'GPUBOX_TOKEN', headers: bearer },
  // The OpenAI-compatible chat endpoint (llama.cpp on the GPU box).
  llm: { baseVar: 'OPENAI_COMPAT_BASE_URL', keyVar: 'OPENAI_COMPAT_API_KEY', keyOptional: true, headers: bearer },
};

const baseOf = (u: Upstream) => setting(u.baseVar);
const keyOf = (u: Upstream) => setting(u.keyVar);
const configured = (u: Upstream) => !!baseOf(u) && (!!keyOf(u) || !!u.keyOptional);

/** Read on every request, so a change in Settings applies to the next exchange. */
const upstream: Upstreams = (name) => {
  const u = upstreams[name];
  const missing = !baseOf(u) ? `no base URL configured for "${name}" (${u.baseVar}; set it in Settings)` : !configured(u) ? `${u.keyVar} is not set (set it in Settings)` : undefined;
  return { base: baseOf(u)?.replace(/\/+$/, ''), headers: u.headers(keyOf(u)), missing };
};

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

const sameOrigin = (req: IncomingMessage) => {
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    return new URL(origin).host === req.headers.host;
  } catch {
    return false;
  }
};

/** The host: in hosted mode, same-origin requests carrying the token cookie; otherwise only the local UI. */
function isHost(req: IncomingMessage): boolean {
  if (!HOST_TOKEN) return localOnly(req);
  return sameOrigin(req) && sameToken(decodeURIComponent(cookieToken(req)));
}

/** Public demo (hosted mode only): anyone may load the page and talk; settings, recordings and the turn model stay the host's. */
const isPublic = () => !!HOST_TOKEN && setting('FV_PUBLIC') === '1';
const VISITOR_PATHS = new Set(['/api/health', '/api/me', '/api/live']);

function allowed(req: IncomingMessage, p: string): boolean {
  if (isHost(req)) return true;
  return isPublic() && sameOrigin(req) && (VISITOR_PATHS.has(p) || (!!STATIC_DIR && !p.startsWith('/api/') && (req.method === 'GET' || req.method === 'HEAD')));
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

function json(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  return Buffer.concat(chunks);
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

/** What demo visitors run: the pipeline the host started last (kept across restarts). */
const DEMO_GRAPH_FILE = join(dirname(SETTINGS_FILE), 'demo-graph.json');
let demoGraph: GraphSpec | undefined;
try {
  demoGraph = checkGraph(JSON.parse(await readFile(DEMO_GRAPH_FILE, 'utf8')) as GraphSpec);
} catch {
  // none yet, or an outdated format: visitors get the default pipeline
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
  if (!allowed(req, p)) {
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
    if (p === '/api/me') {
      // A visitor sees the pipeline it will talk to.
      const host = isHost(req);
      return json(res, 200, { host, public: isPublic(), ...(host ? {} : { graph: demoGraph ?? buildGraph(defaultConfig()) }) });
    }
    if (p === '/api/health') {
      return json(res, 200, { ...Object.fromEntries(Object.entries(upstreams).map(([k, u]) => [k, configured(u)])), acp: !!(setting('FV_ACP_URL') || setting('FV_ACP_COMMAND')) });
    }
    if (p === '/api/turn-model' && req.method === 'POST') {
      // Load the turn model ahead of a session; the page polls this while it loads.
      const r = JSON.parse((await readBody(req)).toString('utf8') || '{}') as TurnModelRequest;
      if (typeof r.encoderUrl !== 'string' || typeof r.headUrl !== 'string' || typeof r.frames !== 'number') return json(res, 400, { error: 'bad turn model request' });
      return json(res, 200, preloadTurnModel(MODELS_DIR, r));
    }
    const m = /^\/api\/sessions(?:\/([^/]+))?$/.exec(p);
    if (m) return sessions(req, res, m[1] && decodeURIComponent(m[1]));
    if (STATIC_DIR && !p.startsWith('/api/') && (req.method === 'GET' || req.method === 'HEAD')) return serveStatic(p, res);
    json(res, 404, { error: 'not found' });
  };
  handle().catch((e) => {
    if (!res.headersSent) json(res, 502, { error: String(e) });
    else res.destroy();
  });
});

/** System 2, as Settings has it now: a remote endpoint, or a command started per worker with its key. */
const acpConfig = (): AcpConfig => {
  const env: Record<string, string> = {};
  for (const k of ['ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL']) {
    const v = setting(k);
    if (v) env[k] = v;
  }
  return { url: setting('FV_ACP_URL'), token: setting('FV_ACP_TOKEN'), command: setting('FV_ACP_COMMAND'), cwd: resolve(setting('FV_ACP_CWD')!), env };
};

// The page's socket: mic in, agent audio and the run's trajectory out.
const live = new LiveHub({ upstreams: upstream, sessionsDir, acp: acpConfig, modelsDir: MODELS_DIR, cannedDir: CANNED_DIR, maxSessions: () => Number(setting('FV_MAX_SESSIONS')) || 1,
  maxSessionMs: () => Math.max(0, Number(setting('FV_MAX_SESSION_S') ?? 180) || 0) * 1000,
  demoGraph: () => demoGraph ?? buildGraph(defaultConfig()),
  hostGraph: (g) => {
    demoGraph = g;
    void writeFile(DEMO_GRAPH_FILE, JSON.stringify(g)).catch((e) => console.error('[server] could not save the demo pipeline:', e));
  },
});
// ws buffers a whole message before handing it over, so the cap must hold at the socket: a visitor sends mic frames
// and controls (small), the host also replay bundles (large).
const hostWss = new WebSocketServer({ noServer: true, maxPayload: HOST_MAX_MESSAGE });
const visitorWss = new WebSocketServer({ noServer: true, maxPayload: VISITOR_MAX_MESSAGE });
server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url ?? '/', 'http://local');
  if (url.pathname !== '/api/live' || !allowed(req, url.pathname)) return socket.destroy();
  const host = isHost(req);
  (host ? hostWss : visitorWss).handleUpgrade(req, socket, head, (ws) => live.handle(ws, { host }));
});

// Last resort: one bad request must not stop the proxy, which would drop every connection of a live session.
process.on('uncaughtException', (e) => console.error('[server] uncaught error, still running:', e));
// Exit on these signals, so the exit hook takes System 2's agent processes down too.
for (const [sig, code] of [['SIGINT', 130], ['SIGTERM', 143], ['SIGHUP', 129]] as const) process.on(sig, () => process.exit(code));

server.listen(PORT, '127.0.0.1', () => {
  const keys = Object.entries(upstreams)
    .map(([k, u]) => `${k}:${configured(u) ? 'ok' : 'off'}`)
    .join(' ');
  const mode = HOST_TOKEN ? `  hosted (token)${STATIC_DIR ? `, UI from ${STATIC_DIR}` : ''}` : '';
  console.log(`frankenvoice server on http://127.0.0.1:${PORT}${mode}  [${keys}]  sessions -> ${sessionsDir()}  settings -> ${SETTINGS_FILE}`);
});
