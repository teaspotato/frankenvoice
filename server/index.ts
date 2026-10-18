// Local proxy: keeps API keys out of the browser and avoids CORS. It forwards
// requests verbatim (adding credentials) and streams responses back untouched,
// so the browser records exactly what the provider sent. It also stores
// session bundles under ./sessions.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { join, resolve } from 'node:path';
import { WebSocket, WebSocketServer } from 'ws';

try {
  process.loadEnvFile('.env');
} catch {
  // no .env: keys come from the shell environment
}

const PORT = Number(process.env.FV_PROXY_PORT ?? 8787);
const SESSIONS_DIR = resolve(process.env.FV_SESSIONS_DIR ?? 'sessions');

interface Upstream {
  base: string;
  key: string | undefined;
  headers(key: string): Record<string, string>;
}

const upstreams: Record<string, Upstream> = {
  anthropic: {
    base: process.env.ANTHROPIC_BASE_URL ?? 'https://api.anthropic.com',
    key: process.env.ANTHROPIC_API_KEY,
    headers: (key) => ({ 'x-api-key': key }),
  },
  cartesia: {
    base: process.env.CARTESIA_BASE_URL ?? 'https://api.cartesia.ai',
    key: process.env.CARTESIA_API_KEY,
    headers: (key) => ({ 'x-api-key': key, 'cartesia-version': process.env.CARTESIA_VERSION ?? '2025-04-16' }),
  },
  deepgram: {
    base: process.env.DEEPGRAM_BASE_URL ?? 'wss://api.deepgram.com',
    key: process.env.DEEPGRAM_API_KEY,
    headers: (key) => ({ authorization: `Token ${key}` }),
  },
};

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
  if (!up.key) return json(res, 500, { error: `${name.toUpperCase()}_API_KEY is not set in the proxy's environment` });
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(req.headers)) if (!DROP_REQ.has(k) && typeof v === 'string') headers[k] = v;
  Object.assign(headers, up.headers(up.key));
  const body = req.method === 'GET' || req.method === 'HEAD' ? undefined : new Uint8Array(await readBody(req));
  const ac = new AbortController();
  res.on('close', () => ac.abort());
  const upstream = await fetch(up.base + rest, { method: req.method, headers, body, signal: ac.signal });
  const outHeaders: Record<string, string> = {};
  upstream.headers.forEach((v, k) => {
    if (!DROP_RES.has(k)) outHeaders[k] = v;
  });
  res.writeHead(upstream.status, outHeaders);
  if (!upstream.body) return res.end();
  Readable.fromWeb(upstream.body as import('node:stream/web').ReadableStream).pipe(res);
}

/** Close codes 1005/1006/1015 are reserved and may not be sent in a close frame. */
function sendableCode(code: number): number {
  if (code < 1000 || code >= 5000 || code === 1004 || code === 1005 || code === 1006 || code === 1015) return code === 1005 ? 1000 : 1011;
  return code;
}

const SAFE_ID = /^[\w.-]+$/;

async function sessions(req: IncomingMessage, res: ServerResponse, id: string | undefined) {
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
  json(res, 405, { error: 'method not allowed' });
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://local');
  const p = url.pathname;
  const handle = async () => {
    if (p === '/api/health') {
      return json(res, 200, Object.fromEntries(Object.entries(upstreams).map(([k, u]) => [k, !!u.key])));
    }
    let m = /^\/api\/(anthropic|cartesia)(\/.*)$/.exec(p);
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
  if (!m) return socket.destroy();
  wss.handleUpgrade(req, socket, head, (client) => {
    const up = upstreams.deepgram;
    if (!up.key) return client.close(1011, 'DEEPGRAM_API_KEY is not set in the proxy environment');
    const remote = new WebSocket(up.base + m[1] + url.search, { headers: up.headers(up.key) });
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
    remote.on('close', (code, reason) => client.readyState === WebSocket.OPEN && client.close(sendableCode(code), reason));
    remote.on('error', (e) => client.readyState === WebSocket.OPEN && client.close(1011, String(e).slice(0, 120)));
    client.on('close', () => remote.readyState <= WebSocket.OPEN && remote.close());
  });
});

server.listen(PORT, '127.0.0.1', () => {
  const keys = Object.entries(upstreams)
    .map(([k, u]) => `${k}:${u.key ? 'ok' : 'missing'}`)
    .join(' ');
  console.log(`frankenvoice proxy on http://127.0.0.1:${PORT}  [${keys}]  sessions -> ${SESSIONS_DIR}`);
});
