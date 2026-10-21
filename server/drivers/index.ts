// Server drivers: the code that turns a node's IO exchanges into model runs and provider calls.
// Same channels and message shapes the page used to have, so recordings replay unchanged. The
// mic and the speaker are the page's (server/live.ts); everything else runs here.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { WebSocket } from 'ws';
import type { DriverRegistry, LiveDriver } from '../../src/core/world';
import type { Json } from '../../src/core/types';
import { loadEncoder, type Encoder } from '../embed/encoder';
import { fireredDriver, sileroDriver } from './vad';
import { turnModelDriver } from './turn-model';
import { rnnoiseDriver } from './rnnoise';
import { cannedDriver } from './canned';

export type UpstreamName = 'gpubox' | 'llm' | 'typesafe';
/** Where an upstream is and how to authenticate; `missing` says what isn't configured. */
export interface Upstream {
  base?: string;
  headers: Record<string, string>;
  missing?: string;
}
export type Upstreams = (name: UpstreamName) => Upstream;

export const errorJson = (e: unknown): Json => ({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });

const asset = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
const MODELS: Record<string, string> = {
  v5: asset('../../node_modules/@ricky0123/vad-web/dist/silero_vad_v5.onnx'),
  v6: asset('../../node_modules/@ricky0123/vad-web/dist/silero_vad_v6.onnx'),
  firered: asset('./firered/stream_vad.onnx'),
};
/**
 * Small models run often (a VAD frame every 32 ms) next to the GPU services: one thread each, and no
 * spin-waiting between runs (onnxruntime's default spins its pool, burning the CPU the gateway needs).
 */
export const quietSession = (threads = 1): ort.InferenceSession.SessionOptions => ({
  intraOpNumThreads: threads,
  interOpNumThreads: 1,
  extra: { session: { intra_op: { allow_spinning: '0' }, inter_op: { allow_spinning: '0' } } },
});
const modelBytes = new Map<string, Promise<Uint8Array>>();
const loadModel = (file: string) => {
  let b = modelBytes.get(file);
  if (!b) modelBytes.set(file, (b = readFile(file).then((x) => new Uint8Array(x))));
  return b.then((x) => ort.InferenceSession.create(x, quietSession()));
};

/** Read a fetch body as Server-Sent Events, yielding each event's data string. Comment lines are dropped. */
export async function* sseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let data: string[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (value) buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, '');
      buf = buf.slice(nl + 1);
      if (line === '') {
        if (data.length) yield data.join('\n');
        data = [];
      } else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (done) {
      if (data.length) yield data.join('\n');
      return;
    }
  }
}

/** POST JSON to an upstream; a missing setting answers 500 as the proxy did, so the node sees the same error. */
async function post(up: Upstream, path: string, body: Json, signal: AbortSignal): Promise<Response> {
  if (up.missing) return new Response(JSON.stringify({ error: up.missing }), { status: 500 });
  // A ws(s) base (the GPU box) serves its REST endpoints over http(s).
  const base = up.base!.replace(/^ws(s?):/, 'http$1:');
  return fetch(base + path, { method: 'POST', headers: { 'content-type': 'application/json', ...up.headers }, body: JSON.stringify(body), signal });
}

/** A streaming WebSocket to the upstream: JSON as text frames, audio as binary frames. `$open` is synthesized on connect. */
const socket =
  (up: Upstreams, name: UpstreamName): LiveDriver =>
  (req, sink) => {
    const r = req as { path: string; query: Record<string, Json> };
    const u = up(name);
    let done = false;
    const close = (data: Json) => {
      if (done) return;
      done = true;
      sink.close(data);
    };
    if (u.missing) {
      setTimeout(() => close({ code: 1011, reason: u.missing!.slice(0, 120), error: 'websocket closed (1011)' }), 0);
      return { send() {}, close() {} };
    }
    const url = new URL(u.base! + r.path);
    for (const [k, v] of Object.entries(r.query ?? {})) url.searchParams.set(k, String(v));
    const ws = new WebSocket(url, { headers: u.headers });
    const pending: { data: string | Uint8Array; binary: boolean }[] = [];
    let closedByNode = false;
    ws.on('open', () => {
      if (closedByNode) return ws.close(1000);
      sink.recv({ type: '$open' });
      for (const p of pending.splice(0)) ws.send(p.data, { binary: p.binary });
    });
    ws.on('message', (data: Buffer, binary) => {
      if (closedByNode) return;
      if (!binary) sink.recv(JSON.parse(data.toString('utf8')) as Json);
      else sink.recv(null, new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice());
    });
    ws.on('unexpected-response', (_req, res) => {
      if (!closedByNode) close({ code: 1011, reason: `${name} HTTP ${res.statusCode}`, error: 'websocket closed (1011)' });
      ws.terminate();
    });
    ws.on('error', (e) => !closedByNode && close({ code: 1006, reason: String(e).slice(0, 120), error: 'websocket closed (1006)' }));
    ws.on('close', (code, reason) => {
      if (!closedByNode) close({ code, reason: reason.toString('utf8'), ...(code !== 1000 ? { error: `websocket closed (${code})` } : {}) });
    });
    return {
      send(data, bin) {
        const p = bin ? { data: bin.slice(), binary: true } : { data: JSON.stringify(data), binary: false };
        if (ws.readyState === WebSocket.OPEN) ws.send(p.data, { binary: p.binary });
        else if (ws.readyState === WebSocket.CONNECTING) pending.push(p);
      },
      close() {
        closedByNode = true;
        if (ws.readyState === WebSocket.OPEN) ws.close(1000);
      },
    };
  };

/** TypeSafe primitives (a Choice question): one JSON request, one JSON answer. */
const typesafe =
  (up: Upstreams): LiveDriver =>
  (req, sink) => {
    const r = req as { path: string; body: Json };
    const ac = new AbortController();
    (async () => {
      const res = await post(up('typesafe'), r.path, r.body, ac.signal);
      const text = await res.text();
      if (!res.ok) return sink.close({ error: { status: res.status, body: text } });
      sink.recv(JSON.parse(text) as Json);
      sink.close(null);
    })().catch((e) => !ac.signal.aborted && sink.close(errorJson(e)));
    return { send() {}, close: () => ac.abort() };
  };

/** OpenAI-compatible chat completions; each SSE data line is recorded parsed, `[DONE]` as {"$done": true}. */
const openai =
  (up: Upstreams): LiveDriver =>
  (req, sink) => {
    const r = req as { body: Json };
    const ac = new AbortController();
    (async () => {
      const res = await post(up('llm'), '/chat/completions', r.body, ac.signal);
      if (!res.ok || !res.body) return sink.close({ error: { status: res.status, body: await res.text() } });
      for await (const data of sseData(res.body)) {
        if (data === '[DONE]') sink.recv({ $done: true });
        else {
          try {
            sink.recv(JSON.parse(data) as Json);
          } catch {
            sink.recv({ $unparsed: data });
          }
        }
      }
      sink.close(null);
    })().catch((e) => !ac.signal.aborted && sink.close(errorJson(e)));
    return { send() {}, close: () => ac.abort() };
  };

/**
 * llama.cpp's own /completion (the tape: a raw prompt under a grammar), on the server root of the `llm`
 * upstream (its base ends in /v1). One JSON answer, recorded whole.
 */
const completion =
  (up: Upstreams): LiveDriver =>
  (req, sink) => {
    const r = req as { body: Json };
    const ac = new AbortController();
    const u = up('llm');
    (async () => {
      const res = await post({ ...u, base: u.base?.replace(/\/v1\/?$/, '') }, '/completion', r.body, ac.signal);
      const text = await res.text();
      if (!res.ok) return sink.close({ error: { status: res.status, body: text.slice(0, 2000) } });
      // The answer without the echoed prompt and settings: they would double the recording. The token
      // odds (n_probs) stay: the Tape inspector reads Gemma's odds for each action from them.
      const j = JSON.parse(text) as Record<string, Json>;
      const keep = ['content', 'completion_probabilities', 'stop_type', 'stopping_word', 'tokens_cached', 'tokens_evaluated', 'truncated', 'timings', 'id_slot'];
      sink.recv(Object.fromEntries(keep.filter((k) => k in j).map((k) => [k, j[k]])));
      sink.close(null);
    })().catch((e) => !ac.signal.aborted && sink.close(errorJson(e)));
    return { send() {}, close: () => ac.abort() };
  };

/** A short title for a finished session, from its transcript. Outside the pipeline, so not recorded. */
export async function sessionTitle(up: Upstreams, model: string, transcript: string): Promise<string> {
  const res = await post(
    up('llm'),
    '/chat/completions',
    {
      model,
      max_tokens: 40,
      messages: [
        {
          role: 'system',
          content:
            'You name recorded voice conversations between a user and a voice agent. Reply with a title of 2 to 6 words that says what the conversation was about, in the language of the conversation. No quotes, no trailing punctuation.',
        },
        { role: 'user', content: `<transcript>\n${transcript}\n</transcript>\n\nTitle:` },
      ],
    },
    AbortSignal.timeout(20000),
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const j = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  return (j.choices?.[0]?.message?.content ?? '').trim();
}

// Sentence encoders for picking backchannel lines, loaded once per model from Hugging Face.
const encoders = new Map<string, Promise<Encoder>>();
function encoder(model: string): Promise<Encoder> {
  let e = encoders.get(model);
  if (!e) {
    const base = `https://huggingface.co/${model}/resolve/main`;
    e = (async () => {
      const [onnx, vocab] = await Promise.all([
        fetch(`${base}/onnx/model_quantized.onnx`).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${model}: HTTP ${r.status}`)))),
        fetch(`${base}/vocab.txt`).then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${model} vocab: HTTP ${r.status}`)))),
      ]);
      return loadEncoder(ort, new Uint8Array(onnx), vocab, quietSession());
    })();
    e.catch(() => encoders.delete(model));
    encoders.set(model, e);
  }
  return e;
}

const embed: LiveDriver = (req, sink) => {
  const r = req as { model: string; texts: string[] };
  let closed = false;
  encoder(r.model)
    .then((enc) => enc.embed(r.texts))
    .then((vecs) => {
      if (closed) return;
      // Rounded: plenty for cosine similarity, and a smaller recording.
      sink.recv({ vectors: vecs.map((v) => v.map((x) => Math.round(x * 1e5) / 1e5)) });
      sink.close(null);
    })
    .catch((e) => !closed && sink.close(errorJson(e)));
  return {
    send() {},
    close() {
      closed = true;
    },
  };
};

/**
 * Everything but the mic and the speaker. `modelsDir` caches downloaded models (the turn model),
 * `cannedDir` the canned lines' audio.
 */
export function serverDrivers(up: Upstreams, o: { modelsDir: string; cannedDir: string }): DriverRegistry {
  const gpubox = socket(up, 'gpubox');
  return {
    silero: sileroDriver(ort, (model) => loadModel(MODELS[model] ?? MODELS.v5)),
    firered: fireredDriver(ort, () => loadModel(MODELS.firered)),
    openai: openai(up),
    'llama-completion': completion(up),
    'gpubox-asr': gpubox,
    'gpubox-tts': gpubox,
    'tts-canned': cannedDriver(up, o.cannedDir),
    typesafe: typesafe(up),
    rnnoise: rnnoiseDriver,
    'turn-model': turnModelDriver(o.modelsDir),
    embed,
  };
}
