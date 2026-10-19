// Turn model inference worker: downloads the ONNX files (kept with the Cache API), then runs
// TurnStream on every 80 ms stereo block (user, agent) and posts the per-frame scores back.

import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import { TurnStream } from './turn-stream';

ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

/** Blocks (3.2 s) of unprocessed audio before giving up: the model can't keep up. */
const MAX_BACKLOG = 40;
const CACHE = 'frankenvoice-turn-model';

export type ToWorker =
  | { type: 'init'; encoderUrl: string; headUrl: string; frames: number; ep: string }
  | { type: 'block'; a: Float32Array; b: Float32Array }
  /** A new session: forget the audio so far, then say `ready` again. */
  | { type: 'reset' };

let ts: TurnStream | null = null;
let queue: { a: Float32Array; b: Float32Array }[] = [];
let busy = false;
let dead = false;
let readyMsg: unknown = null;

const post = (m: unknown) => postMessage(m);

async function fetchCached(url: string, label: string): Promise<Uint8Array> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE);
  } catch {
    /* no Cache API */
  }
  const hit = cache && (await cache.match(url));
  if (hit) {
    post({ type: 'progress', label, done: 1, total: 1, cached: true });
    return new Uint8Array(await hit.arrayBuffer());
  }
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`could not download ${url} (HTTP ${r.status})`);
  const total = +(r.headers.get('content-length') ?? 0);
  const reader = r.body.getReader();
  const parts: Uint8Array[] = [];
  let done = 0;
  let lastPct = -1;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    parts.push(value);
    done += value.length;
    // Whole percents: every message is recorded in the session log.
    const pct = total ? Math.floor((done / total) * 100) : Math.floor(done / 2 ** 20);
    if (pct !== lastPct) {
      lastPct = pct;
      post({ type: 'progress', label, done, total });
    }
  }
  const out = new Uint8Array(done);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  if (cache) cache.put(url, new Response(out.slice())).catch(() => {});
  return out;
}

async function pickEp(want: string): Promise<string> {
  if (want !== 'auto') return want;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<unknown> } }).gpu;
  const a = gpu && (await gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null));
  return a ? 'webgpu' : 'wasm';
}

async function init(m: Extract<ToWorker, { type: 'init' }>) {
  const ep = await pickEp(m.ep);
  const enc = await fetchCached(m.encoderUrl, 'encoder');
  const head = await fetchCached(m.headUrl, 'head');
  post({ type: 'status', text: `compiling for ${ep}` });
  const s = await TurnStream.create(ort, { encoder: enc, head, frames: m.frames, ep });
  // Warm-up so the first real call does not pay shader compilation.
  for (let i = 0; i < m.frames; i++) await s.push(new Float32Array(1280), new Float32Array(1280));
  s.reset();
  ts = s;
  readyMsg = { type: 'ready', ep, frames: m.frames, threads: ort.env.wasm.numThreads ?? 1 };
  post(readyMsg);
}

async function reset() {
  if (!ts) return; // still loading: `ready` follows when it is done
  queue = [];
  while (busy) await new Promise((r) => setTimeout(r, 5));
  ts.reset();
  dead = false;
  post(readyMsg);
}

async function pump() {
  if (busy || !ts) return;
  busy = true;
  try {
    while (queue.length && !dead) {
      const blk = queue.shift()!;
      const s = performance.now();
      const outs = await ts.push(blk.a, blk.b);
      if (outs.length) post({ type: 'frames', outs, ms: Math.round(performance.now() - s), backlog: queue.length });
    }
  } catch (e) {
    dead = true;
    post({ type: 'error', text: `inference failed: ${e instanceof Error ? e.message : String(e)}` });
  } finally {
    busy = false;
  }
}

onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === 'init') init(m).catch((err) => post({ type: 'error', text: err instanceof Error ? err.message : String(err) }));
  else if (m.type === 'reset') void reset();
  else if (m.type === 'block' && !dead) {
    queue.push({ a: m.a, b: m.b });
    if (queue.length > MAX_BACKLOG) {
      dead = true;
      queue = [];
      post({ type: 'error', text: `the turn model can't keep up (over ${MAX_BACKLOG} blocks behind)` });
      return;
    }
    void pump();
  }
};
