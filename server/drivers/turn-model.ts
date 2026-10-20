// The ssl_turn turn-taking model on onnxruntime-node (CPU). A model is downloaded once (cached in
// the models folder), compiled once, and kept across sessions; each session resets its stream.

import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import * as ort from 'onnxruntime-node';
import type { LiveDriver } from '../../src/core/world';
import type { Json } from '../../src/core/types';
import type { TurnModelLoad } from '../../src/live-protocol';
import { TurnStream } from './turn-stream';

/** Blocks (3.2 s) of unprocessed audio before giving up: the model can't keep up. */
const MAX_BACKLOG = 40;

export interface TurnModelRequest {
  encoderUrl: string;
  headUrl: string;
  frames: number;
  ep?: string;
}
type Msg = { type: string; [k: string]: Json };


interface Model {
  load: TurnModelLoad;
  stream: Promise<TurnStream>;
  /** The session using it gets its messages. */
  session?: (m: Msg) => void;
  /** Inference and resets, one at a time. */
  chain: Promise<void>;
}

const models = new Map<string, Model>();
const keyOf = (r: TurnModelRequest) => JSON.stringify([r.encoderUrl, r.headUrl, r.frames]);

async function fetchCached(dir: string, url: string, label: string, post: (m: Msg) => void): Promise<Uint8Array> {
  const file = join(dir, createHash('sha256').update(url).digest('hex').slice(0, 24) + '.onnx');
  try {
    const b = new Uint8Array(await readFile(file));
    post({ type: 'progress', label, done: 1, total: 1, cached: true });
    return b;
  } catch {
    /* not cached yet */
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
  const out = Buffer.concat(parts);
  await mkdir(dir, { recursive: true });
  await writeFile(file + '.part', out);
  await rename(file + '.part', file);
  return new Uint8Array(out);
}

function start(dir: string, r: TurnModelRequest): Model {
  const key = keyOf(r);
  const existing = models.get(key);
  if (existing) return existing;
  const m: Model = { load: { state: 'loading' }, stream: null!, chain: Promise.resolve() };
  const post = (msg: Msg) => {
    if (msg.type === 'progress' || msg.type === 'status') m.load = { state: 'loading', last: msg };
    m.session?.(msg);
  };
  m.stream = (async () => {
    const enc = await fetchCached(dir, r.encoderUrl, 'encoder', post);
    const head = await fetchCached(dir, r.headUrl, 'head', post);
    post({ type: 'status', text: 'compiling for cpu' });
    const s = await TurnStream.create(ort, { encoder: enc, head, frames: r.frames });
    // Warm-up, so the first real call is not slower.
    for (let i = 0; i < r.frames; i++) await s.push(new Float32Array(1280), new Float32Array(1280));
    s.reset();
    m.load = { state: 'ready', last: { type: 'ready', ep: 'cpu', frames: r.frames } };
    return s;
  })();
  m.stream.catch((e: unknown) => {
    // Forget a failed model so the next session or preload starts over.
    if (models.get(key) === m) models.delete(key);
    m.load = { state: 'failed', last: m.load.last, error: e instanceof Error ? e.message : String(e) };
  });
  models.set(key, m);
  return m;
}

/** Download and compile the turn model ahead of a session; returns its load state. */
export function preloadTurnModel(dir: string, r: TurnModelRequest): TurnModelLoad {
  return start(dir, r).load;
}

/**
 * Each send is one 80 ms block: 1280 user samples then 1280 agent samples (16 kHz). Replies:
 * progress while it still downloads, `ready` once compiled and reset for this session, then
 * `frames` with per-frame scores as inference catches up.
 */
export const turnModelDriver =
  (dir: string): LiveDriver =>
  (req, sink) => {
    const m = start(dir, req as unknown as TurnModelRequest);
    let closed = false;
    let ready = false;
    let backlog = 0;
    const fail = (error: string) => {
      if (closed) return;
      closed = true;
      if (m.session === onMsg) m.session = undefined;
      sink.close({ error });
    };
    const onMsg = (msg: Msg) => !closed && !ready && sink.recv(msg);
    // A second session on the same model (a replay alongside) takes it over.
    m.session = onMsg;
    if (m.load.state === 'ready') setTimeout(() => !closed && sink.recv({ type: 'status', text: 'preloaded' }), 0);
    else if (m.load.last) {
      const last = m.load.last;
      setTimeout(() => !closed && !ready && sink.recv(last), 0);
    }
    m.chain = m.chain.then(async () => {
      let s: TurnStream;
      try {
        s = await m.stream;
      } catch {
        return fail(m.load.error ?? 'turn model failed');
      }
      if (closed) return;
      s.reset();
      ready = true;
      sink.recv({ type: 'ready', ep: 'cpu', frames: (req as unknown as TurnModelRequest).frames, threads: 0 });
    });
    return {
      send(_data, bin) {
        if (closed || !bin || !ready) return;
        if (++backlog > MAX_BACKLOG) return fail(`the turn model can't keep up (over ${MAX_BACKLOG} blocks behind)`);
        const x = new Float32Array(bin.slice().buffer);
        m.chain = m.chain.then(async () => {
          if (closed) return;
          const s = await m.stream;
          const t = performance.now();
          try {
            const outs = await s.push(x.slice(0, x.length / 2), x.slice(x.length / 2));
            backlog--;
            if (outs.length && !closed) sink.recv({ type: 'frames', outs: outs as unknown as Json, ms: Math.round(performance.now() - t), backlog });
          } catch (e) {
            fail(`inference failed: ${e instanceof Error ? e.message : String(e)}`);
          }
        });
      },
      close() {
        closed = true;
        if (m.session === onMsg) m.session = undefined;
      },
    };
  };
