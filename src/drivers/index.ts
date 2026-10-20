// Browser drivers: the only code that touches real devices and networks. Each
// one turns an IO exchange opened by a node into real-world calls and feeds
// whatever comes back into the sink; the runtime records all of it.

import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import sileroV5Url from '@ricky0123/vad-web/dist/silero_vad_v5.onnx?url';
import sileroV6Url from '@ricky0123/vad-web/dist/silero_vad_v6.onnx?url';
import fireredUrl from './firered/stream_vad.onnx?url';
import captureWorkletUrl from './capture-worklet.js?url';
import type { DriverRegistry, LiveDriver } from '../core/world';
import type { Json } from '../core/types';
import { toJson } from '../core/json';
import { loadEncoder, type Encoder } from '../embed/encoder';
import { fireredDriver, sileroDriver } from './vad';

/**
 * Built pages take onnxruntime's wasm (14 MB) from jsDelivr, cached worldwide, rather than from
 * wherever the page is hosted (the GPU box sits behind a tunnel). The dev server serves its own.
 */
export const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
ort.env.wasm.wasmPaths = import.meta.env.PROD ? ORT_CDN : { wasm: ortWasmUrl, mjs: ortMjsUrl };
ort.env.wasm.numThreads = 1;

/** Fetch what a live session loads first (the wasm, the VAD models) while the page is idle, so Run live finds them cached. */
export function warmUp() {
  const wasm = import.meta.env.PROD ? `${ORT_CDN}ort-wasm-simd-threaded.wasm` : ortWasmUrl;
  for (const url of [wasm, sileroV5Url, fireredUrl]) void fetch(url, { priority: 'low' } as RequestInit).catch(() => {});
}

const errorJson = (e: unknown): Json => ({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });

const mic: LiveDriver = (req, sink) => {
  const r = req as { sampleRate: number; frameSamples: number; echoCancellation: boolean; noiseSuppression: boolean; autoGainControl: boolean };
  let closed = false;
  let stream: MediaStream | undefined;
  // Capture at the target rate so the browser resamples with a proper filter (the worklet's own
  // resampler has none). Browsers that can't feed a mic into a context at another rate get the default.
  let ctx = new AudioContext({ sampleRate: r.sampleRate });
  (async () => {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: r.echoCancellation, noiseSuppression: r.noiseSuppression, autoGainControl: r.autoGainControl },
    });
    if (closed) return stream.getTracks().forEach((t) => t.stop());
    let src: MediaStreamAudioSourceNode;
    try {
      src = ctx.createMediaStreamSource(stream);
    } catch {
      void ctx.close();
      ctx = new AudioContext();
      src = ctx.createMediaStreamSource(stream);
    }
    await ctx.audioWorklet.addModule(captureWorkletUrl);
    const node = new AudioWorkletNode(ctx, 'frankenvoice-capture', { processorOptions: { targetRate: r.sampleRate, frameSamples: r.frameSamples } });
    const sinkGain = ctx.createGain();
    sinkGain.gain.value = 0;
    const track = stream.getAudioTracks()[0];
    sink.recv(toJson({ type: 'ready', label: track.label, contextSampleRate: ctx.sampleRate, settings: track.getSettings() }));
    node.port.onmessage = (e: MessageEvent<{ samples: Float32Array; frame: number; ctxTime: number }>) => {
      if (!closed) sink.recv({ frame: e.data.frame, ctxTime: e.data.ctxTime }, new Uint8Array(e.data.samples.buffer));
    };
    src.connect(node).connect(sinkGain).connect(ctx.destination);
  })().catch((e) => !closed && sink.close(errorJson(e)));
  return {
    send() {},
    close() {
      closed = true;
      stream?.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
};

const sileroModels: Record<string, string> = { v5: sileroV5Url, v6: sileroV6Url };
// A session per exchange, as before: a wasm session runs one inference at a time.
const silero = sileroDriver(ort, (model) => ort.InferenceSession.create(sileroModels[model] ?? sileroV5Url));
const firered = fireredDriver(ort, () => ort.InferenceSession.create(fireredUrl));

function wsUrl(upstream: string, path: string, query: Record<string, Json>): string {
  const u = new URL(`/api/${upstream}${path}`, location.href);
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  for (const [k, v] of Object.entries(query)) u.searchParams.set(k, String(v));
  return u.toString();
}

/** A streaming WebSocket through the proxy: JSON as text frames, audio as binary frames. `$open` is synthesized on connect. */
const proxiedSocket = (upstream: string): LiveDriver => (req, sink) => {
  const r = req as { path: string; query: Record<string, Json> };
  const ws = new WebSocket(wsUrl(upstream, r.path, r.query));
  ws.binaryType = 'arraybuffer';
  const pending: (string | ArrayBuffer)[] = [];
  let closedByNode = false;
  ws.onopen = () => {
    sink.recv({ type: '$open' });
    for (const p of pending.splice(0)) ws.send(p);
  };
  ws.onmessage = (e) => {
    if (typeof e.data === 'string') sink.recv(JSON.parse(e.data) as Json);
    else sink.recv(null, new Uint8Array(e.data as ArrayBuffer));
  };
  ws.onclose = (e) => {
    if (!closedByNode) sink.close({ code: e.code, reason: e.reason, ...(e.code !== 1000 ? { error: `websocket closed (${e.code})` } : {}) });
  };
  return {
    send(data, bin) {
      const payload = bin ? bin.slice().buffer : JSON.stringify(data);
      if (ws.readyState === WebSocket.OPEN) ws.send(payload);
      else if (ws.readyState === WebSocket.CONNECTING) pending.push(payload);
    },
    close() {
      closedByNode = true;
      if (ws.readyState === WebSocket.CONNECTING) ws.onopen = () => ws.close(1000);
      else ws.close(1000);
    },
  };
};

const gpubox = proxiedSocket('gpubox');

/** Read a fetch body as Server-Sent Events, yielding each event's data string. Comment lines are dropped. */
async function* sseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
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

async function postJson(url: string, body: Json, signal: AbortSignal): Promise<Response> {
  return fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal });
}

/**
 * A short title for a finished session, from its transcript. Called by the UI, outside the pipeline,
 * so it is not part of the recording.
 */
export async function sessionTitle(model: string, transcript: string): Promise<string> {
  const res = await postJson(
    '/api/llm/chat/completions',
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
    new AbortController().signal,
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const j = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  return (j.choices?.[0]?.message?.content ?? '').trim();
}

/** TypeSafe primitives (a Choice question): one JSON request, one JSON answer. */
const typesafe: LiveDriver = (req, sink) => {
  const r = req as { path: string; body: Json };
  const ac = new AbortController();
  (async () => {
    const res = await postJson(`/api/typesafe${r.path}`, r.body, ac.signal);
    const text = await res.text();
    if (!res.ok) return sink.close({ error: { status: res.status, body: text } });
    sink.recv(JSON.parse(text) as Json);
    sink.close(null);
  })().catch((e) => !ac.signal.aborted && sink.close(errorJson(e)));
  return { send() {}, close: () => ac.abort() };
};

/** OpenAI-compatible chat completions; each SSE data line is recorded parsed, `[DONE]` as {"$done": true}. */
const openai: LiveDriver = (req, sink) => {
  const r = req as { body: Json };
  const ac = new AbortController();
  (async () => {
    const res = await postJson('/api/llm/chat/completions', r.body, ac.signal);
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
 * RNNoise at 48 kHz: frames are upsampled, denoised in 480-sample blocks and
 * downsampled back. Each frame sent gets exactly one frame back of the same
 * length (the first ones lead with silence while the pipeline fills).
 */
const rnnoise: LiveDriver = (req, sink) => {
  const { sampleRate } = req as { sampleRate: number };
  const ratio = 48000 / sampleRate;
  let closed = false;
  if (!Number.isInteger(ratio)) {
    setTimeout(() => sink.close({ error: { message: `RNNoise needs a rate that divides 48 kHz, got ${sampleRate}` } }), 0);
    return { send() {}, close() {} };
  }
  const ready = import('@shiguredo/rnnoise-wasm').then(async ({ Rnnoise }) => (await Rnnoise.load()).createDenoiseState());
  ready.catch((e) => !closed && sink.close(errorJson(e)));
  let last = 0;
  let in48: number[] = [];
  let out48: number[] = [];
  let out: number[] = [];
  let chain = Promise.resolve();
  return {
    send(_data, bin) {
      if (!bin) return;
      const frame = new Float32Array(bin.slice().buffer);
      chain = chain.then(async () => {
        const st = await ready;
        if (closed) return;
        // upsample (linear) to 48 kHz, in 16-bit scale as RNNoise expects
        for (const v of frame) {
          for (let k = 1; k <= ratio; k++) in48.push((last + ((v - last) * k) / ratio) * 32768);
          last = v;
        }
        const block = new Float32Array(480);
        let vadSum = 0;
        let blocks = 0;
        while (in48.length >= 480) {
          block.set(in48.slice(0, 480));
          in48 = in48.slice(480);
          vadSum += st.processFrame(block);
          blocks++;
          for (const v of block) out48.push(v / 32768);
        }
        // downsample: average each group of `ratio` samples
        const n = Math.floor(out48.length / ratio);
        for (let i = 0; i < n; i++) {
          let a = 0;
          for (let k = 0; k < ratio; k++) a += out48[i * ratio + k];
          out.push(a / ratio);
        }
        out48 = out48.slice(n * ratio);
        const reply = new Float32Array(frame.length);
        const have = Math.min(out.length, frame.length);
        reply.set(out.slice(0, have), frame.length - have);
        out = out.slice(have);
        sink.recv({ vad: blocks ? vadSum / blocks : null }, new Uint8Array(reply.buffer));
      });
      chain.catch((e) => !closed && sink.close(errorJson(e)));
    },
    close() {
      closed = true;
      void ready.then((st) => st.destroy()).catch(() => {});
    },
  };
};

type TurnModelRequest = { encoderUrl: string; headUrl: string; frames: number; ep: string };
type WorkerMsg = { type: string; text?: string; [k: string]: unknown };

/** Where a loaded (or loading) turn model is, for the page to show before any session runs. */
export interface TurnModelLoad {
  /** 'loading' until compiled, then 'ready'; 'failed' with the error. */
  state: 'loading' | 'ready' | 'failed';
  /** The latest progress or status message from the worker. */
  last?: WorkerMsg;
  error?: string;
}

/** A turn model worker, kept across sessions so the model is downloaded and compiled once. */
interface TurnModelWorker {
  key: string;
  worker: Worker;
  load: TurnModelLoad;
  /** The session using it, if any: it gets the worker's messages. */
  session?: (m: WorkerMsg) => void;
}

const turnModels = new Map<string, TurnModelWorker>();
let turnModelResets = 0;
const turnModelWatchers = new Set<() => void>();

function startTurnModel(r: TurnModelRequest): TurnModelWorker {
  const key = JSON.stringify([r.encoderUrl, r.headUrl, r.frames, r.ep]);
  const existing = turnModels.get(key);
  if (existing) return existing;
  const worker = new Worker(new URL('./turn-model.worker.ts', import.meta.url), { type: 'module' });
  const tm: TurnModelWorker = { key, worker, load: { state: 'loading' } };
  const fail = (error: string) => {
    // Forget a failed model so the next session or preload starts over.
    if (turnModels.get(key) === tm) turnModels.delete(key);
    worker.terminate();
    tm.load = { state: 'failed', last: tm.load.last, error };
  };
  worker.onmessage = (e: MessageEvent<WorkerMsg>) => {
    const m = e.data;
    if (m.type === 'error') fail(m.text ?? 'turn model failed');
    else if (m.type === 'ready') tm.load = { state: 'ready', last: m };
    else if (m.type === 'progress' || m.type === 'status') tm.load = { state: 'loading', last: m };
    tm.session?.(m);
    if (m.type !== 'frames') for (const w of turnModelWatchers) w();
  };
  worker.onerror = (e) => {
    fail(e.message || 'turn model worker failed to start');
    tm.session?.({ type: 'error', text: tm.load.error });
    for (const w of turnModelWatchers) w();
  };
  worker.postMessage({ type: 'init', ...r });
  turnModels.set(key, tm);
  for (const w of turnModelWatchers) w();
  return tm;
}

/** Download and compile the turn model ahead of a session; returns its load state and a way to watch it. */
export function preloadTurnModel(r: TurnModelRequest): TurnModelLoad {
  return startTurnModel(r).load;
}

/** Load state of the turn model for these settings (undefined: never started). */
export function turnModelLoad(r: TurnModelRequest): TurnModelLoad | undefined {
  return turnModels.get(JSON.stringify([r.encoderUrl, r.headUrl, r.frames, r.ep]))?.load;
}

export function watchTurnModels(fn: () => void): () => void {
  turnModelWatchers.add(fn);
  return () => turnModelWatchers.delete(fn);
}

/**
 * The ssl_turn turn-taking model, in a worker that outlives the session (see preloadTurnModel).
 * Each send is one 80 ms block: 1280 user samples then 1280 agent samples (16 kHz). Replies:
 * progress while it still downloads, `ready` once compiled and reset for this session, then
 * `frames` with per-frame scores as inference catches up.
 */
const turnModel: LiveDriver = (req, sink) => {
  const r = req as TurnModelRequest;
  let tm = startTurnModel(r);
  // Another session (say, a replay running alongside) holds it: give this one its own worker.
  if (tm.session) {
    turnModels.delete(tm.key);
    tm = startTurnModel(r);
  }
  const own = tm;
  let closed = false;
  /** Frames are this session's only after the worker has been reset for it. */
  let ready = false;
  /** The reset sent for this session; only its `ready` counts (an earlier session's may still arrive). */
  let resetSeq: number | undefined;
  const release = () => {
    if (own.session === onMsg) own.session = undefined;
    // A second worker made for an overlapping session isn't kept.
    if (turnModels.get(own.key) !== own) own.worker.terminate();
  };
  const onMsg = (m: WorkerMsg) => {
    if (closed) return;
    if (m.type === 'error') {
      closed = true;
      release();
      sink.close({ error: m.text ?? 'turn model failed' });
    } else if (m.type === 'ready') {
      if (m.seq !== resetSeq) return;
      ready = true;
      sink.recv(toJson(m));
    } else if (m.type !== 'frames' || ready) sink.recv(toJson(m));
  };
  own.session = onMsg;
  if (own.load.state === 'ready') {
    setTimeout(() => !closed && sink.recv({ type: 'status', text: 'preloaded' }), 0);
    resetSeq = ++turnModelResets;
    own.worker.postMessage({ type: 'reset', seq: resetSeq });
  } else if (own.load.last?.type === 'progress' || own.load.last?.type === 'status') {
    // Still loading from a preload: pick up where it is.
    const last = own.load.last;
    setTimeout(() => !closed && !ready && sink.recv(toJson(last)), 0);
  }
  return {
    send(_data, bin) {
      if (closed || !bin || !ready) return;
      const x = new Float32Array(bin.slice().buffer);
      const a = x.slice(0, x.length / 2);
      const b = x.slice(x.length / 2);
      own.worker.postMessage({ type: 'block', a, b }, [a.buffer, b.buffer]);
    },
    close() {
      closed = true;
      release();
    },
  };
};

/** Output audio contexts, so a paused replay can freeze what is already scheduled. */
const outputContexts = new Set<AudioContext>();

/** While set, outputs drop what they are asked to play: a replay is skipping ahead. */
let outputSuppressed = false;
const playingSources = new Set<AudioBufferSourceNode>();
/** Outputs holding audio scheduled during a skip; each plays the part still ahead once the skip ends. */
const heldOutputs = new Set<() => void>();

/**
 * Silence outputs while a replay fast-forwards, cutting off whatever is playing.
 * Audio scheduled to play at a session time is held instead of dropped, so when
 * the skip lands in the middle of the agent speaking, the rest of it still plays.
 */
export function setOutputSuppressed(on: boolean) {
  outputSuppressed = on;
  if (!on) {
    for (const release of heldOutputs) release();
    heldOutputs.clear();
    return;
  }
  for (const s of playingSources) s.stop();
  playingSources.clear();
}

/** Muted outputs: the agent's voice and the recorded mic during replay. */
const muted = { agent: false, mic: false };
const gains = { agent: new Set<GainNode>(), mic: new Set<GainNode>() };

export function setOutputMuted(which: 'agent' | 'mic', on: boolean) {
  muted[which] = on;
  for (const g of gains[which]) g.gain.value = on ? 0 : 1;
}

/** An output context whose audio goes through a gain that follows the mute switch for `which`. */
function outputContext(which: 'agent' | 'mic', sampleRate?: number) {
  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ latencyHint: 'interactive', sampleRate });
  } catch {
    ctx = new AudioContext({ latencyHint: 'interactive' });
  }
  const gain = ctx.createGain();
  gain.gain.value = muted[which] ? 0 : 1;
  gain.connect(ctx.destination);
  outputContexts.add(ctx);
  gains[which].add(gain);
  return {
    ctx,
    out: gain,
    close() {
      outputContexts.delete(ctx);
      gains[which].delete(gain);
      void ctx.close();
    },
  };
}

/**
 * Plays a stream of PCM chunks. The context runs at the audio's own rate, so no chunk is resampled on
 * its own (that leaves a click at every chunk edge), and a chunk that continues the previous one starts
 * exactly where it ended on the audio clock instead of at a time converted from the session clock,
 * which jitters by a millisecond or so and left tiny gaps and overlaps: a steady crackle.
 */
function pcmOutput(which: 'agent' | 'mic', clock?: () => number) {
  let o: ReturnType<typeof outputContext> | null = null;
  let next = 0; // audio-clock end of the last chunk
  let nextAt = NaN; // and its session time, when chunks carry one
  const playing = new Set<AudioBufferSourceNode>();
  // Ducking, ahead of the mute switch; kept across a context change.
  let level = 1;
  let duck: GainNode | null = null;
  /** The sources' fade-out at a cut; each cut gets a fresh one, so its ramp never reaches later audio. */
  let fade: GainNode | null = null;
  /** When each source plays: context time, and session time when it was given one. */
  const when = new WeakMap<AudioBufferSourceNode, { t: number; at?: number; dur: number }>();
  /** Timed chunks that arrived while a replay was skipping ahead (needs `clock`). */
  let held: { samples: Float32Array<ArrayBuffer>; sampleRate: number; at: number }[] = [];
  const release = () => {
    const chunks = held;
    held = [];
    const now = clock!();
    for (const c of chunks) {
      const cut = Math.max(0, Math.round(((now - c.at) / 1000) * c.sampleRate));
      if (cut >= c.samples.length) continue;
      play(c.samples.subarray(cut), c.sampleRate, Math.max(c.at, now), now);
    }
  };
  /**
   * Stop everything: now, or at session time `at` (ms, with the session clock `now`) after fading
   * out over `releaseMs`, or with `tail` taking over at `at` (sample-exact, in the same context).
   */
  const stop = (at?: number, now?: number, releaseMs = 0, tail?: { samples: Float32Array<ArrayBuffer>; sampleRate: number; stop: number }) => {
    // A tail this context can't play (another sample rate): fade out where the sound ends instead.
    if (tail && o && o.ctx.sampleRate !== tail.sampleRate) [at, tail] = [tail.stop, undefined];
    // While a replay skips ahead nothing is heard: no fade, no tail.
    if (o && duck && fade && at !== undefined && now !== undefined && !outputSuppressed) {
      // The context time of `at`, from the source playing then, so the cut lands on its samples.
      let t = o.ctx.currentTime + Math.max(0, (at - now) / 1000);
      for (const src of playing) {
        const w = when.get(src);
        if (w?.at !== undefined && w.at <= at && at < w.at + w.dur * 1000) t = Math.max(o.ctx.currentTime, w.t + (at - w.at) / 1000);
      }
      if (tail) {
        for (const src of playing) src.stop(t);
        const buf = o.ctx.createBuffer(1, tail.samples.length, tail.sampleRate);
        buf.copyToChannel(tail.samples, 0);
        const src = o.ctx.createBufferSource();
        src.buffer = buf;
        // Past the fade node: a later cut doesn't touch it.
        src.connect(duck);
        src.onended = () => playingSources.delete(src);
        playingSources.add(src);
        src.start(t);
      } else {
        const r = Math.max(0.001, releaseMs / 1000);
        fade.gain.setValueAtTime(1, t);
        fade.gain.linearRampToValueAtTime(0, t + r);
        for (const src of playing) src.stop(t + r);
        fade = o.ctx.createGain();
        fade.connect(duck);
      }
    } else for (const src of playing) src.stop();
    playing.clear();
    nextAt = NaN;
    held = [];
  };
  /** Play `samples`; at session time `at` (ms, with the session clock `now`), or else right after the previous chunk. */
  function play(samples: Float32Array<ArrayBuffer>, sampleRate: number, at?: number, now?: number) {
    if (!samples.length) return;
    if (outputSuppressed) {
      if (at === undefined || !clock) return;
      held.push({ samples, sampleRate, at });
      heldOutputs.add(release);
      return;
    }
    if (!o || o.ctx.sampleRate !== sampleRate) {
      stop();
      o?.close();
      o = outputContext(which, sampleRate);
      duck = o.ctx.createGain();
      duck.gain.value = level;
      duck.connect(o.out);
      fade = o.ctx.createGain();
      fade.connect(duck);
    }
    const { ctx } = o;
    const out = fade!;
    const buf = ctx.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(out);
    src.onended = () => {
      playing.delete(src);
      playingSources.delete(src);
    };
    playing.add(src);
    playingSources.add(src);
    const live = next >= ctx.currentTime;
    let t: number;
    if (at === undefined) t = live && next <= ctx.currentTime + 0.5 ? next : ctx.currentTime + 0.08;
    else if (live && Math.abs(at - nextAt) < 2) t = next;
    else t = Math.max(ctx.currentTime, ctx.currentTime + (at - now!) / 1000);
    src.start(t);
    when.set(src, { t, at, dur: buf.duration });
    next = t + buf.duration;
    nextAt = at === undefined ? NaN : at + buf.duration * 1000;
  }
  return {
    play,
    stop,
    /** Fade to `g` (linear) over a few tens of ms, so the step doesn't click. */
    setGain(g: number) {
      level = g;
      if (o && duck) duck.gain.setTargetAtTime(g, o.ctx.currentTime, 0.015);
    },
    close() {
      stop();
      o?.close();
      o = null;
      duck = null;
      fade = null;
    },
  };
}

/** Plays the recorded mic while a replay runs in real time, so both sides of the conversation are heard. */
export function micMonitor() {
  const out = pcmOutput('mic');
  return { play: (samples: Float32Array<ArrayBuffer>, sampleRate: number) => out.play(samples, sampleRate), close: out.close };
}

/** Pause or resume every playing output. */
export function setOutputPaused(paused: boolean) {
  for (const ctx of outputContexts) void (paused ? ctx.suspend() : ctx.resume());
}

const speaker: LiveDriver = (_req, sink, env) => {
  const out = pcmOutput('agent', env.now);
  // Backchannels play on their own output, so they neither wait for nor shift the reply's audio.
  const cues = pcmOutput('agent', env.now);
  setTimeout(() => sink.recv({ type: 'ready' }), 0);
  return {
    send(data, bin) {
      const d = data as { cmd: string; at?: number; stop?: number; sampleRate?: number; gain?: number; releaseMs?: number };
      if (d.cmd === 'cue' && bin) return cues.play(new Float32Array(bin.slice().buffer), d.sampleRate!, d.at!, env.now());
      if (d.cmd === 'flush') {
        const tail = bin && d.sampleRate ? { samples: new Float32Array(bin.slice().buffer), sampleRate: d.sampleRate, stop: d.stop ?? d.at! } : undefined;
        return d.at === undefined ? out.stop() : out.stop(d.at, env.now(), d.releaseMs, tail);
      }
      if (d.cmd === 'gain') return out.setGain(d.gain ?? 1);
      if (d.cmd !== 'play' || !bin) return;
      out.play(new Float32Array(bin.slice().buffer), d.sampleRate!, d.at!, env.now());
    },
    close: () => {
      out.close();
      cues.close();
    },
  };
};

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
      return loadEncoder(ort as never, new Uint8Array(onnx), vocab);
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

export const browserDrivers: DriverRegistry = { mic, silero, firered, openai, 'gpubox-asr': gpubox, 'gpubox-tts': gpubox, rnnoise, speaker, 'turn-model': turnModel, embed, typesafe };
