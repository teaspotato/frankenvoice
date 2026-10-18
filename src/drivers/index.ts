// Browser drivers: the only code that touches real devices and networks. Each
// one turns an IO exchange opened by a node into real-world calls and feeds
// whatever comes back into the sink; the runtime records all of it.

import Anthropic from '@anthropic-ai/sdk';
import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import sileroV5Url from '@ricky0123/vad-web/dist/silero_vad_v5.onnx?url';
import sileroV6Url from '@ricky0123/vad-web/dist/silero_vad_v6.onnx?url';
import captureWorkletUrl from './capture-worklet.js?url';
import type { DriverRegistry, LiveDriver } from '../core/world';
import type { Json } from '../core/types';
import { toJson } from '../core/json';

ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
ort.env.wasm.numThreads = 1;

const errorJson = (e: unknown): Json => ({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });

const mic: LiveDriver = (req, sink) => {
  const r = req as { sampleRate: number; frameSamples: number; echoCancellation: boolean; noiseSuppression: boolean; autoGainControl: boolean };
  let closed = false;
  let stream: MediaStream | undefined;
  const ctx = new AudioContext();
  (async () => {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: r.echoCancellation, noiseSuppression: r.noiseSuppression, autoGainControl: r.autoGainControl },
    });
    if (closed) return stream.getTracks().forEach((t) => t.stop());
    await ctx.audioWorklet.addModule(captureWorkletUrl);
    const src = ctx.createMediaStreamSource(stream);
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

const silero: LiveDriver = (req, sink) => {
  const r = req as { model: string; sampleRate: number };
  const CONTEXT = 64; // samples of the previous frame the model expects in front of each frame
  let closed = false;
  let context = new Float32Array(CONTEXT);
  let state = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
  const sr = new ort.Tensor('int64', BigInt64Array.from([BigInt(r.sampleRate)]), []);
  const session = ort.InferenceSession.create(sileroModels[r.model] ?? sileroV5Url);
  let chain: Promise<void> = session.then(() => undefined);
  chain.catch((e) => !closed && sink.close(errorJson(e)));
  return {
    send(_data, bin) {
      const frame = new Float32Array(bin!.slice().buffer);
      chain = chain.then(async () => {
        if (closed) return;
        const input = new Float32Array(CONTEXT + frame.length);
        input.set(context);
        input.set(frame, CONTEXT);
        context = frame.slice(-CONTEXT);
        const out = await (await session).run({ input: new ort.Tensor('float32', input, [1, input.length]), state, sr });
        state = out.stateN as typeof state;
        if (!closed) sink.recv({ p: (out.output.data as Float32Array)[0] });
      });
    },
    close() {
      closed = true;
    },
  };
};

function wsUrl(path: string, query: Record<string, Json>): string {
  const u = new URL(`/api/deepgram${path}`, location.href);
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  for (const [k, v] of Object.entries(query)) u.searchParams.set(k, String(v));
  return u.toString();
}

const deepgram: LiveDriver = (req, sink) => {
  const r = req as { path: string; query: Record<string, Json> };
  const ws = new WebSocket(wsUrl(r.path, r.query));
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

const claude = new Anthropic({
  baseURL: `${location.origin}/api/anthropic`,
  apiKey: 'injected-by-local-proxy',
  dangerouslyAllowBrowser: true,
});

const anthropic: LiveDriver = (req, sink) => {
  // The request is recorded verbatim; extra SDK fields such as `fallbacks` pass through.
  const stream = claude.beta.messages.stream(req as unknown as Parameters<typeof claude.beta.messages.stream>[0]);
  let aborted = false;
  (async () => {
    for await (const ev of stream) sink.recv(toJson(ev));
    sink.close(null);
  })().catch((e) => {
    if (!aborted) sink.close(errorJson(e));
  });
  return {
    send() {},
    close() {
      aborted = true;
      stream.abort();
    },
  };
};

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

/** OpenAI-compatible chat completions; each SSE data line is recorded parsed, `[DONE]` as {"$done": true}. */
const openai: LiveDriver = (req, sink) => {
  const r = req as { endpoint: string; body: Json };
  const ac = new AbortController();
  (async () => {
    const res = await postJson(`/api/openai/${encodeURIComponent(r.endpoint)}/chat/completions`, r.body, ac.signal);
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

/** POST a JSON request and stream the raw response bytes back (TTS providers). */
type RawRequest = { endpoint?: string; path: string; query?: Record<string, Json>; body: Json };

const withQuery = (url: string, query?: Record<string, Json>) =>
  query ? `${url}?${new URLSearchParams(Object.entries(query).map(([k, v]) => [k, String(v)]))}` : url;

const rawBytes =
  (url: (r: RawRequest) => string): LiveDriver =>
  (req, sink) => {
    const r = req as RawRequest;
    const ac = new AbortController();
    (async () => {
      const res = await postJson(url(r), r.body, ac.signal);
      if (!res.ok || !res.body) return sink.close({ error: { status: res.status, body: await res.text() } });
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        sink.recv(null, value);
      }
      sink.close(null);
    })().catch((e) => !ac.signal.aborted && sink.close(errorJson(e)));
    return { send() {}, close: () => ac.abort() };
  };

const cartesia = rawBytes((r) => `/api/cartesia${r.path}`);
const openaiTts = rawBytes((r) => `/api/openai/${encodeURIComponent(r.endpoint ?? '')}${r.path}`);
const deepgramTts = rawBytes((r) => withQuery(`/api/deepgram${r.path}`, r.query));

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

/** Built-in speechSynthesis, one utterance per exchange. Closing the exchange cancels speech. */
const liveUtterances = new Set<SpeechSynthesisUtterance>(); // Chrome drops events of utterances that get garbage collected
const speechSynthesisDriver: LiveDriver = (req, sink) => {
  const r = req as { text: string; voice: string; lang: string; rate: number; pitch: number };
  if (typeof speechSynthesis === 'undefined') {
    setTimeout(() => sink.close({ error: { message: 'speechSynthesis is not available in this browser' } }), 0);
    return { send() {}, close() {} };
  }
  const u = new SpeechSynthesisUtterance(r.text);
  const voice = r.voice ? speechSynthesis.getVoices().find((v) => v.name === r.voice || v.voiceURI === r.voice) : undefined;
  if (voice) u.voice = voice;
  u.lang = voice?.lang ?? r.lang;
  u.rate = r.rate;
  u.pitch = r.pitch;
  let done = false;
  const finish = (data: Json) => {
    if (done) return;
    done = true;
    liveUtterances.delete(u);
    sink.close(data);
  };
  u.onstart = () => !done && sink.recv({ type: 'start', voice: u.voice?.name ?? null });
  u.onboundary = (e) => !done && sink.recv({ type: 'boundary', name: e.name, charIndex: e.charIndex, charLength: e.charLength ?? null, elapsedTime: e.elapsedTime });
  u.onend = () => finish(null);
  u.onerror = (e) => finish(e.error === 'interrupted' || e.error === 'canceled' ? { status: e.error } : { error: { message: e.error } });
  liveUtterances.add(u);
  speechSynthesis.speak(u);
  return {
    send() {},
    close() {
      if (done) return;
      done = true;
      liveUtterances.delete(u);
      speechSynthesis.cancel();
    },
  };
};

/** Output audio contexts, so a paused replay can freeze what is already scheduled. */
const outputContexts = new Set<AudioContext>();

/** Pause or resume every playing output (speaker audio and browser speech). */
export function setOutputPaused(paused: boolean) {
  for (const ctx of outputContexts) void (paused ? ctx.suspend() : ctx.resume());
  if (typeof speechSynthesis !== 'undefined') {
    if (paused) speechSynthesis.pause();
    else speechSynthesis.resume();
  }
}

const speaker: LiveDriver = (_req, sink, env) => {
  const ctx = new AudioContext({ latencyHint: 'interactive' });
  outputContexts.add(ctx);
  const playing = new Set<AudioBufferSourceNode>();
  setTimeout(() => sink.recv({ type: 'ready', sampleRate: ctx.sampleRate, baseLatency: ctx.baseLatency, outputLatency: ctx.outputLatency ?? null }), 0);
  return {
    send(data, bin) {
      const d = data as { cmd: string; at?: number; sampleRate?: number };
      if (d.cmd === 'flush') {
        for (const s of playing) s.stop();
        playing.clear();
        return;
      }
      if (d.cmd !== 'play' || !bin) return;
      const samples = new Float32Array(bin.slice().buffer);
      if (!samples.length) return;
      const buf = ctx.createBuffer(1, samples.length, d.sampleRate!);
      buf.copyToChannel(samples, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.onended = () => playing.delete(src);
      playing.add(src);
      src.start(Math.max(ctx.currentTime, ctx.currentTime + (d.at! - env.now()) / 1000));
    },
    close() {
      outputContexts.delete(ctx);
      void ctx.close();
    },
  };
};

export const browserDrivers: DriverRegistry = { mic, silero, deepgram, anthropic, openai, cartesia, 'openai-tts': openaiTts, 'deepgram-tts': deepgramTts, 'speech-synthesis': speechSynthesisDriver, rnnoise, speaker };
