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

const cartesia: LiveDriver = (req, sink) => {
  const r = req as { path: string; body: Json };
  const ac = new AbortController();
  (async () => {
    const res = await fetch(`/api/cartesia${r.path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(r.body),
      signal: ac.signal,
    });
    if (!res.ok || !res.body) return sink.close({ error: `HTTP ${res.status}`, body: await res.text() });
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

const speaker: LiveDriver = (_req, sink, env) => {
  const ctx = new AudioContext({ latencyHint: 'interactive' });
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
      void ctx.close();
    },
  };
};

export const browserDrivers: DriverRegistry = { mic, silero, deepgram, anthropic, cartesia, speaker };
