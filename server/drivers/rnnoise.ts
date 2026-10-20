import { Rnnoise } from '@shiguredo/rnnoise-wasm';
import type { LiveDriver } from '../../src/core/world';
import type { Json } from '../../src/core/types';

const errorJson = (e: unknown): Json => ({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });

let module: Promise<Rnnoise> | undefined;
/** The wasm build refuses to load outside a browser or worker; it runs fine in Node once it believes it is in a worker. */
function load(): Promise<Rnnoise> {
  module ??= (async () => {
    const g = globalThis as { WorkerGlobalScope?: unknown };
    g.WorkerGlobalScope = class {};
    try {
      return await Rnnoise.load();
    } finally {
      delete g.WorkerGlobalScope;
    }
  })();
  module.catch(() => (module = undefined));
  return module;
}

/**
 * RNNoise at 48 kHz: frames are upsampled, denoised in 480-sample blocks and
 * downsampled back. Each frame sent gets exactly one frame back of the same
 * length (the first ones lead with silence while the pipeline fills).
 */
export const rnnoiseDriver: LiveDriver = (req, sink) => {
  const { sampleRate } = req as { sampleRate: number };
  const ratio = 48000 / sampleRate;
  let closed = false;
  if (!Number.isInteger(ratio)) {
    setTimeout(() => sink.close({ error: { message: `RNNoise needs a rate that divides 48 kHz, got ${sampleRate}` } }), 0);
    return { send() {}, close() {} };
  }
  const ready = load().then((m) => m.createDenoiseState());
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
