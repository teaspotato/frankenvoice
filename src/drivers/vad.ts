// VAD model drivers on onnxruntime. No bundler-specific imports: the browser registry passes in
// onnxruntime-web and a session loader, and Node scripts pass the same, so both run this code.

import type * as Ort from 'onnxruntime-web';
import type { LiveDriver } from '../core/world';
import type { Json } from '../core/types';
import { FireRedStream } from './firered/stream';

type OrtModule = typeof Ort;
/** Loads (once) the session for a model name from the request. */
export type SessionLoader = (model: string) => Promise<Ort.InferenceSession>;

const errorJson = (e: unknown): Json => ({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });

/** Silero VAD v5/v6: one 512-sample frame in, one speech probability out. */
export const sileroDriver =
  (ort: OrtModule, load: SessionLoader): LiveDriver =>
  (req, sink) => {
    const r = req as { model: string; sampleRate: number };
    const CONTEXT = 64; // samples of the previous frame the model expects in front of each frame
    let closed = false;
    let context = new Float32Array(CONTEXT);
    let state = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
    const sr = new ort.Tensor('int64', BigInt64Array.from([BigInt(r.sampleRate)]), []);
    const session = load(r.model);
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

/** FireRedVAD (streaming): a chunk of samples in, the speech probability of each 10 ms frame it completed out. */
export const fireredDriver =
  (ort: OrtModule, load: SessionLoader): LiveDriver =>
  (req, sink) => {
    const r = req as { model: string };
    let closed = false;
    const stream = load(r.model).then((s) => new FireRedStream(ort, s));
    let chain: Promise<void> = stream.then(() => undefined);
    chain.catch((e) => !closed && sink.close(errorJson(e)));
    return {
      send(_data, bin) {
        const chunk = new Float32Array(bin!.slice().buffer);
        chain = chain.then(async () => {
          if (closed) return;
          const p = await (await stream).push(chunk);
          if (!closed) sink.recv({ p });
        });
      },
      close() {
        closed = true;
      },
    };
  };
