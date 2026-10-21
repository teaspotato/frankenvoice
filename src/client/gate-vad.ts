// FireRedVAD in the page, on onnxruntime-web (wasm). Imported only when the page gate is on.

import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import fireredUrl from '../../server/drivers/firered/stream_vad.onnx?url';
import { FireRedStream } from '../../server/drivers/firered/stream';

/**
 * Built pages take onnxruntime's wasm (14 MB) from jsDelivr, cached worldwide, rather than from
 * wherever the page is hosted (the GPU box sits behind a tunnel). The dev server serves its own.
 */
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
ort.env.wasm.wasmPaths = import.meta.env.PROD ? ORT_CDN : { wasm: ortWasmUrl, mjs: ortMjsUrl };
ort.env.wasm.numThreads = 1;

/** Load the model; each call of the result starts a fresh stream on it. */
export async function loadFireRed(): Promise<() => FireRedStream> {
  const session = await ort.InferenceSession.create(fireredUrl);
  return () => new FireRedStream(ort, session);
}
