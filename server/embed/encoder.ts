import type * as Ort from 'onnxruntime-node';
import { meanPool, WordPiece } from './wordpiece';

export interface Encoder {
  /** One unit vector per text. */
  embed(texts: string[]): Promise<number[][]>;
}

/** A sentence encoder (BERT-style ONNX export with mean pooling) on onnxruntime. */
export async function loadEncoder(ort: typeof Ort, model: Uint8Array, vocab: string, options?: Ort.InferenceSession.SessionOptions): Promise<Encoder> {
  const session = await ort.InferenceSession.create(model, options ?? {});
  const tok = new WordPiece(vocab);
  const needsTypes = session.inputNames.includes('token_type_ids');
  return {
    async embed(texts) {
      const out: number[][] = [];
      // One at a time: no padding, and these are short.
      for (const text of texts) {
        const ids = tok.encode(text);
        const n = ids.length;
        const t = (a: number[]) => new ort.Tensor('int64', BigInt64Array.from(a.map(BigInt)), [1, n]);
        const feeds: Record<string, Ort.Tensor> = { input_ids: t(ids), attention_mask: t(ids.map(() => 1)) };
        if (needsTypes) feeds.token_type_ids = t(ids.map(() => 0));
        const res = await session.run(feeds);
        const h = res[session.outputNames[0]];
        out.push(meanPool(h.data as Float32Array, n, h.dims[2]));
      }
      return out;
    },
  };
}
