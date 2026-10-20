// FireRedVAD streaming VAD (FireRedTeam/FireRedVAD, Stream-VAD, Apache-2.0): Kaldi fbank features
// and the causal DFSMN model on onnxruntime. No DOM or bundler imports, so Node scripts and tests
// run the same code the browser driver does.
//
//   const vad = new FireRedStream(ort, session);
//   for each chunk of 16 kHz float samples: const probs = await vad.push(chunk)  // one per 10 ms frame
//
// The model sees 25 ms windows every 10 ms (frame k covers samples [160k, 160k + 400) of the stream)
// and keeps 19 frames of FSMN memory per block as its cache, so each frame is scored once, causally.

import type * as Ort from 'onnxruntime-web';
import cmvn from './cmvn.json';

export const SAMPLE_RATE = 16000;
export const WINDOW = 400; // 25 ms
export const HOP = 160; // 10 ms
export const MEL_BINS = 80;
const FFT = 512;
const CACHE_SHAPE = [8, 1, 128, 19]; // [FSMN blocks, batch, projection, lookback - 1]

/**
 * Kaldi-compatible log mel filterbank (kaldi-native-fbank defaults as FireRedVAD sets them): 25 ms
 * Povey window every 10 ms, snip_edges, DC removal, pre-emphasis 0.97, 512-point power spectrum,
 * 80 mel bins from 20 Hz to Nyquist, no dither, log floored at float epsilon. Input is float audio
 * in [-1, 1]; it is scaled to the 16-bit range the model was trained on.
 */
export class KaldiFbank {
  private buf = new Float64Array(0);
  private readonly window = new Float64Array(WINDOW);
  private readonly banks: { first: number; w: Float64Array }[] = [];
  private readonly re = new Float64Array(FFT);
  private readonly im = new Float64Array(FFT);
  private readonly cos = new Float64Array(FFT / 2);
  private readonly sin = new Float64Array(FFT / 2);
  private readonly rev = new Uint16Array(FFT);

  constructor() {
    for (let i = 0; i < WINDOW; i++) this.window[i] = Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WINDOW - 1)), 0.85);
    const mel = (f: number) => 1127 * Math.log(1 + f / 700);
    const lo = mel(20);
    const hi = mel(SAMPLE_RATE / 2);
    const delta = (hi - lo) / (MEL_BINS + 1);
    const binHz = SAMPLE_RATE / FFT;
    for (let m = 0; m < MEL_BINS; m++) {
      const left = lo + m * delta;
      const center = left + delta;
      const right = center + delta;
      const w: number[] = [];
      let first = -1;
      for (let i = 0; i < FFT / 2; i++) {
        const x = mel(binHz * i);
        if (x > left && x < right) {
          if (first < 0) first = i;
          w.push(x <= center ? (x - left) / (center - left) : (right - x) / (right - center));
        } else if (first >= 0) break;
      }
      this.banks.push({ first, w: Float64Array.from(w) });
    }
    for (let i = 0; i < FFT / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / FFT);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / FFT);
    }
    const bits = Math.log2(FFT);
    for (let i = 0; i < FFT; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  reset(): void {
    this.buf = new Float64Array(0);
  }

  /** Appends samples and returns the log mel frames that are now complete, each MEL_BINS long. */
  push(samples: Float32Array): Float32Array[] {
    const buf = new Float64Array(this.buf.length + samples.length);
    buf.set(this.buf);
    for (let i = 0; i < samples.length; i++) buf[this.buf.length + i] = samples[i] * 32768;
    const out: Float32Array[] = [];
    let off = 0;
    for (; off + WINDOW <= buf.length; off += HOP) out.push(this.frame(buf.subarray(off, off + WINDOW)));
    this.buf = buf.slice(off);
    return out;
  }

  private frame(x: Float64Array): Float32Array {
    const { re, im } = this;
    let mean = 0;
    for (let i = 0; i < WINDOW; i++) mean += x[i];
    mean /= WINDOW;
    const f = new Float64Array(WINDOW);
    for (let i = 0; i < WINDOW; i++) f[i] = x[i] - mean;
    for (let i = WINDOW - 1; i > 0; i--) f[i] -= 0.97 * f[i - 1];
    f[0] -= 0.97 * f[0];
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < WINDOW; i++) re[this.rev[i]] = f[i] * this.window[i];
    for (let size = 2; size <= FFT; size *= 2) {
      const half = size / 2;
      const step = FFT / size;
      for (let s = 0; s < FFT; s += size) {
        for (let j = 0; j < half; j++) {
          const c = this.cos[j * step];
          const sn = this.sin[j * step];
          const a = s + j;
          const b = a + half;
          const tr = re[b] * c - im[b] * sn;
          const ti = re[b] * sn + im[b] * c;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
    const out = new Float32Array(MEL_BINS);
    for (let m = 0; m < MEL_BINS; m++) {
      const { first, w } = this.banks[m];
      let e = 0;
      for (let k = 0; k < w.length; k++) e += w[k] * (re[first + k] ** 2 + im[first + k] ** 2);
      out[m] = Math.log(Math.max(e, 1.1920928955078125e-7));
    }
    return out;
  }
}

/** Global CMVN with the model's statistics, in place. */
export function applyCmvn(feat: Float32Array): Float32Array {
  for (let i = 0; i < MEL_BINS; i++) feat[i] = (feat[i] - cmvn.means[i]) * cmvn.istd[i];
  return feat;
}

type OrtModule = typeof Ort;

/** Streaming FireRedVAD on an onnxruntime session of stream_vad.onnx: audio in, a speech probability per 10 ms frame out. */
export class FireRedStream {
  private fbank = new KaldiFbank();
  private cache: Ort.Tensor;

  constructor(
    private ort: OrtModule,
    private session: Ort.InferenceSession,
  ) {
    this.cache = this.zeroCache();
  }

  private zeroCache(): Ort.Tensor {
    return new this.ort.Tensor('float32', new Float32Array(CACHE_SHAPE.reduce((a, b) => a * b)), CACHE_SHAPE);
  }

  reset(): void {
    this.fbank.reset();
    this.cache = this.zeroCache();
  }

  async push(samples: Float32Array): Promise<number[]> {
    const frames = this.fbank.push(samples);
    if (!frames.length) return [];
    const feat = new Float32Array(frames.length * MEL_BINS);
    frames.forEach((f, i) => feat.set(applyCmvn(f), i * MEL_BINS));
    const out = await this.session.run({ feat: new this.ort.Tensor('float32', feat, [1, frames.length, MEL_BINS]), cache: this.cache });
    this.cache = out.cacheN;
    return Array.from(out.probs.data as Float32Array);
  }
}
