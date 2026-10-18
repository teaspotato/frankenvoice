// AudioWorklet: downmix to mono, resample to the target rate, and post
// fixed-size float32 frames with the audio-clock time of each frame's end.
class FrankenvoiceCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.ratio = sampleRate / o.targetRate;
    this.integer = Math.abs(this.ratio - Math.round(this.ratio)) < 1e-9 ? Math.round(this.ratio) : 0;
    this.frame = new Float32Array(o.frameSamples);
    this.fill = 0;
    this.n = 0;
    // integer-ratio decimation (box filter) state
    this.acc = 0;
    this.accN = 0;
    // fractional-ratio linear interpolation state
    this.pos = 0;
    this.prev = 0;
  }

  out(s) {
    this.frame[this.fill++] = s;
    if (this.fill === this.frame.length) {
      const samples = this.frame;
      this.frame = new Float32Array(samples.length);
      this.fill = 0;
      this.port.postMessage({ samples, frame: this.n++, ctxTime: currentTime }, [samples.buffer]);
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length) return true;
    const len = input[0].length;
    const mono = new Float32Array(len);
    for (const ch of input) for (let i = 0; i < len; i++) mono[i] += ch[i] / input.length;

    if (this.integer) {
      for (let i = 0; i < len; i++) {
        this.acc += mono[i];
        if (++this.accN === this.integer) {
          this.out(this.acc / this.integer);
          this.acc = 0;
          this.accN = 0;
        }
      }
    } else {
      while (this.pos <= len - 1) {
        const i = Math.floor(this.pos);
        const f = this.pos - i;
        const a = i < 0 ? this.prev : mono[i];
        const b = mono[i + 1] ?? a;
        this.out(a + (b - a) * f);
        this.pos += this.ratio;
      }
      this.pos -= len;
      this.prev = mono[len - 1];
    }
    return true;
  }
}

registerProcessor('frankenvoice-capture', FrankenvoiceCapture);
