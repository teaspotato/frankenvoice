import { describe, expect, it } from 'vitest';
import { addTail } from '../src/nodes/tts/tail';

const rate = 24000;
const tone = (hz: number, s: number) => Float32Array.from({ length: Math.round(rate * s) }, (_, i) => 0.5 * Math.sin((2 * Math.PI * hz * i) / rate));
const rms = (x: Float32Array) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);

describe('canned line tail', () => {
  it('sustains the last sound and fades it to silence', () => {
    const x = tone(150, 0.4);
    const y = addTail(x, rate, 150);
    expect(y.length).toBe(x.length + 0.15 * rate);
    expect(rms(y.subarray(x.length - 480, x.length))).toBeGreaterThan(0.3); // still sounding where it used to stop
    expect(Math.max(...y.subarray(-48).map(Math.abs))).toBeLessThan(0.01);
  });
});
