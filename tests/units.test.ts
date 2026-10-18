import { describe, expect, it } from 'vitest';
import { SentenceSplitter } from '../src/nodes/segment';
import { Framer } from '../src/nodes/audio';
import { SimScheduler } from '../src/core/scheduler';
import type { AudioMsg } from '../src/core/types';

describe('SentenceSplitter', () => {
  it('emits sentences as soon as their boundary is certain', () => {
    const s = new SentenceSplitter();
    expect(s.push('Hello there')).toEqual([]);
    expect(s.push('! How are')).toEqual(['Hello there!']);
    expect(s.push(' you? Pi is 3.')).toEqual(['How are you?']);
    expect(s.push('14.')).toEqual([]);
    expect(s.flush()).toBe('Pi is 3.14.');
  });
});

describe('Framer', () => {
  it('re-chunks audio into fixed frames with exact sample ranges', () => {
    const f = new Framer(4);
    const frames: [number, number][] = [];
    const msg = (start: number, n: number): AudioMsg => ({ type: 'audio', track: 'm', start, end: start + n, sampleRate: 16000, samples: new Float32Array(n) });
    f.push(msg(0, 3), (_x, _t, a, b) => frames.push([a, b]));
    f.push(msg(3, 6), (_x, _t, a, b) => frames.push([a, b]));
    expect(frames).toEqual([
      [0, 4],
      [4, 8],
    ]);
  });
});

describe('SimScheduler', () => {
  it('runs items in time order, breaking ties by order', () => {
    const sim = new SimScheduler();
    const out: string[] = [];
    sim.at(5, () => out.push('c'));
    sim.at(1, () => out.push('b'), 20);
    sim.at(1, () => out.push('a'), 10);
    const cancel = sim.at(3, () => out.push('x'));
    cancel();
    sim.runFast();
    expect(out).toEqual(['a', 'b', 'c']);
    expect(sim.now()).toBe(5);
  });
});
