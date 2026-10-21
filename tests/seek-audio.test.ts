import { beforeAll, describe, expect, it } from 'vitest';

let audio: typeof import('../src/client/audio');
beforeAll(async () => {
  audio = await import('../src/client/audio');
}, 30000);

/** Records what gets scheduled, on a clock that stands still. */
const started: { at: number; samples: number }[] = [];
class FakeContext {
  currentTime = 10;
  destination = {};
  constructor(public opts: { sampleRate?: number }) {}
  get sampleRate() {
    return this.opts.sampleRate ?? 48000;
  }
  createGain() {
    return { gain: { value: 1, setTargetAtTime() {} }, connect() {} };
  }
  createBuffer(_ch: number, length: number, sampleRate: number) {
    return { length, duration: length / sampleRate, copyToChannel() {} };
  }
  createBufferSource() {
    const src = {
      buffer: null as { length: number } | null,
      onended: null,
      connect() {},
      stop() {},
      start: (at: number) => started.push({ at, samples: src.buffer!.length }),
    };
    return src;
  }
  close() {}
}
(globalThis as { AudioContext?: unknown }).AudioContext = FakeContext;

describe('seeking a realtime replay', () => {
  it('plays the rest of agent audio that was scheduled before the seek point', () => {
    let now = 0;
    const conn = audio.replaySpeaker({}, { recv() {}, close() {} }, { now: () => now });
    const chunk = (n: number) => new Float32Array(n).buffer;
    audio.setOutputSuppressed(true);
    // Sent during the skip: one second of audio from 1000 ms, and another second after it.
    conn.send({ cmd: 'play', at: 1000, sampleRate: 16000 }, new Uint8Array(chunk(16000)));
    conn.send({ cmd: 'play', at: 2000, sampleRate: 16000 }, new Uint8Array(chunk(16000)));
    now = 1500;
    expect(started).toEqual([]);
    audio.setOutputSuppressed(false);
    // Half of the first chunk is already past; the rest plays now, the second right after it.
    expect(started).toEqual([
      { at: 10, samples: 8000 },
      { at: 10.5, samples: 16000 },
    ]);
    conn.close();
  });

  it('drops held audio the replay flushed during the skip', () => {
    started.length = 0;
    const conn = audio.replaySpeaker({}, { recv() {}, close() {} }, { now: () => 1500 });
    audio.setOutputSuppressed(true);
    conn.send({ cmd: 'play', at: 1000, sampleRate: 16000 }, new Uint8Array(new Float32Array(16000).buffer));
    conn.send({ cmd: 'flush' });
    audio.setOutputSuppressed(false);
    expect(started).toEqual([]);
    conn.close();
  });
});
