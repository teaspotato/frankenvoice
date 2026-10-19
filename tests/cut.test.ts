import { describe, expect, it } from 'vitest';
import { heardText, speakerNode } from '../src/nodes/speaker';
import { resolveParams, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

describe('speaker pause and resume', () => {
  function speaker() {
    let now = 0;
    const out: Msg[] = [];
    const sent: Json[] = [];
    const ctx = {
      id: 'speaker',
      params: resolveParams(speakerNode, {}),
      now: () => now,
      emit: (_port: string, msg: Msg) => (out.push(msg), out.length),
      setState() {},
      log() {},
      setTimer: () => ({ clear() {} }),
      io: { open: () => ({ xid: '1', send: (d: Json, o?: { media?: Json }) => void sent.push({ ...(d as object), media: o?.media ?? null }), close() {}, closed: false }) },
    } as unknown as NodeContext;
    const node = speakerNode.create(ctx as never);
    node.start?.();
    let pos = 0;
    /** 100 ms of audio at 24 kHz. */
    const play = (t: number, seg = 0, samples = new Float32Array(2400)) => {
      now = t;
      node.onInput?.('audio', { type: 'audio', track: 'tts.audio', start: pos, end: pos + samples.length, sampleRate: 24000, samples, rid: 'r0', seg } as never);
      pos += samples.length;
    };
    const input = (t: number, msg: Msg) => ((now = t), node.onInput?.('audio', msg));
    const segment = (t: number, seg: number, text: string) => ((now = t), node.onInput?.('audio', { type: 'tts.segment_start', rid: 'r0', seg, text }));
    const control = (t: number, msg: Msg) => ((now = t), node.onInput?.('control', msg));
    const plays = () =>
      sent.filter((d) => (d as { cmd: string }).cmd === 'play').map((d) => {
        const p = d as { at: number; media: { start: number; end: number } };
        return [p.at, p.media.start, p.media.end];
      });
    const of = (type: string) => out.filter((m) => m.type === type);
    return { play, segment, control, plays, sent, of, out, input };
  }

  it('stops at once and plays the rest from where it stopped, later by the pause', () => {
    const s = speaker();
    s.play(0);
    s.play(0);
    s.control(50, { type: 'pause' });
    expect(s.sent.at(-1)).toMatchObject({ cmd: 'flush' });
    expect(s.of('played.flush').at(-1)).toMatchObject({ at: 50 });
    expect(s.of('playback.paused')).toEqual([{ type: 'playback.paused', rid: 'r0', at: 50 }]);
    s.control(300, { type: 'resume' });
    expect(s.of('playback.resumed')).toEqual([{ type: 'playback.resumed', rid: 'r0', at: 300, from: 50, shift: 250 }]);
    expect(s.plays().slice(2)).toEqual([
      [300, 1200, 2400],
      [350, 2400, 4800],
    ]);
  });

  it('holds audio and sentences that arrive while paused, and announces them on resume', () => {
    const s = speaker();
    s.segment(0, 0, 'Tokyo is big.');
    s.play(0);
    s.control(50, { type: 'pause' });
    s.segment(60, 1, 'It has trains.');
    s.play(60, 1);
    expect(s.plays()).toEqual([[0, 0, 2400]]);
    expect(s.of('playback.segment').map((m) => m.at)).toEqual([0]);
    s.control(500, { type: 'resume' });
    expect(s.plays().slice(1)).toEqual([
      [500, 1200, 2400],
      [550, 2400, 4800],
    ]);
    expect(s.of('playback.segment').map((m) => [m.seg, m.at])).toEqual([
      [0, 0],
      [1, 550],
    ]);
  });

  it('reports what was heard up to the pause when the reply is then cancelled', () => {
    const s = speaker();
    s.segment(0, 0, 'Tokyo is big.');
    s.play(0);
    s.segment(0, 1, 'It has trains.');
    s.play(0, 1);
    s.control(150, { type: 'pause' });
    s.control(900, { type: 'cancel' });
    expect(s.of('playback.interrupted')).toEqual([{ type: 'playback.interrupted', rid: 'r0', at: 900, heard: 'Tokyo is big. It has' }]);
    expect(s.plays()).toHaveLength(2);
  });

  it('leaves the pause out when working out how much of a sentence was heard', () => {
    const segs = [{ seg: 0, text: 'one two three four', start: 0, end: 1400, gaps: [{ at: 200, ms: 1000 }] }];
    // 400 ms of audio, paused from 200 (half way) to 1200; three quarters played by 1300.
    expect(heardText(segs, 100)).toBe('one');
    expect(heardText(segs, 1100)).toBe('one two');
    expect(heardText(segs, 1300)).toBe('one two three');
    expect(heardText(segs, 1400)).toBe('one two three four');
  });

  it('keeps all the held audio when more arrives late in a long pause', () => {
    const s = speaker();
    for (let i = 0; i < 10; i++) s.play(10);
    s.control(150, { type: 'pause' });
    s.play(600);
    s.control(800, { type: 'resume' });
    const resent = s.plays().slice(10);
    expect(resent[0]).toEqual([800, 3360, 4800]);
    expect(resent).toHaveLength(10);
    expect(resent.at(-1)).toEqual([1660, 24000, 26400]);
  });

  it('reports the interruption before cutting the echo reference, as before pausing existed', () => {
    const s = speaker();
    s.play(0);
    s.control(50, { type: 'cancel' });
    const types = s.out.map((m) => m.type).filter((t) => t === 'playback.interrupted' || t === 'played.flush');
    expect(types).toEqual(['playback.interrupted', 'played.flush']);
  });

  /** "Words": 300 ms of tone, 40 ms of silence, 300 ms of tone. */
  const speech = () => {
    const x = new Float32Array(Math.round(0.64 * 24000));
    for (let i = 0; i < x.length; i++) if (i < 7200 || i >= 8160) x[i] = 0.3 * Math.sin((2 * Math.PI * 150 * i) / 24000);
    return x;
  };

  it('finishes the sound playing and fades out, instead of stopping mid-sound', () => {
    const s = speaker();
    s.play(0, 0, speech());
    s.control(100, { type: 'pause' });
    const flush = s.sent.at(-1) as { cmd: string; at: number; releaseMs: number };
    expect(flush.cmd).toBe('flush');
    expect(flush.at).toBeGreaterThanOrEqual(295);
    expect(flush.at).toBeLessThanOrEqual(310);
    expect(flush.releaseMs).toBe(40);
    const paused = s.of('playback.paused')[0] as unknown as { at: number; from?: number };
    expect(paused.at).toBe(flush.at + 40);
    // No word timings: it carries on from where the sound stopped.
    expect(paused.from).toBe(flush.at);
  });

  it('resumes from the start of the word it stopped in, fading in', () => {
    const s = speaker();
    s.play(0, 0, speech());
    s.input(0, { type: 'tts.words', rid: 'r0', words: [{ word: 'one', start: 0, end: 7200 }, { word: 'two', start: 8160, end: 15360 }] });
    s.control(500, { type: 'pause' }); // in the middle of "two" (340 to 640 ms)
    const paused = s.of('playback.paused')[0] as unknown as { at: number; from: number };
    expect(paused.at).toBe(640 + 40); // its sound ends with the audio
    expect(paused.from).toBeGreaterThanOrEqual(300);
    expect(paused.from).toBeLessThanOrEqual(345);
    s.control(1500, { type: 'resume' });
    const [at, start] = s.plays().at(-1)!;
    expect(at).toBe(1500);
    expect(start).toBe(Math.round((paused.from / 1000) * 24000));
    const replay = (s.sent.at(-1) as { cmd: string }).cmd;
    expect(replay).toBe('play');
  });

  it('lets a cancelled reply finish its sound too, and starts the next one after it', () => {
    const s = speaker();
    s.play(0, 0, speech());
    s.control(100, { type: 'cancel' });
    expect(s.sent.at(-1)).toMatchObject({ cmd: 'flush', releaseMs: 40 });
    const at = (s.of('playback.interrupted')[0] as unknown as { at: number }).at;
    expect(at).toBeGreaterThan(330);
    expect(s.of('played.flush').at(-1)).toMatchObject({ at });
  });
});
