import { describe, expect, it } from 'vitest';
import { duplexTurnNode } from '../src/nodes/turn';
import { heardText } from '../src/nodes/speaker';
import { resolveParams, type NodeContext } from '../src/core/node';
import type { Msg } from '../src/core/types';

/** Drives one node by hand: a manual clock, timers, and a list of what it emitted. */
function harness(params: Record<string, unknown> = {}) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ctx: NodeContext = {
    id: 'turn',
    params: resolveParams(duplexTurnNode, params as never),
    now: () => now,
    cause: undefined,
    emit: (port, msg) => (out.push({ port, msg }), out.length),
    setState() {},
    log() {},
    setTimer(ms, fn) {
      const t = { at: now + ms, fn, live: true };
      timers.push(t);
      return { clear: () => void (t.live = false) };
    },
    io: {
      open: () => {
        throw new Error('no io');
      },
    },
    media: {
      append: () => {
        throw new Error('no media');
      },
    },
  };
  const node = duplexTurnNode.create(ctx as never);
  node.start?.();
  const advance = (to: number) => {
    for (;;) {
      const t = timers.filter((x) => x.live && x.at <= to).sort((a, b) => a.at - b.at)[0];
      if (!t) break;
      t.live = false;
      now = t.at;
      t.fn();
    }
    now = to;
  };
  const send = (t: number, port: string, msg: Msg) => (advance(t), node.onInput?.(port, msg));
  const sr = 16; // samples per ms at 16 kHz
  const user = (t: number, text: string, speechFinal = true) =>
    send(t, 'transcript', { type: 'asr.final', text, start: (t - 500) * sr, end: t * sr, speechFinal });
  const decisions = () => out.filter((o) => o.msg.type === 'turn.decision').map((o) => o.msg.decision);
  return { send, user, advance, out, decisions, sr };
}

describe('full-duplex turn-taking', () => {
  it('lets a short backchannel through and keeps the agent talking', () => {
    const h = harness();
    h.user(1000, 'tell me about Tokyo');
    h.send(1500, 'playback', { type: 'playback.start' });
    h.send(3000, 'vad', { type: 'vad.speech_start', at: 3000 * h.sr });
    h.send(3300, 'transcript', { type: 'asr.partial', text: 'yeah', start: 3000 * h.sr, end: 3300 * h.sr });
    h.send(3500, 'vad', { type: 'vad.speech_end', at: 3500 * h.sr });
    h.advance(5000);
    expect(h.decisions()).toEqual(['candidate', 'backchannel']);
    expect(h.out.some((o) => o.port === 'control')).toBe(false);
  });

  it('stops the agent when the user says real words over it, and reports what was heard', () => {
    const h = harness();
    h.user(1000, 'tell me about Tokyo');
    h.send(1500, 'playback', { type: 'playback.start' });
    h.send(3000, 'vad', { type: 'vad.speech_start', at: 3000 * h.sr });
    h.send(3200, 'transcript', { type: 'asr.partial', text: 'how about', start: 3000 * h.sr, end: 3200 * h.sr });
    h.advance(3400);
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
    expect(h.out.find((o) => o.port === 'control')?.msg).toEqual({ type: 'cancel' });
    h.send(3420, 'playback', { type: 'playback.interrupted', heard: 'Tokyo is' });
    expect(h.out.at(-1)?.msg).toEqual({ type: 'turn.heard', turnId: 'turn-u0', text: 'Tokyo is' });
    // Their sentence then commits as a fresh turn.
    h.user(4500, 'how about Osaka');
    expect(h.out.at(-1)?.msg).toMatchObject({ type: 'turn.user', text: 'how about Osaka' });
    expect(h.out.at(-1)?.msg.replaces).toBeUndefined();
  });

  it('interrupts on sustained speech even without words, but not on a long backchannel below the limit', () => {
    const h = harness({ minSpeechMs: 400, maxBackchannelMs: 1200 });
    h.user(1000, 'hi');
    h.send(1500, 'playback', { type: 'playback.start' });
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * h.sr });
    h.send(2100, 'transcript', { type: 'asr.partial', text: 'uh huh', start: 2000 * h.sr, end: 2100 * h.sr });
    h.advance(3000);
    expect(h.decisions()).toEqual(['candidate']);
    h.advance(3300);
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
  });

  it('merges speech that arrives while the agent is still thinking into the previous turn', () => {
    const h = harness();
    h.user(1000, 'how about');
    h.send(1200, 'vad', { type: 'vad.speech_start', at: 1200 * h.sr });
    h.send(1500, 'transcript', { type: 'asr.partial', text: 'about New York', start: 1200 * h.sr, end: 1500 * h.sr });
    h.advance(1700);
    expect(h.decisions()).toEqual(['candidate', 'continue']);
    h.user(2500, 'about New York?');
    expect(h.out.at(-1)?.msg).toMatchObject({ type: 'turn.user', text: 'how about about New York?', replaces: 'turn-u0' });
  });

  it('ignores speech over the agent when barge-in is off', () => {
    const h = harness({ bargeIn: false });
    h.user(1000, 'hi');
    h.send(1500, 'playback', { type: 'playback.start' });
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * h.sr });
    h.user(3000, 'stop please');
    h.advance(5000);
    expect(h.decisions()).toEqual([]);
    expect(h.out.filter((o) => o.msg.type === 'turn.user')).toHaveLength(1);
  });
});

describe('heardText', () => {
  const segs = [
    { seg: 0, text: 'Tokyo is huge.', start: 0, end: 1000 },
    { seg: 1, text: 'It has fourteen million people.', start: 1000, end: 3000 },
  ];
  it('keeps finished sentences and the words of the current one played so far', () => {
    expect(heardText(segs, 300)).toBe('Tokyo');
    expect(heardText(segs, 500)).toBe('Tokyo is');
    expect(heardText(segs, 1000)).toBe('Tokyo is huge.');
    expect(heardText(segs, 2000)).toBe('Tokyo is huge. It has fourteen');
    expect(heardText(segs, 5000)).toBe('Tokyo is huge. It has fourteen million people.');
  });
  it('is empty before any audio played', () => {
    expect(heardText(segs, 0)).toBe('');
  });
});
