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
  /** The last message of a type. */
  const last = (type: string) => out.filter((o) => o.msg.type === type).at(-1)?.msg;
  const states = () => out.filter((o) => o.msg.type === 'turn.transition').map((o) => o.msg.to);
  return { send, user, advance, out, decisions, last, states, sr };
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
    expect(h.last('turn.user')).toMatchObject({ type: 'turn.user', text: 'how about Osaka' });
    expect(h.last('turn.user')?.replaces).toBeUndefined();
    expect(h.states()).toEqual(['thinking', 'speaking', 'speaking+user', 'listening', 'thinking']);
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
    expect(h.last('turn.user')).toMatchObject({ type: 'turn.user', text: 'how about about New York?', replaces: 'turn-u0' });
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

describe('full-duplex turn-taking with the turn model', () => {
  const ready = { type: 'turnmodel.status', status: 'ready', ep: 'webgpu' };
  const ev = (kind: 'eot' | 'int', at: number, p = 0.9) => ({ type: `turnmodel.${kind}`, who: 'user', track: 'mic', at, p });

  it('stops the agent as soon as the model says the user takes the floor', () => {
    const h = harness();
    h.send(0, 'scores', ready);
    h.user(1000, 'tell me about Tokyo');
    h.send(1100, 'scores', ev('eot', 1000 * h.sr));
    h.send(1500, 'playback', { type: 'playback.start' });
    h.send(3000, 'vad', { type: 'vad.speech_start', at: 3000 * h.sr });
    h.send(3150, 'transcript', { type: 'asr.partial', text: 'yeah but', start: 3000 * h.sr, end: 3150 * h.sr });
    h.send(3200, 'scores', ev('int', 3100 * h.sr, 0.4));
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
    expect(h.last('turn.decision')?.why).toBe('model: take the floor p=0.4');
    expect(h.out.find((o) => o.port === 'control')?.msg).toEqual({ type: 'cancel' });
  });

  it('treats speech without a floor take as a backchannel, whatever the words', () => {
    const h = harness();
    h.send(0, 'scores', ready);
    h.user(1000, 'tell me about Tokyo');
    h.send(1100, 'scores', ev('eot', 1000 * h.sr));
    h.send(1500, 'playback', { type: 'playback.start' });
    h.send(3000, 'vad', { type: 'vad.speech_start', at: 3000 * h.sr });
    h.send(3500, 'transcript', { type: 'asr.partial', text: 'oh that is interesting', start: 3000 * h.sr, end: 3500 * h.sr });
    h.send(3900, 'vad', { type: 'vad.speech_end', at: 3900 * h.sr });
    expect(h.decisions()).toEqual(['candidate', 'backchannel']);
    expect(h.states()).toEqual(['thinking', 'speaking', 'speaking+user', 'speaking']);
    // ...but speech that goes on past the max is a turn anyway.
    h.send(5000, 'vad', { type: 'vad.speech_start', at: 5000 * h.sr });
    h.advance(6300);
    expect(h.decisions()).toEqual(['candidate', 'backchannel', 'candidate', 'interrupt']);
  });

  it('commits on the model’s end of turn, not on ASR endpointing, waiting for the final words', () => {
    const h = harness();
    h.send(0, 'scores', ready);
    h.user(1000, 'so I was thinking');
    expect(h.last('turn.user')).toBeUndefined();
    h.send(1300, 'transcript', { type: 'asr.partial', text: 'about Osaka', start: 1100 * h.sr, end: 1300 * h.sr });
    h.send(1500, 'scores', ev('eot', 1400 * h.sr));
    expect(h.last('turn.user')).toBeUndefined();
    h.user(1700, 'about Osaka');
    expect(h.last('turn.user')).toMatchObject({ text: 'so I was thinking about Osaka' });
    expect(h.last('turn.transition')).toMatchObject({ to: 'thinking', why: 'model: end of turn p=0.9' });
  });

  it('commits with the partial when the ASR final is late', () => {
    const h = harness({ asrGraceMs: 300 });
    h.send(0, 'scores', ready);
    h.send(1300, 'transcript', { type: 'asr.partial', text: 'hello there', start: 1000 * h.sr, end: 1300 * h.sr });
    h.send(1500, 'scores', ev('eot', 1400 * h.sr));
    h.advance(1900);
    expect(h.last('turn.user')).toMatchObject({ text: 'hello there' });
  });

  it('judges speech while the agent is still thinking by the rules: the user adding to their turn', () => {
    const h = harness();
    h.send(0, 'scores', ready);
    h.user(1000, 'book a table');
    h.send(1100, 'scores', ev('eot', 1000 * h.sr));
    h.send(1300, 'vad', { type: 'vad.speech_start', at: 1300 * h.sr });
    h.send(1700, 'transcript', { type: 'asr.partial', text: 'for two at seven', start: 1300 * h.sr, end: 1700 * h.sr });
    h.advance(1800);
    expect(h.decisions()).toEqual(['candidate', 'continue']);
    h.user(2400, 'for two at seven');
    h.send(2500, 'scores', ev('eot', 2450 * h.sr));
    expect(h.last('turn.user')).toMatchObject({ text: 'book a table for two at seven', replaces: 'turn-u0' });
  });

  it('drops a pending end of turn when the user starts talking again', () => {
    const h = harness({ asrGraceMs: 300 });
    h.send(0, 'scores', ready);
    h.send(1300, 'transcript', { type: 'asr.partial', text: 'so I was', start: 1000 * h.sr, end: 1300 * h.sr });
    h.send(1400, 'scores', ev('eot', 1350 * h.sr));
    h.send(1500, 'vad', { type: 'vad.speech_start', at: 1500 * h.sr });
    h.advance(2000);
    expect(h.last('turn.user')).toBeUndefined();
    h.user(2500, 'so I was thinking about Osaka', false);
    h.send(2600, 'scores', ev('eot', 2550 * h.sr));
    expect(h.last('turn.user')).toMatchObject({ text: 'so I was thinking about Osaka' });
  });

  it('falls back to the rules when the model fails', () => {
    const h = harness();
    h.send(0, 'scores', ready);
    h.send(500, 'scores', { type: 'turnmodel.status', status: 'failed' });
    h.user(1000, 'hi there');
    expect(h.last('turn.user')).toMatchObject({ text: 'hi there' });
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
