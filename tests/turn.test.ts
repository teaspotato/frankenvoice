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
    params: resolveParams(duplexTurnNode, { overlap: 'legacy', ...params } as never),
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
    h.send(2420, 'vad', { type: 'vad.speech_end', at: 2420 * h.sr });
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
    h.send(2550, 'vad', { type: 'vad.speech_end', at: 2550 * h.sr });
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

describe('speech over the agent, classified by its words', () => {
  const sr = 16;
  /** The agent is mid-reply: sentences start at 1500, 4000 and 7000 ms. */
  function speaking() {
    const h = harness({ overlap: 'classify' });
    h.user(1000, 'tell me about Tokyo');
    h.send(1500, 'playback', { type: 'playback.start', rid: 'r0' });
    for (const [seg, at] of [1500, 4000, 7000].entries()) h.send(1500, 'playback', { type: 'playback.segment', rid: 'r0', seg, at });
    return h;
  }
  const controls = (h: ReturnType<typeof harness>) => h.out.filter((o) => o.port === 'control').map((o) => o.msg.type);
  const over = (h: ReturnType<typeof harness>, t0: number, t1: number, text: string) => {
    h.send(t0, 'vad', { type: 'vad.speech_start', at: t0 * sr });
    h.send(t1, 'vad', { type: 'vad.speech_end', at: t1 * sr });
    h.send(t1 + 150, 'transcript', { type: 'asr.final', text, start: t0 * sr, end: t1 * sr, speechFinal: true, fromFinalize: true });
  };

  it('cuts the agent off at once, then drops a backchannel and lets it go on', () => {
    const h = speaking();
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * sr });
    expect(controls(h)).toEqual(['pause']);
    h.send(2400, 'vad', { type: 'vad.speech_end', at: 2400 * sr });
    expect(h.decisions()).toEqual(['candidate']); // waiting for the words
    h.send(2550, 'transcript', { type: 'asr.final', text: 'Yeah.', start: 2000 * sr, end: 2400 * sr, speechFinal: true });
    expect(h.decisions()).toEqual(['candidate', 'backchannel']);
    expect(controls(h)).toEqual(['pause', 'resume']);
    expect(h.states().at(-1)).toBe('speaking');
    h.advance(9000);
    expect(h.last('turn.user')?.text).toBe('tell me about Tokyo');
  });

  it('calls speech with no words noise', () => {
    const h = speaking();
    over(h, 2000, 2300, '');
    expect(h.decisions()).toEqual(['candidate', 'noise']);
    expect(controls(h)).toEqual(['pause', 'resume']);
  });

  it('stops at once on a short floor-taking opener like "So"', () => {
    const h = speaking();
    over(h, 2000, 2380, 'So');
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
    expect(h.last('turn.decision')?.why).toBe('floor-taking opener');
    expect(controls(h)).toEqual(['pause', 'cancel']);
  });

  it('stops while the user is still talking when the words open with one', () => {
    const h = speaking();
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * sr });
    h.send(2300, 'transcript', { type: 'asr.partial', text: 'wait', start: 2000 * sr, end: 2300 * sr });
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
  });

  it('holds an aside until the sentence ends, then stops there and answers it', () => {
    const h = speaking();
    over(h, 2000, 2700, 'The blue one.');
    expect(h.decisions()).toEqual(['candidate', 'aside']);
    expect(controls(h)).toEqual(['pause', 'resume']);
    h.advance(3990);
    expect(controls(h)).toEqual(['pause', 'resume']); // still mid-sentence
    h.advance(4000);
    expect(controls(h)).toEqual(['pause', 'resume', 'cancel']);
    h.send(4010, 'playback', { type: 'playback.interrupted', heard: 'Tokyo is big.' });
    expect(h.out.filter((o) => o.msg.type === 'turn.heard').at(-1)?.msg).toEqual({ type: 'turn.heard', turnId: 'turn-u0', text: 'Tokyo is big.' });
    expect(h.last('turn.user')).toMatchObject({ text: 'The blue one.', aside: true });
    expect(h.states().slice(-3)).toEqual(['speaking', 'listening', 'thinking']);
  });

  it('answers an aside when the reply ends before the next sentence', () => {
    const h = speaking();
    over(h, 7500, 8000, 'And Kyoto too.');
    expect(h.decisions()).toEqual(['candidate', 'aside']);
    h.send(9000, 'playback', { type: 'playback.end', rid: 'r0' });
    expect(controls(h)).toEqual(['pause', 'resume']);
    expect(h.last('turn.user')).toMatchObject({ text: 'And Kyoto too.', aside: true });
  });

  it('takes the floor when the speech goes on, whatever the words', () => {
    const h = speaking();
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * sr });
    h.send(2500, 'transcript', { type: 'asr.partial', text: 'yeah that is', start: 2000 * sr, end: 2500 * sr });
    h.advance(3150);
    expect(h.decisions()).toEqual(['candidate']);
    h.advance(3200);
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
    expect(h.last('turn.decision')?.why).toBe('sustained speech');
  });

  it('decides on the words heard so far when the ASR final is late', () => {
    const h = speaking();
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * sr });
    h.send(2300, 'transcript', { type: 'asr.partial', text: 'mhm', start: 2000 * sr, end: 2300 * sr });
    h.send(2400, 'vad', { type: 'vad.speech_end', at: 2400 * sr });
    h.advance(3800);
    expect(h.decisions()).toEqual(['candidate']);
    h.advance(3900);
    expect(h.decisions()).toEqual(['candidate', 'backchannel']);
  });

  it('treats a short pause before the words came in as one stretch of speech', () => {
    const h = speaking();
    h.send(2000, 'vad', { type: 'vad.speech_start', at: 2000 * sr });
    h.send(2300, 'vad', { type: 'vad.speech_end', at: 2300 * sr });
    h.send(2500, 'vad', { type: 'vad.speech_start', at: 2500 * sr });
    expect(h.decisions()).toEqual(['candidate']);
    h.advance(3200);
    expect(h.decisions()).toEqual(['candidate', 'interrupt']);
  });

  it('includes a held aside when the user then takes the floor', () => {
    const h = speaking();
    over(h, 2000, 2600, 'The blue one.');
    over(h, 3000, 3300, 'Actually wait');
    expect(h.decisions()).toEqual(['candidate', 'aside', 'candidate', 'interrupt']);
    h.user(4500, 'make it red', true);
    expect(h.last('turn.user')?.text).toBe('The blue one. Actually wait make it red');
  });

  it('keeps the old rules for speech over the agent while it is still thinking', () => {
    const h = harness({ overlap: 'classify' });
    h.user(1000, 'how about');
    h.send(1200, 'vad', { type: 'vad.speech_start', at: 1200 * sr });
    h.send(1500, 'transcript', { type: 'asr.partial', text: 'about New York', start: 1200 * sr, end: 1500 * sr });
    h.advance(1700);
    expect(h.decisions()).toEqual(['candidate', 'continue']);
    expect(controls(h)).toEqual(['cancel']);
  });

  it('holds the aside for the sentence after a pause, announced with its new time', () => {
    const h = speaking();
    over(h, 2000, 2700, 'The blue one.');
    // The speaker paused at 2000 and resumed at 2850: the sentence at 4000 now starts at 4850.
    h.send(2850, 'playback', { type: 'playback.paused', rid: 'r0', at: 2000 });
    h.send(2850, 'playback', { type: 'playback.segment', rid: 'r0', seg: 1, at: 4850 });
    h.advance(4500);
    expect(controls(h)).toEqual(['pause', 'resume']);
    h.advance(4850);
    expect(controls(h)).toEqual(['pause', 'resume', 'cancel']);
  });

  it('ducks instead with onset: duck, and talks on with onset: none', () => {
    const d = harness({ overlap: 'classify', onset: 'duck' });
    const n = harness({ overlap: 'classify', onset: 'none' });
    for (const h of [d, n]) {
      h.user(1000, 'tell me about Tokyo');
      h.send(1500, 'playback', { type: 'playback.start', rid: 'r0' });
      over(h, 2000, 2400, 'Yeah.');
    }
    expect(controls(d)).toEqual(['duck', 'unduck']);
    expect(controls(n)).toEqual([]);
    expect(n.decisions()).toEqual(['candidate', 'backchannel']);
  });
});
