import { describe, expect, it } from 'vitest';
import { backchannelNode, bcState, parseBackchannelLines, takes } from '../src/nodes/backchannel';
import { speakerNode } from '../src/nodes/speaker';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';
import { BcHead, type BcWeights } from '../src/nodes/bc-head';
import bcWeights from '../src/nodes/bc-head.json';
import fixture from './bc-head.fixture.json';

/** Drives one node by hand: a manual clock, timers, io, and what it emitted. */
function harness(def: NodeDef<never>, params: Record<string, unknown> = {}) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ios: { channel: string; request: Json; h: IOHandlers; sent: Json[] }[] = [];
  const ctx: NodeContext = {
    id: def.type,
    params: resolveParams(def, params as never),
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
      open(channel, request, h) {
        const x = { channel, request, h, sent: [] as Json[] };
        ios.push(x);
        return { xid: String(ios.length), send: (d: Json) => void x.sent.push(d), close() {}, closed: false };
      },
    },
    media: { append: (_t: string, s: Float32Array) => ({ track: 'tts.audio', start: 0, end: s.length }) },
  };
  const node = def.create(ctx as never);
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
  const of = (type: string) => out.filter((o) => o.msg.type === type).map((o) => o.msg);
  return { send, advance, out, of, ios };
}

const bc = (params: Record<string, unknown> = {}) => harness(backchannelNode as NodeDef<never>, { lines: 'Mm-hmm.\nYeah.', source: 'turn model', ...params });
const talk = (h: ReturnType<typeof bc>, from: number, to: number) => {
  h.send(from, 'vad', { type: 'vad.speech_start' });
  h.send(to, 'vad', { type: 'vad.speech_end' });
};
const fire = (h: ReturnType<typeof bc>, t: number) => h.send(t, 'scores', { type: 'turnmodel.bc', who: 'agent', p: 0.7 });

describe('backchannels', () => {
  it('asks TTS to render several takes of each line at start', () => {
    const h = bc({ takes: 3 });
    expect(h.of('bc.lines')[0].lines).toEqual(['Mm-hmm.', 'Mm-hmm,', 'Mm-hmm...', 'Yeah.', 'Yeah,', 'Yeah...']);
    expect(takes('Mm-hmm.', 6)).toEqual(['Mm-hmm.', 'Mm-hmm,', 'Mm-hmm...', 'mm-hmm.', 'mm-hmm,', 'mm-hmm...']);
  });

  it('never plays the same take of a line twice in a row', () => {
    const h = bc({ lines: 'Mm-hmm.', minTurnMs: 0, minGapMs: 0 });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    for (let t = 100; t < 2000; t += 100) fire(h, t);
    const got = h.of('bc.say').map((m) => m.take);
    expect(got.length).toBeGreaterThan(10);
    got.slice(1).forEach((t, i) => expect(t).not.toBe(got[i]));
    expect(new Set(got).size).toBeGreaterThan(3);
  });

  it('says a continuer when the model fires mid-turn, then rotates and keeps a gap', () => {
    const h = bc({ minTurnMs: 2000, minGapMs: 4000 });
    talk(h, 0, 2500);
    fire(h, 2600);
    h.send(2700, 'vad', { type: 'vad.speech_start' });
    fire(h, 5000); // too soon after the first
    fire(h, 7000);
    expect(h.of('bc.say').map((m) => m.line)).toEqual(['Mm-hmm.', 'Yeah.']);
    expect(h.of('bc.skip').map((m) => m.reason)).toEqual(['last backchannel 2400 ms ago']);
  });

  it("stays quiet early in a turn, while the agent has the floor, or when the user seems done", () => {
    const h = bc({ minTurnMs: 2000 });
    talk(h, 0, 1000);
    fire(h, 1100); // only 1 s of speech
    h.send(1200, 'turn', { type: 'turn.transition', from: 'listening', to: 'thinking' });
    fire(h, 1300);
    h.send(1400, 'turn', { type: 'turn.transition', from: 'speaking', to: 'listening' });
    h.send(1500, 'vad', { type: 'vad.speech_start' });
    h.send(5050, 'scores', { type: 'turnmodel.frame', eot: [0.8, 0] });
    fire(h, 5100);
    h.send(5150, 'scores', { type: 'turnmodel.frame', eot: [0.1, 0] });
    h.send(5200, 'turn', { type: 'turn.draft', draftId: 'd0' });
    fire(h, 5300);
    expect(h.of('bc.say')).toEqual([]);
    expect(h.of('bc.skip').map((m) => String(m.reason).split(' ')[0])).toEqual(['user', 'agent', 'end-of-turn', 'a']);
  });

  it('can fire on pauses instead of the model', () => {
    const h = bc({ source: 'pause', pauseMs: 100 });
    talk(h, 0, 3000);
    fire(h, 3010); // ignored: source is pause
    h.advance(3100);
    expect(h.of('bc.say').map((m) => m.why)).toEqual(['pause']);
    talk(h, 3200, 9000);
    h.send(9050, 'vad', { type: 'vad.speech_start' }); // speech resumed before the pause was long enough
    h.advance(9500);
    expect(h.of('bc.say')).toHaveLength(1);
  });

  it('picks a line by meaning from the recent words', () => {
    const h = bc({ lines: 'Mm-hmm.\nOh no. = my laptop died', minScore: 0.5, minTurnMs: 2000 });
    const enc = () => h.ios.filter((x) => x.channel === 'embed');
    enc()[0].h.onRecv?.({ vectors: [[1, 0]] }, undefined);
    talk(h, 0, 2500);
    h.send(2400, 'transcript', { type: 'asr.final', text: 'and then the laptop just died' });
    fire(h, 2600);
    expect(enc()[1].request).toMatchObject({ texts: ['and then the laptop just died'] });
    enc()[1].h.onRecv?.({ vectors: [[0.9, 0.1]] }, undefined);
    expect(h.of('bc.say').map((m) => m.line)).toEqual(['Oh no.']);
  });

  it('parses bare lines as continuers and the rest as examples', () => {
    const l = parseBackchannelLines('# c\nMm-hmm.\nOh wow. = I won | it was huge\n');
    expect(l.continuers).toEqual(['Mm-hmm.']);
    expect(l.examples.map((e) => e.text)).toEqual(['I won', 'it was huge']);
  });
});

describe('backchannel turns', () => {
  it('counts a turn that started by talking over the agent', () => {
    const h = bc({ minTurnMs: 2000 });
    h.send(0, 'turn', { type: 'turn.transition', from: 'listening', to: 'speaking' });
    h.send(1000, 'vad', { type: 'vad.speech_start' });
    h.send(1300, 'turn', { type: 'turn.transition', from: 'speaking+user', to: 'listening' });
    fire(h, 3500);
    expect(h.of('bc.say')).toHaveLength(1);
  });

  it("drops a line whose encoder answer comes after the user's turn ended", () => {
    const h = bc({ lines: 'Mm-hmm.\nOh no. = my laptop died', minTurnMs: 0 });
    const enc = () => h.ios.filter((x) => x.channel === 'embed');
    enc()[0].h.onRecv?.({ vectors: [[1, 0]] }, undefined);
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(500, 'transcript', { type: 'asr.final', text: 'my laptop died' });
    fire(h, 600);
    h.send(700, 'turn', { type: 'turn.transition', from: 'listening', to: 'thinking' });
    h.send(800, 'turn', { type: 'turn.transition', from: 'thinking', to: 'listening' });
    enc()[1].h.onRecv?.({ vectors: [[1, 0]] }, undefined);
    expect(h.of('bc.say')).toEqual([]);
    expect(String(h.of('bc.skip').at(-1)?.reason)).toMatch(/moment passed/);
  });
});

describe('speaker cues', () => {
  it('plays a backchannel at once without starting a reply', () => {
    const h = harness(speakerNode as NodeDef<never>);
    const samples = new Float32Array(2400);
    h.send(100, 'audio', { type: 'audio', track: 'tts.audio', start: 0, end: 2400, sampleRate: 24000, samples, rid: 'bc0', cue: true } as Msg);
    expect(h.of('playback.start')).toEqual([]);
    expect(h.of('playback.cue')[0]).toMatchObject({ rid: 'bc0', at: 100, end: 200 });
    expect(h.ios[0].sent.at(-1)).toMatchObject({ cmd: 'cue', at: 100 });
    expect(h.out.filter((o) => o.port === 'played')).toHaveLength(1);
  });

  it('keeps a cue in the echo reference when the reply is cut off under it', () => {
    const h = harness(speakerNode as NodeDef<never>);
    const rate = 24000;
    h.send(0, 'audio', { type: 'audio', track: 'tts.audio', start: 0, end: rate, sampleRate: rate, samples: new Float32Array(rate), rid: 'r0', seg: 0 } as Msg);
    h.send(100, 'audio', { type: 'audio', track: 'tts.audio', start: rate, end: rate + 9600, sampleRate: rate, samples: new Float32Array(9600), rid: 'bc0', cue: true } as Msg);
    h.send(300, 'control', { type: 'cancel' });
    const played = h.out.filter((o) => o.port === 'played').map((o) => o.msg);
    const flush = played.findIndex((m) => m.type === 'played.flush');
    expect(flush).toBeGreaterThan(0);
    // After the flush, the rest of the cue (from 300 ms; it runs to 500 ms) goes back in.
    expect(played[flush + 1]).toMatchObject({ cue: true, at: 300, start: rate + 4800 });
    expect((played[flush + 1].samples as Float32Array).length).toBe(4800);
  });
});

describe('backchannel gates on the VAD and ASR', () => {
  it("doesn't trust the turn model alone: a long silence or an ASR endpoint blocks it", () => {
    const h = bc({ maxSilenceMs: 700 });
    talk(h, 0, 3000);
    fire(h, 4000); // 1 s of silence, eot low or not
    h.send(4100, 'vad', { type: 'vad.speech_start' });
    h.send(6000, 'vad', { type: 'vad.speech_end' });
    h.send(6100, 'transcript', { type: 'asr.final', text: 'right?', speechFinal: true });
    fire(h, 6200);
    h.send(6300, 'vad', { type: 'vad.speech_start' });
    fire(h, 6400);
    expect(h.of('bc.skip').map((m) => String(m.reason).split(' ')[0])).toEqual(['user', 'ASR']);
    expect(h.of('bc.say')).toHaveLength(1);
  });
});

describe('backchannel head', () => {
  it('matches the trained model (Python forward pass on recorded turn-model outputs)', () => {
    const head = new BcHead(bcWeights as BcWeights);
    const got = fixture.frames.map((f) => head.step(f));
    got.forEach((p, i) => expect(p).toBeCloseTo(fixture.p[i], 4));
    expect(Math.max(...got)).toBeGreaterThan(0.4);
  });
});

describe('backchannel defaults', () => {
  it('fires on pauses by default', () => {
    expect(resolveParams(backchannelNode as NodeDef, {}).source).toBe('pause');
  });
});

describe('backchannels decided by TypeSafe', () => {
  const answer = (choice: string, confidence: number) => ({ answers: { pick: { type: 'choice', choice, confidence, probabilities: { [choice]: confidence } } } });
  const ts = (params: Record<string, unknown> = {}) =>
    harness(backchannelNode as NodeDef<never>, { decide: 'typesafe', source: 'pause', options: 'Mm-hmm. | Yeah. = continuer\nOh no. = bad news\n- = nothing', ...params });

  it('asks with the turn\'s ASR words at a pause and says the picked line, taking the lines of an option in turn', () => {
    const h = ts({ gapMs: 0 });
    expect(h.of('bc.lines')[0].lines).toContain('Oh no.');
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(500, 'transcript', { type: 'asr.final', text: 'So I went to the shop' });
    h.send(700, 'transcript', { type: 'asr.partial', text: 'and it was closed.' });
    h.send(800, 'vad', { type: 'vad.speech_end' });
    h.advance(800);
    expect(h.ios).toHaveLength(1);
    const req = h.ios[0].request as { path: string; body: { state: string; questions: { pick: { criteria: Record<string, string> } } } };
    expect(req.path).toBe('/v1/systemone');
    expect(req.body.state).toBe("[Agent hasn't backchannelled yet this turn]\nSo I went to the shop and it was closed.");
    expect(Object.keys(req.body.questions.pick.criteria)).toEqual(['Mm-hmm.', 'Oh no.', '-']);
    h.advance(1000);
    h.ios[0].h.onRecv?.(answer('Mm-hmm.', 0.8), undefined);
    // Only TypeSafe judges whether the user is done: no minimum turn length.
    h.send(1100, 'vad', { type: 'vad.speech_start' });
    h.send(1500, 'vad', { type: 'vad.speech_end' });
    h.advance(1500);
    // The next question says what was said last, and how long ago.
    expect((h.ios[1].request as { body: { state: string } }).body.state).toBe('[Agent\'s last backchannel: "Mm-hmm.", 1 s ago]\nSo I went to the shop and it was closed.');
    h.ios[1].h.onRecv?.(answer('Mm-hmm.', 0.9), undefined);
    expect(h.of('bc.say').map((m) => m.line)).toEqual(['Mm-hmm.', 'Yeah.']);
    expect(h.of('bc.say')[0]).toMatchObject({ choice: 'Mm-hmm.', confidence: 0.8 });
  });

  it('says nothing on "-", below the min confidence, on a failed request, or once the turn is over', () => {
    const h = ts({ gapMs: 0, minConfidence: 0.5 });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(500, 'transcript', { type: 'asr.final', text: 'What do you think?' });
    for (const [i, t] of [600, 1200, 1800, 2400].entries()) {
      h.send(t - 100, 'vad', { type: 'vad.speech_start' });
      h.send(t, 'vad', { type: 'vad.speech_end' });
    h.advance(t);
      if (i === 3) h.send(t + 50, 'turn', { type: 'turn.transition', from: 'listening', to: 'thinking' });
      const io = h.ios[i].h;
      if (i === 0) io.onRecv?.(answer('-', 0.9), undefined);
      else if (i === 1) io.onRecv?.(answer('Oh no.', 0.3), undefined);
      else if (i === 2) io.onClose?.({ error: { status: 401, body: 'no key' } });
      else io.onRecv?.(answer('Oh no.', 0.9), undefined);
    }
    expect(h.of('bc.say')).toHaveLength(0);
    expect(h.of('bc.skip').map((m) => m.reason)).toEqual(['TypeSafe picked - at 0.9', 'TypeSafe picked Oh no. at 0.3', 'TypeSafe gave no answer', 'moment passed: the turn ended']);
  });

  it('keeps the gap between backchannels it said', () => {
    const h = ts({ gapMs: 4000 });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(500, 'transcript', { type: 'asr.final', text: 'my flight got cancelled' });
    h.send(600, 'vad', { type: 'vad.speech_end' });
    h.advance(600);
    h.ios[0].h.onRecv?.(answer('Oh no.', 0.9), undefined);
    h.send(700, 'vad', { type: 'vad.speech_start' });
    h.send(1600, 'vad', { type: 'vad.speech_end' });
    h.advance(1600);
    expect(h.ios).toHaveLength(1);
    expect(h.of('bc.skip')[0].reason).toMatch(/^last backchannel/);
  });
});

describe('TypeSafe backchannel state', () => {
  it('leaves out the last backchannel when context is off', () => {
    expect(bcState('hi there', { line: 'Yeah.', agoMs: 3400 }, false)).toBe('hi there');
    expect(bcState('hi there', { line: 'Yeah.', agoMs: 3400 }, true)).toBe('[Agent\'s last backchannel: "Yeah.", 3 s ago]\nhi there');
  });
});
