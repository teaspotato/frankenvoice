import { describe, expect, it } from 'vitest';
import { backchannelNode, bcGrammar, cleanBackchannel } from '../src/nodes/backchannel';
import { llmNode } from '../src/nodes/llm';
import { kyutaiTtsNode } from '../src/nodes/tts/stream';
import { speakerNode } from '../src/nodes/speaker';
import { tailAfter } from '../src/nodes/tts/tail';
import type { NodeDef } from '../src/core/node';
import type { Json, JsonObject, Msg } from '../src/core/types';
import { harness } from './node-harness';

/** One SSE payload of a streamed chat completion. */
const chunk = (content: string, finish: string | null = null) => ({ choices: [{ index: 0, delta: { content }, finish_reason: finish }] });
const body = (io: { request: Json }) => (io.request as { body: JsonObject }).body;

describe('generated backchannels: the LLM request', () => {
  const llm = () => harness(llmNode, { model: 'gemma', system: 'Be brief.', utterances: 'plain' }, 'llm');
  /** A finished exchange, so there is history. */
  const exchange = (h: ReturnType<typeof llm>) => {
    h.send(0, 'turn', { type: 'turn.user', text: 'Hi there.', turnId: 't0' });
    h.ios[0].h.onRecv?.(chunk('Hello!', 'stop'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
  };

  it("sends the reply's prefix, then the words and the instruction as the last user turn, with the prefill, on its own slot", () => {
    const h = llm();
    exchange(h);
    h.send(1000, 'bc', {
      type: 'bc.ask',
      id: 'b0',
      text: 'so my laptop died',
      prompt: '(Backchannel.)',
      prefill: '[i0g0] ',
      body: { max_tokens: 9, temperature: 0.3, grammar: 'root ::= "x"' },
    });
    const reply = body(h.ios[0]);
    const ask = body(h.ios[1]);
    const msgs = ask.messages as JsonObject[];
    expect(reply.id_slot).toBe(0);
    expect(ask.id_slot).toBe(1);
    expect(msgs.slice(0, 3)).toEqual([...(reply.messages as JsonObject[]), { role: 'assistant', content: 'Hello!' }]);
    expect(msgs.slice(3)).toEqual([
      { role: 'user', content: '(Backchannel.)\n\nso my laptop died' },
      { role: 'assistant', content: '[i0g0] ' },
    ]);
    expect(ask).toMatchObject({ max_tokens: 9, temperature: 0.3, grammar: 'root ::= "x"', stream: true });
    h.ios[1].h.onRecv?.(chunk('Oh'), undefined);
    h.ios[1].h.onRecv?.(chunk(' no.', 'stop'), undefined);
    h.ios[1].h.onRecv?.({ $done: true }, undefined);
    expect(h.of('llm.bc')).toMatchObject([{ id: 'b0', status: 'done', text: 'Oh no.' }]);
    // Nothing of it is kept: the next reply sends the same history as before, plus the new turn.
    h.send(2000, 'turn', { type: 'turn.user', text: 'Anyway.', turnId: 't1' });
    expect((body(h.ios[2]).messages as JsonObject[]).map((m) => m.content)).toEqual(['Be brief.', 'Hi there.', 'Hello!', 'Anyway.']);
    expect(body(h.ios[2]).id_slot).toBe(0);
    expect(h.of('llm.start')).toHaveLength(2);
  });

  it('cancels the request on bc.drop, and never touches a reply', () => {
    const h = llm();
    h.send(0, 'bc', { type: 'bc.ask', id: 'b0', text: 'hmm', prompt: '', prefill: '' });
    h.send(100, 'bc', { type: 'bc.drop', id: 'b0' });
    expect(h.ios[0].closed).toBe(true);
    h.ios[0].h.onRecv?.(chunk('Yeah.', 'stop'), undefined);
    expect(h.of('llm.bc')).toEqual([]);
    expect(h.of('llm.delta')).toEqual([]);
  });
});

describe('generated backchannels: the stage', () => {
  const stage = (params: Record<string, unknown> = {}) => harness(backchannelNode, { source: 'pause', minTurnMs: 0, speculate: false, ...params }, 'bc');
  const pauseAfter = (h: ReturnType<typeof stage>, words: string, t = 1000) => {
    h.send(t - 900, 'vad', { type: 'vad.speech_start' });
    h.send(t - 100, 'transcript', { type: 'asr.final', text: words });
    h.send(t, 'vad', { type: 'vad.speech_end' });
    h.advance(t);
  };

  it('writes lines ahead as the words come, one at a time, and says a fitting one at once', () => {
    const h = stage({ speculate: true });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(100, 'transcript', { type: 'asr.partial', text: 'so my laptop' });
    h.send(250, 'transcript', { type: 'asr.partial', text: 'so my laptop just' }); // one in flight
    const asks = () => h.of('bc.ask');
    expect(asks().map((m) => [m.id, m.text, m.draft])).toEqual([['bc-d0', 'so my laptop', true]]);
    h.send(400, 'llm', { type: 'llm.bc', id: 'bc-d0', status: 'done', text: 'Oh.' });
    // Done, and the words moved on: written again from them.
    expect(asks().map((m) => m.text)).toEqual(['so my laptop', 'so my laptop just']);
    h.send(500, 'llm', { type: 'llm.bc', id: 'bc-d1', status: 'done', text: 'Oh no' });
    h.send(700, 'transcript', { type: 'asr.partial', text: 'so my laptop just died' });
    expect(asks()).toHaveLength(3);
    h.send(750, 'llm', { type: 'llm.bc', id: 'bc-d2', status: 'done', text: 'Oh no.' });
    h.send(800, 'vad', { type: 'vad.speech_end' });
    h.advance(800);
    // Written from the same words: said at once, no new request.
    expect(asks()).toHaveLength(3);
    expect(h.of('bc.say')).toMatchObject([{ line: 'Oh no.', gen: true, ms: 0, draft: { text: 'so my laptop just died', ms: 50 } }]);
    expect(h.of('bc.prep')).toHaveLength(1);
  });

  it('waits for a line being written from the same words, and asks afresh when the words moved on', () => {
    const h = stage({ minGapMs: 0, speculate: true });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(100, 'transcript', { type: 'asr.partial', text: 'we drove along the coast' });
    h.send(200, 'vad', { type: 'vad.speech_end' });
    h.advance(200);
    expect(h.of('bc.ask')).toHaveLength(1);
    expect(h.of('bc.prep')).toEqual([{ type: 'bc.prep', id: 'bc-d0' }]);
    h.send(260, 'llm', { type: 'llm.bc', id: 'bc-d0', status: 'done', text: 'Nice.' });
    expect(h.of('bc.say')).toMatchObject([{ id: 'bc-d0', line: 'Nice.', ms: 60 }]);
    h.send(1000, 'vad', { type: 'vad.speech_start' });
    h.send(1100, 'transcript', { type: 'asr.final', text: 'we drove along the coast and the view was absolutely incredible' });
    h.send(1150, 'llm', { type: 'llm.bc', id: 'bc-d1', status: 'done', text: 'Wow.' });
    h.send(1200, 'transcript', { type: 'asr.partial', text: 'and then we stopped at a little town for lunch' });
    h.send(1300, 'vad', { type: 'vad.speech_end' });
    h.advance(1300);
    // The draft in flight is from older words: a new request, which drops it.
    const ask = h.of('bc.ask').at(-1)!;
    expect(ask.draft).toBeUndefined();
    expect(ask.text).toMatch(/lunch$/);
  });

  it('asks the LLM at the moment, opens a TTS socket, and says what it wrote', () => {
    const h = stage();
    expect(h.of('bc.lines')[0].lines).toContain('Mm-hmm.');
    pauseAfter(h, 'and then my laptop died');
    const ask = h.of('bc.ask')[0];
    expect(ask).toMatchObject({ id: 'bc-bc0', text: 'and then my laptop died', prefill: '[i0g0] ' });
    expect((ask.body as JsonObject).grammar).toBe(bcGrammar(4));
    expect(h.out.find((o) => o.msg.type === 'bc.ask')?.port).toBe('ask');
    expect(h.of('bc.prep')).toEqual([{ type: 'bc.prep', id: 'bc-bc0' }]);
    h.send(1180, 'llm', { type: 'llm.bc', id: 'bc-bc0', status: 'done', text: 'Oh no' });
    expect(h.of('bc.say')).toMatchObject([{ id: 'bc-bc0', line: 'Oh no.', take: 'Oh no.', gen: true, ms: 180, askedAt: 1000 }]);
    // The wait timer was cleared: no fallback later.
    h.advance(3000);
    expect(h.of('bc.say')).toHaveLength(1);
  });

  it('says a canned line when the LLM is late or writes something else, and drops its request', () => {
    const h = stage({ waitMs: 400, minGapMs: 0 });
    pauseAfter(h, 'so I went to the shop');
    h.advance(1400);
    expect(h.of('bc.drop').map((m) => m.id)).toEqual(['bc-bc0', 'bc-bc0']);
    expect(h.of('bc.say')).toMatchObject([{ line: 'Mm-hmm.', fallback: 'the LLM took over 400 ms' }]);
    h.send(1450, 'llm', { type: 'llm.bc', id: 'bc-bc0', status: 'done', text: 'Oh.' });
    expect(h.of('bc.say')).toHaveLength(1);
    pauseAfter(h, 'and it was closed', 3000);
    h.send(3100, 'llm', { type: 'llm.bc', id: 'bc-bc2', status: 'done', text: 'Why was it closed?' });
    expect(h.of('bc.say').at(-1)).toMatchObject({ line: 'Yeah.', fallback: 'not a backchannel: "Why was it closed?"' });
  });

  it("drops a line when the user's turn ends first: before it is written, or before its audio starts", () => {
    const h = stage({ minGapMs: 0 });
    pauseAfter(h, 'my cat knocked over the plant');
    h.send(1100, 'turn', { type: 'turn.transition', from: 'listening', to: 'thinking' });
    expect(h.of('bc.skip').at(-1)).toMatchObject({ reason: 'moment passed: the turn ended' });
    expect(h.of('bc.drop')).toHaveLength(2);
    h.send(1200, 'llm', { type: 'llm.bc', id: 'bc-bc0', status: 'done', text: 'Oh no.' });
    expect(h.of('bc.say')).toEqual([]);
    h.send(2000, 'turn', { type: 'turn.transition', from: 'speaking', to: 'listening' });
    pauseAfter(h, 'and then the dog ate it', 3000);
    h.send(3100, 'llm', { type: 'llm.bc', id: 'bc-bc1', status: 'done', text: 'Oh no.' });
    h.send(3200, 'turn', { type: 'turn.transition', from: 'listening', to: 'thinking' });
    // TTS drops it only if none of it has played.
    expect(h.of('bc.drop').at(-1)).toMatchObject({ id: 'bc-bc1', unheard: true });
  });

  it('cleans what the LLM wrote', () => {
    expect(cleanBackchannel('Oh, wow', 4)).toBe('Oh, wow.');
    expect(cleanBackchannel('[i0g0] Mm-hmm!', 4)).toBe('Mm-hmm!');
    expect(cleanBackchannel('"Right."', 4)).toBe('Right.');
    expect(cleanBackchannel('Really?', 4)).toBeNull();
    expect(cleanBackchannel('', 4)).toBeNull();
    expect(cleanBackchannel('that is a very long line', 4)).toBeNull();
  });
});

describe('generated backchannels: TTS', () => {
  const tts = () => harness(kyutaiTtsNode, { alignWords: false }, 'tts');
  const pcm = (n: number) => {
    const b = new Uint8Array(n * 2);
    const v = new DataView(b.buffer);
    for (let i = 0; i < n; i++) v.setInt16(i * 2, Math.round(Math.sin(i / 10) * 8000), true);
    return b;
  };

  it("speaks on its own socket, opened ahead, and plays the audio as cues as it comes, with a tail", () => {
    const h = tts();
    h.send(0, 'text', { type: 'bc.prep', id: 'b0' });
    expect(h.ios).toHaveLength(1);
    // A reply starting meanwhile gets its own socket; the backchannel's stays.
    h.send(50, 'text', { type: 'llm.start', rid: 'r0' });
    expect(h.ios).toHaveLength(2);
    h.send(200, 'text', { type: 'bc.say', id: 'b0', line: 'Oh no.', gen: true, tailMs: 100 });
    expect(h.ios[0].sent).toEqual([{ type: 'Speak', text: 'Oh no.' }, { type: 'Close' }]);
    h.ios[0].h.onRecv?.(null, pcm(1920));
    h.ios[0].h.onRecv?.(null, pcm(1920));
    // The reply is cancelled (barge-in): the backchannel goes on.
    h.send(300, 'control', { type: 'cancel' });
    expect(h.ios[0].closed).toBe(false);
    h.send(320, 'text', { type: 'bc.drop', id: 'b0', unheard: true });
    expect(h.ios[0].closed).toBe(false);
    h.ios[0].h.onRecv?.(null, pcm(1920));
    h.ios[0].h.onClose?.(null);
    const audio = h.of('audio');
    expect(audio.every((m) => m.cue && m.rid === 'b0')).toBe(true);
    expect(audio.map((m) => (m.samples as Float32Array).length)).toEqual([1920, 1920, 1920, 2400]);
    expect(h.of('tts.cue_end')).toMatchObject([{ rid: 'b0', text: 'Oh no.' }]);
  });

  it('closes the socket of a line dropped before any audio, or replaced by a canned one', () => {
    const h = tts();
    h.send(0, 'text', { type: 'bc.prep', id: 'b0' });
    h.send(100, 'text', { type: 'bc.drop', id: 'b0', unheard: true, why: 'the turn ended' });
    expect(h.ios[0].closed).toBe(true);
    expect(h.of('tts.cue_dropped')).toMatchObject([{ rid: 'b0', why: 'the turn ended' }]);
    h.send(200, 'text', { type: 'bc.prep', id: 'b1' });
    h.send(300, 'text', { type: 'bc.say', id: 'b1', line: 'Mm-hmm.', take: 'Mm-hmm.' });
    expect(h.ios[1].closed).toBe(true);
  });

  it('a tail continues the sound', () => {
    const rate = 24000;
    const last = new Float32Array(2400).map((_, i) => Math.sin(i / 10));
    const t = tailAfter(last, rate, 100);
    expect(t.length).toBe(2400);
    expect(Math.abs(t[0] - Math.sin(2400 / 10))).toBeLessThan(0.3);
    expect(Math.abs(t[t.length - 1])).toBeLessThan(0.01);
  });
});

describe('generated backchannels: the speaker', () => {
  it('plays the pieces back to back, and nothing the user does (pause, gate, cancel) stops them', () => {
    const h = harness(speakerNode as NodeDef, {}, 'speaker');
    const rate = 24000;
    const piece = (t: number, rid: string, cue: boolean) =>
      h.send(t, 'audio', { type: 'audio', track: 'tts.audio', start: 0, end: 2400, sampleRate: rate, samples: new Float32Array(2400).fill(0.2), rid, cue, seg: 0 } as Msg);
    piece(0, 'r0', false);
    piece(10, 'b0', true);
    piece(30, 'b0', true);
    h.send(40, 'control', { type: 'pause' });
    h.send(50, 'control', { type: 'gate', until: 5000 });
    h.send(60, 'control', { type: 'cancel' });
    piece(70, 'b0', true);
    const cues = h.ios[0].sent.filter((d) => (d as JsonObject).cmd === 'cue') as JsonObject[];
    expect(cues.map((c) => c.at)).toEqual([10, 110, 210]);
    expect(h.of('playback.cue').map((m) => [m.at, m.end])).toEqual([
      [10, 110],
      [110, 210],
      [210, 310],
    ]);
    // Cutting the reply off sends flushes for the reply's output only (the page plays cues on their own).
    expect(h.ios[0].sent.some((d) => (d as JsonObject).cmd === 'flush')).toBe(true);
    expect(h.of('playback.interrupted')).toHaveLength(1);
  });
});
