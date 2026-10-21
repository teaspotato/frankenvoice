import { describe, expect, it } from 'vitest';
import { interjectNode } from '../src/nodes/interject';
import { duplexTurnNode } from '../src/nodes/turn';
import { llmNode } from '../src/nodes/llm';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

/** Drives one node by hand: a manual clock, timers, fake IO, and a list of what it emitted. */
function harness(def: NodeDef, params: Record<string, unknown> = {}) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ios: { channel: string; request: Json; h: IOHandlers; closed: boolean }[] = [];
  const ctx: NodeContext = {
    id: def.type.split('.')[0],
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
        const x = { channel, request, h, closed: false };
        ios.push(x);
        return { xid: String(ios.length), send() {}, close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: {
      append: () => {
        throw new Error('no media');
      },
    },
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

const words = (text: string) => ({ type: 'turn.words', text });
const gateAnswer = (choice: string, confidence = 0.9) => ({ answers: { pick: { type: 'choice', choice, confidence, probabilities: {} } } });

describe('the interjection gate', () => {
  it('asks TypeSafe on new words, and the LLM only when they end on something complete', () => {
    const h = harness(interjectNode);
    h.send(0, 'turn', words('So I think'));
    // Fewer than minWords (4): not asked.
    expect(h.ios).toHaveLength(0);
    h.send(100, 'turn', words('So I think the Eiffel'));
    expect(h.ios).toHaveLength(1);
    expect((h.ios[0].request as { body: { state: string } }).body.state).toBe('So I think the Eiffel');
    // Busy: newer words wait for the answer.
    h.send(200, 'turn', words('So I think the Eiffel Tower is in Berlin.'));
    expect(h.ios).toHaveLength(1);
    h.ios[0].h.onRecv!(gateAnswer('-'), undefined);
    expect(h.of('interject.check')).toHaveLength(0);
    // The answer came: the newer words are asked about at once.
    expect(h.ios).toHaveLength(2);
    h.ios[1].h.onRecv!(gateAnswer('complete'), undefined);
    expect(h.of('interject.gate').map((g) => g.go)).toEqual([false, true]);
    expect(h.of('interject.check')).toMatchObject([{ checkId: 'interject-c0', text: 'So I think the Eiffel Tower is in Berlin.', pass: '[listen]' }]);
    // The same words (modulo case and punctuation) are not asked about again.
    h.send(300, 'turn', words('so I think the Eiffel tower is in Berlin'));
    expect(h.ios).toHaveLength(2);
  });

  it('takes the floor at the next pause when the LLM wants to cut in', () => {
    const h = harness(interjectNode);
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(100, 'turn', words('The Eiffel Tower is in Berlin.'));
    h.ios[0].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(800, 'text', { type: 'llm.check_verdict', checkId: 'interject-c0', verdict: 'interject', text: 'Paris, actually.' });
    // The user is still talking: wait.
    expect(h.of('interject.take')).toHaveLength(0);
    h.send(900, 'turn', words('The Eiffel Tower is in Berlin. And'));
    h.send(1200, 'vad', { type: 'vad.speech_end' });
    expect(h.of('interject.take')).toMatchObject([{ checkId: 'interject-c0', text: 'Paris, actually.', why: 'the user paused' }]);
    expect(h.of('interject.drop')).toHaveLength(0);
  });

  it('drops an interjection the user talked past, and a check the LLM passed on', () => {
    const h = harness(interjectNode, { maxExtraWords: 3 });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(100, 'turn', words('The Eiffel Tower is in Berlin.'));
    h.ios[0].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(800, 'text', { type: 'llm.check_verdict', checkId: 'interject-c0', verdict: 'interject', text: 'Paris, actually.' });
    h.send(900, 'turn', words('The Eiffel Tower is in Berlin. No wait, I mean Paris'));
    expect(h.of('interject.drop')).toMatchObject([{ checkId: 'interject-c0' }]);
    h.send(1200, 'vad', { type: 'vad.speech_end' });
    expect(h.of('interject.take')).toHaveLength(0);
    // A pass ends the check without a take.
    h.ios[1].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(1500, 'text', { type: 'llm.check_verdict', checkId: 'interject-c1', verdict: 'pass', text: '[listen]' });
    h.send(1600, 'vad', { type: 'vad.speech_end' });
    expect(h.of('interject.take')).toHaveLength(0);
  });

  it('drops an interjection the user already talked past when the verdict comes', () => {
    const h = harness(interjectNode, { maxExtraWords: 3 });
    h.send(100, 'turn', words('The Eiffel Tower is in Berlin.'));
    h.ios[0].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(500, 'turn', words('The Eiffel Tower is in Berlin. No wait, I mean Paris.'));
    h.send(800, 'text', { type: 'llm.check_verdict', checkId: 'interject-c0', verdict: 'interject', text: 'Paris, actually.' });
    expect(h.of('interject.take')).toHaveLength(0);
    expect(h.of('interject.drop')).toMatchObject([{ checkId: 'interject-c0' }]);
  });

  it("ignores a TypeSafe answer about an earlier turn's words", () => {
    const h = harness(interjectNode);
    h.send(100, 'turn', words('The Eiffel Tower is in Berlin.'));
    h.send(200, 'turn', words(''));
    h.send(300, 'turn', words('And another thing'));
    h.ios[0].h.onRecv!(gateAnswer('complete'), undefined);
    expect(h.of('interject.check')).toHaveLength(0);
  });

  it('lets a check in flight finish, then checks the newer words', () => {
    const h = harness(interjectNode);
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(100, 'turn', words('The Eiffel Tower is in Berlin.'));
    h.ios[0].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(600, 'turn', words('The Eiffel Tower is in Berlin. It was built in 1999.'));
    h.ios[1].h.onRecv!(gateAnswer('complete'), undefined);
    // Not cancelled for the newer words.
    expect(h.of('interject.drop')).toHaveLength(0);
    expect(h.of('interject.check')).toHaveLength(1);
    h.send(1500, 'text', { type: 'llm.check_verdict', checkId: 'interject-c0', verdict: 'pass', text: '[listen]' });
    expect(h.of('interject.check').map((m) => m.text)).toEqual(['The Eiffel Tower is in Berlin.', 'The Eiffel Tower is in Berlin. It was built in 1999.']);
    // A failed check moves on too.
    h.send(1600, 'turn', words('The Eiffel Tower is in Berlin. It was built in 1999. By Gustave.'));
    h.ios[2].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(2000, 'text', { type: 'llm.check_end', checkId: 'interject-c1', status: 'error' });
    expect(h.of('interject.check')).toHaveLength(3);
  });

  it('waits at most maxWaitMs for a pause', () => {
    const h = harness(interjectNode, { maxWaitMs: 1000 });
    h.send(0, 'vad', { type: 'vad.speech_start' });
    h.send(100, 'turn', words('The Eiffel Tower is in Berlin.'));
    h.ios[0].h.onRecv!(gateAnswer('complete'), undefined);
    h.send(800, 'text', { type: 'llm.check_verdict', checkId: 'interject-c0', verdict: 'interject', text: 'Paris, actually.' });
    h.advance(1900);
    expect(h.of('interject.drop')).toMatchObject([{ why: 'no pause within 1000 ms' }]);
  });
});

describe('the LLM on interjection checks', () => {
  const params = { model: 'm', system: 'sys', utterances: 'plain', liveContext: false };
  const tok = (text: string) => ({ choices: [{ index: 0, delta: { content: text } }] });
  const messages = (r: Json) => (r as { body: { messages: { role: string; content: string }[] } }).body.messages.map((m) => m.content);
  const check = (text: string) => ({ type: 'interject.check', checkId: 'c0', text, prompt: '[cut in? else [listen]]', pass: '[listen]' });

  it('asks with a short token limit', () => {
    const h = harness(llmNode, params);
    h.send(0, 'interject', check('The tower is in Berlin.'));
    expect(h.ios[0].request).toMatchObject({ body: { max_tokens: 300 } });
    h.send(500, 'turn', { type: 'turn.user', turnId: 'u0', text: 'Hi.' });
    expect(h.ios[1].request).toMatchObject({ body: { max_tokens: 4096 } });
  });

  it('passes when the reply is the pass reply', () => {
    const h = harness(llmNode, params);
    h.send(0, 'interject', check('The tower is in Berlin.'));
    expect(messages(h.ios[0].request)).toEqual(['sys', 'The tower is in Berlin.\n\n[cut in? else [listen]]']);
    h.ios[0].h.onRecv!(tok('[lis'), undefined);
    expect(h.of('llm.check_verdict')).toHaveLength(0);
    h.ios[0].h.onRecv!(tok('ten]'), undefined);
    h.ios[0].h.onRecv!({ $done: true }, undefined);
    expect(h.of('llm.check_verdict')).toMatchObject([{ checkId: 'c0', verdict: 'pass' }]);
    expect(h.of('llm.check_end')).toMatchObject([{ checkId: 'c0', status: 'pass' }]);
    expect(h.of('llm.delta')).toHaveLength(0);
  });

  it('says interject at the first words that are not the pass reply, and plays them when the turn keeps the check', () => {
    const h = harness(llmNode, params);
    h.send(0, 'interject', check('The tower is in Berlin.'));
    h.ios[0].h.onRecv!(tok('Paris,'), undefined);
    expect(h.of('llm.check_verdict')).toMatchObject([{ checkId: 'c0', verdict: 'interject', text: 'Paris,' }]);
    expect(h.of('llm.delta')).toHaveLength(0);
    h.send(600, 'turn', { type: 'turn.user', turnId: 'u0', text: 'The tower is in Berlin. And', interject: 'c0' });
    expect(h.ios).toHaveLength(1);
    expect(h.of('llm.start')).toMatchObject([{ rid: 'llm-r0', turnId: 'u0', interject: 'c0', headStartMs: 600 }]);
    expect(h.of('llm.check_end')).toMatchObject([{ status: 'kept' }]);
    h.ios[0].h.onRecv!(tok(' actually.'), undefined);
    h.ios[0].h.onRecv!({ $done: true }, undefined);
    expect(h.of('llm.end')).toMatchObject([{ status: 'done', text: 'Paris, actually.' }]);
    h.send(3000, 'turn', { type: 'turn.user', turnId: 'u1', text: 'Oh right.' });
    // The history has all the committed words, cut off, without the prompt.
    expect(messages(h.ios[1].request)).toEqual(['sys', 'The tower is in Berlin. And—', 'Paris, actually.', 'Oh right.']);
  });

  it('takes a quoted pass reply as a pass, and a failed request as an error, not a pass', () => {
    const h = harness(llmNode, params);
    h.send(0, 'interject', check('The tower is in Berlin.'));
    h.ios[0].h.onRecv!(tok('"[listen]"'), undefined);
    h.ios[0].h.onRecv!({ $done: true }, undefined);
    expect(h.of('llm.check_verdict')).toMatchObject([{ verdict: 'pass' }]);
    h.send(500, 'interject', { ...check('The tower is in Berlin. It is tall.'), checkId: 'c1' });
    h.ios[1].h.onClose!({ error: 'HTTP 500' });
    expect(h.of('llm.check_verdict')).toHaveLength(1);
    expect(h.of('llm.check_end').at(-1)).toMatchObject({ checkId: 'c1', status: 'error' });
  });

  it('marks words committed for an interjection it could not keep as cut off', () => {
    const h = harness(llmNode, params);
    h.send(0, 'interject', check('The tower is in Berlin.'));
    h.send(500, 'turn', { type: 'turn.user', turnId: 'u0', text: 'The tower is in Berlin.', interject: 'c0' });
    expect(messages(h.ios[1].request)).toEqual(['sys', 'The tower is in Berlin.—']);
  });

  it('drops a check the turn does not keep, leaving no trace in the history', () => {
    const h = harness(llmNode, params);
    h.send(0, 'interject', check('The tower is in Berlin.'));
    h.ios[0].h.onRecv!(tok('Paris.'), undefined);
    h.send(500, 'turn', { type: 'turn.user', turnId: 'u0', text: 'The tower is in Berlin, I think.' });
    expect(h.ios[0].closed).toBe(true);
    expect(h.of('llm.check_end')).toMatchObject([{ status: 'dropped' }]);
    expect(messages(h.ios[1].request)).toEqual(['sys', 'The tower is in Berlin, I think.']);
  });
});

describe('turn-taking on an interjection', () => {
  const sr = 16;
  const asr = (final: boolean, t: number, text: string) => ({ type: final ? 'asr.final' : 'asr.partial', text, start: (t - 800) * sr, end: t * sr });

  it('tells the words so far and commits them on a take', () => {
    const h = harness(duplexTurnNode, { useModel: false });
    h.send(0, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(800, 'transcript', asr(true, 800, 'The tower'));
    h.send(1200, 'transcript', asr(false, 1200, 'is in Berlin.'));
    expect(h.of('turn.words').map((m) => m.text)).toEqual(['The tower', 'The tower is in Berlin.']);
    h.send(1500, 'vad', { type: 'vad.speech_end', at: 1500 * sr });
    // A draft would go out once ASR caught up; this take comes first.
    h.send(1500, 'interject', { type: 'interject.take', checkId: 'c0', why: 'the user paused' });
    expect(h.of('turn.interject')).toMatchObject([{ checkId: 'c0', taken: true }]);
    expect(h.of('turn.user')).toMatchObject([{ text: 'The tower is in Berlin.', interject: 'c0' }]);
    expect(h.of('turn.words').at(-1)).toMatchObject({ text: '' });
  });

  it('takes the floor over a draft that went out at the same pause', () => {
    const h = harness(duplexTurnNode, { useModel: false });
    h.send(0, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(800, 'transcript', asr(false, 800, 'what about'));
    h.send(1000, 'vad', { type: 'vad.speech_end', at: 1000 * sr });
    h.send(1200, 'transcript', asr(false, 1200, 'What about Osaka?'));
    expect(h.of('turn.draft')).toHaveLength(1);
    h.send(1300, 'interject', { type: 'interject.take', checkId: 'c0' });
    expect(h.of('turn.interject')).toMatchObject([{ taken: true }]);
    expect(h.of('turn.draft_drop')).toMatchObject([{ why: 'early interjection' }]);
    expect(h.of('turn.user')).toMatchObject([{ text: 'What about Osaka?', interject: 'c0' }]);
    expect(h.of('turn.user')[0].draft).toBeUndefined();
  });

  it('tells the words are gone when a timer commits the turn', () => {
    const h = harness(duplexTurnNode, { useModel: false, maxSilenceMs: 500 });
    h.send(0, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(800, 'transcript', asr(true, 800, 'The tower is in Berlin', ));
    // A Finalize'd final while still speaking is held; the silence timer commits it.
    h.send(900, 'transcript', { ...asr(true, 900, 'and'), fromFinalize: true });
    h.send(1000, 'vad', { type: 'vad.speech_end', at: 1000 * sr });
    h.advance(2000);
    expect(h.of('turn.user')).toHaveLength(1);
    expect(h.of('turn.words').at(-1)).toMatchObject({ text: '' });
  });
});
