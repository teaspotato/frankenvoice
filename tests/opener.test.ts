import { describe, expect, it } from 'vitest';
import { llmNode } from '../src/nodes/llm';
import { choiceRequest, parseOpenerChoices, parseOpenerLines, pickOpener, readChoice } from '../src/nodes/llm/opener';
import { resolveParams, type IOHandlers, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function harness(params: Record<string, unknown>) {
  let now = 0;
  const out: Msg[] = [];
  const timers: { at: number; fn: () => void }[] = [];
  const ios: { channel: string; request: Json; h: IOHandlers; closed: boolean }[] = [];
  const ctx: NodeContext = {
    id: 'llm',
    params: resolveParams(llmNode, params as never),
    now: () => now,
    cause: undefined,
    emit: (_port, msg) => out.push(msg),
    setState() {},
    log() {},
    setTimer: (ms, fn) => (timers.push({ at: now + ms, fn }), { clear() {} }),
    io: {
      open(channel, request, h) {
        const x = { channel, request, h, closed: false };
        ios.push(x);
        return { xid: String(ios.length), send() {}, close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: { append: () => { throw new Error('no media'); } },
  };
  const node = llmNode.create(ctx as never);
  node.start?.();
  const send = (t: number, msg: Msg) => ((now = t), node.onInput?.('turn', msg));
  const of = (type: string) => out.filter((m) => m.type === type);
  const chat = () => ios.filter((x) => x.channel === 'openai');
  const embeds = () => ios.filter((x) => x.channel === 'embed');
  const advance = (t: number) => {
    now = t;
    const due = timers.filter((x) => x.at <= t);
    for (const x of due) timers.splice(timers.indexOf(x), 1), x.fn();
  };
  return { send, of, ios, chat, embeds, advance, choices: () => ios.filter((x) => x.channel === 'typesafe') };
}

const lines = 'Hmm, let me think. = why is that | which one\n- = thanks';
const params = { provider: 'openai-compatible', model: 'big', system: 'sys', opener: 'canned', openerLines: lines, openerMinScore: 0.5 };
const vecs = { why: [1, 0, 0], which: [0.8, 0.6, 0], thanks: [0, 0, 1] };
const tok = (text: string) => ({ choices: [{ index: 0, delta: { content: text } }] });
const done = { $done: true };
const body = (r: Json) => (r as { body: { messages: { role: string; content: string }[] } }).body;
const answer = (x: { h: IOHandlers }, v: number[]) => x.h.onRecv!({ vectors: [v] }, undefined);

/** A harness whose encoder has embedded the examples. */
function ready() {
  const h = harness(params);
  expect(h.embeds()[0].request).toEqual({ model: 'Xenova/all-MiniLM-L6-v2', texts: ['why is that', 'which one', 'thanks'] });
  h.embeds()[0].h.onRecv!({ vectors: [vecs.why, vecs.which, vecs.thanks] }, undefined);
  return h;
}

describe('canned openers', () => {
  it('parses lines and picks the nearest example, or nothing below the threshold', () => {
    const ex = parseOpenerLines(`# comment\n${lines}`);
    expect(ex).toEqual([
      { line: 'Hmm, let me think.', text: 'why is that' },
      { line: 'Hmm, let me think.', text: 'which one' },
      { line: null, text: 'thanks' },
    ]);
    const all = [vecs.why, vecs.which, vecs.thanks];
    expect(pickOpener([0.6, 0.8, 0], ex, all, 0.5)).toEqual({ line: 'Hmm, let me think.', score: 0.96, example: 'which one' });
    expect(pickOpener([0, 0.1, 0.99], ex, all, 0.5).line).toBeNull();
    expect(pickOpener([0, 1, 0], ex, all, 0.7)).toMatchObject({ line: null, score: 0.6 });
  });

  it('says the line the moment the turn commits, and the request knows it was said', () => {
    const h = ready();
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'why is the sky blue' });
    expect(h.of('llm.start')).toHaveLength(1);
    expect(h.chat()).toHaveLength(0); // waits for the pick
    answer(h.embeds()[1], [0.9, 0.1, 0]);
    expect(h.of('llm.opener')).toMatchObject([{ text: 'why is the sky blue', line: 'Hmm, let me think.', example: 'why is that' }]);
    expect(h.of('llm.delta')).toMatchObject([{ text: 'Hmm, let me think. ', opener: true }]);
    const main = h.chat()[0];
    expect(body(main.request).messages[0].content).toContain('You have already said aloud: "Hmm, let me think."');
    main.h.onRecv!(tok('Because of scattering.'), undefined);
    main.h.onRecv!(done, undefined);
    expect(h.of('llm.end')).toMatchObject([{ text: 'Hmm, let me think. Because of scattering.' }]);
    h.send(5000, { type: 'turn.user', turnId: 'u1', text: 'thanks' });
    answer(h.embeds()[2], vecs.thanks);
    expect(body(h.chat()[1].request).messages.map((m) => m.content)).toEqual(['sys', 'why is the sky blue', 'Hmm, let me think. Because of scattering.', 'thanks']);
  });

  it('says nothing for a turn that needs no opener', () => {
    const h = ready();
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'thanks!' });
    answer(h.embeds()[1], vecs.thanks);
    expect(h.of('llm.opener')).toMatchObject([{ line: null }]);
    expect(h.of('llm.delta')).toHaveLength(0);
    expect(body(h.chat()[0].request).messages[0].content).toBe('sys');
  });

  it('picks on the draft, so a kept draft speaks the line before its held tokens', () => {
    const h = ready();
    h.send(1000, { type: 'turn.draft', draftId: 'd0', text: 'which is better' });
    answer(h.embeds()[1], vecs.which);
    expect(h.of('llm.delta')).toHaveLength(0);
    h.chat()[0].h.onRecv!(tok('The first.'), undefined);
    h.send(1400, { type: 'turn.draft', draftId: 'd1', text: 'which is better' });
    expect(h.embeds()).toHaveLength(2); // same words: picked once
    h.chat()[1].h.onRecv!(tok('The first.'), undefined);
    h.send(1500, { type: 'turn.user', turnId: 'u0', text: 'which is better', draft: 'd1' });
    expect(h.of('llm.delta').map((d) => d.text)).toEqual(['Hmm, let me think. ', 'The first.']);
  });

  it('does not wait for an encoder that is not ready', () => {
    const h = harness(params);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'why is the sky blue' });
    expect(h.chat()).toHaveLength(1);
    expect(h.embeds()).toHaveLength(1);
    expect(h.of('llm.delta')).toHaveLength(0);
  });

  it('sends the request without an opener when the encoder fails', () => {
    const h = ready();
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'why' });
    h.embeds()[1].h.onClose!({ error: 'boom' });
    expect(h.chat()).toHaveLength(1);
    expect(h.of('llm.delta')).toHaveLength(0);
  });

  it('tells TTS the lines at start', () => {
    const h = harness(params);
    expect(h.of('llm.opener_lines')).toEqual([{ type: 'llm.opener_lines', lines: ['Hmm, let me think.'] }]);
  });
});

describe('openers picked by TypeSafe Choice', () => {
  const options = 'Hmm, let me think. = needs weighing up\nOh? = a teaser, invite them to go on\n- = anything else';
  const cparams = { provider: 'openai-compatible', model: 'big', system: 'sys', opener: 'choice', openerChoices: options, openerInstructions: 'Pick.', openerMinConfidence: 0.5 };
  const answer = (choice: string, confidence: number) => ({ model: 'jev-1.13.0', answers: { opener: { type: 'choice', choice, confidence, probabilities: {} } } });

  it('builds the request from the whole conversation and reads the answer', () => {
    const req = choiceRequest(
      [
        { role: 'user', text: 'hi' },
        { role: 'assistant', text: 'Hello!' },
        { role: 'user', text: "I'm cooking some interesting stuff." },
      ],
      parseOpenerChoices(options),
      'Pick.',
      'jev-latest',
    );
    expect(req).toEqual({
      model: 'jev-latest',
      state: "User: hi\nAgent: Hello!\nUser: I'm cooking some interesting stuff.",
      questions: { opener: { type: 'choice', instructions: 'Pick.', criteria: { 'Hmm, let me think.': 'needs weighing up', 'Oh?': 'a teaser, invite them to go on', '-': 'anything else' } } },
    });
    expect(readChoice(answer('Oh?', 0.8), 0.5)).toEqual({ line: 'Oh?', score: 0.8, example: 'Oh?' });
    expect(readChoice(answer('Oh?', 0.4), 0.5)?.line).toBeNull();
    expect(readChoice(answer('-', 0.9), 0.5)?.line).toBeNull();
    expect(readChoice({ error: 'x' }, 0.5)).toBeNull();
  });

  it('sends the conversation, says the chosen line and tells the model', () => {
    const h = harness(cparams);
    expect(h.of('llm.opener_lines')).toEqual([{ type: 'llm.opener_lines', lines: ['Hmm, let me think.', 'Oh?'] }]);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: "I'm cooking some interesting stuff." });
    expect(h.chat()).toHaveLength(0);
    expect(h.choices()[0].request).toMatchObject({ path: '/v1/systemone', body: { state: "User: I'm cooking some interesting stuff." } });
    h.choices()[0].h.onRecv!(answer('Oh?', 0.9), undefined);
    expect(h.of('llm.delta')).toMatchObject([{ text: 'Oh? ', opener: true }]);
    expect(body(h.chat()[0].request).messages[0].content).toContain('You have already said aloud: "Oh?"');
  });

  it('gives up waiting after openerWaitMs and sends the request without an opener', () => {
    const h = harness({ ...cparams, openerWaitMs: 300 });
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'why is the sky blue' });
    h.advance(1299);
    expect(h.chat()).toHaveLength(0);
    h.advance(1300);
    expect(h.chat()).toHaveLength(1);
    expect(body(h.chat()[0].request).messages[0].content).toBe('sys');
    h.choices()[0].h.onRecv!(answer('Hmm, let me think.', 0.9), undefined);
    expect(h.of('llm.delta')).toHaveLength(0);
    expect(h.chat()).toHaveLength(1);
  });

  it('asks again when the conversation before the same words differs', () => {
    const h = harness(cparams);
    h.send(1000, { type: 'turn.draft', draftId: 'd0', text: 'yeah' });
    h.send(1100, { type: 'turn.draft', draftId: 'd1', text: 'yeah' });
    expect(h.choices()).toHaveLength(1);
  });
});

describe('fillers while a reasoning model thinks', () => {
  const fparams = { provider: 'openai-compatible', model: 'big', system: 'sys', fillerLines: 'Let me see.\nOne moment.', fillerAfterMs: 500 };
  const think = { choices: [{ index: 0, delta: { reasoning: 'hmm' } }] };

  it('says a filler when the model is still thinking 500 ms after the turn, and keeps it out of the history', () => {
    const h = harness(fparams);
    expect(h.of('llm.opener_lines')).toEqual([{ type: 'llm.opener_lines', lines: ['Let me see.', 'One moment.'] }]);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'plan a week in Japan' });
    const main = h.chat()[0];
    expect(body(main.request).messages[0].content).toBe('sys');
    main.h.onRecv!(think, undefined);
    h.advance(1499);
    expect(h.of('llm.delta')).toHaveLength(0);
    h.advance(1500);
    expect(h.of('llm.delta')).toMatchObject([{ text: 'Let me see. ', opener: true }]);
    main.h.onRecv!(tok('Day one: Tokyo.'), undefined);
    main.h.onRecv!(done, undefined);
    expect(h.of('llm.end')).toMatchObject([{ text: 'Let me see. Day one: Tokyo.' }]);
    h.send(5000, { type: 'turn.user', turnId: 'u1', text: 'and then?' });
    expect(body(h.chat()[1].request).messages.map((m) => m.content)).toEqual(['sys', 'plan a week in Japan', 'Day one: Tokyo.', 'and then?']);
  });

  it('says nothing when the model answers without thinking, or starts talking in time', () => {
    const h = harness(fparams);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'hi' });
    h.advance(3000);
    expect(h.of('llm.delta')).toHaveLength(0);
    h.send(4000, { type: 'turn.user', turnId: 'u1', text: 'why?' });
    h.chat()[1].h.onRecv!(think, undefined);
    h.chat()[1].h.onRecv!(tok('Because.'), undefined);
    h.advance(5000);
    expect(h.of('llm.delta').map((d) => d.text)).toEqual(['Because.']);
  });

  it('fills at once when thinking starts late, and does not repeat the last filler', () => {
    const h = harness(fparams);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'a' });
    h.advance(2000);
    h.chat()[0].h.onRecv!(think, undefined);
    h.chat()[0].h.onRecv!(done, undefined);
    h.send(3000, { type: 'turn.user', turnId: 'u1', text: 'b' });
    h.chat()[1].h.onRecv!(think, undefined);
    h.advance(3500);
    expect(h.of('llm.delta').map((d) => d.text)).toEqual(['Let me see. ', 'One moment. ']);
  });
});
