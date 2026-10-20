import { describe, expect, it } from 'vitest';
import { choiceOpenerNode } from '../src/nodes/opener';
import { parseChoiceOptions, readChoice } from '../src/nodes/typesafe';
import { resolveParams, type IOHandlers } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function opener(params: Record<string, unknown> = {}) {
  let now = 0;
  const out: Msg[] = [];
  const timers: { at: number; fn: () => void; dead: boolean }[] = [];
  const ios: { request: Json; h: IOHandlers }[] = [];
  const ctx = {
    id: 'opener',
    params: resolveParams(choiceOpenerNode, { options: 'Good question. = how or why\nOh no. | Oh dear. = bad news\n- = anything else', afterMs: 200, ...params } as never),
    now: () => now,
    emit: (_port: string, msg: Msg) => out.push(msg),
    setState() {},
    log() {},
    setTimer: (ms: number, fn: () => void) => {
      const x = { at: now + ms, fn, dead: false };
      timers.push(x);
      return { clear: () => void (x.dead = true) };
    },
    io: {
      open(_channel: string, request: Json, h: IOHandlers) {
        ios.push({ request, h });
        return { xid: String(ios.length), send() {}, close() {}, closed: false };
      },
    },
  };
  const node = choiceOpenerNode.create(ctx as never);
  node.start?.();
  const advance = (t: number) => {
    now = t;
    for (const x of timers.filter((y) => y.at <= t && !y.dead)) (x.dead = true), x.fn();
  };
  const send = (t: number, port: string, msg: Msg) => (advance(t), node.onInput?.(port, msg));
  const answer = (i: number, choice: string, confidence = 0.9) => ios[i].h.onRecv?.({ answers: { pick: { type: 'choice', choice, confidence, probabilities: {} } } }, undefined);
  return { out, ios, send, advance, answer, said: () => out.filter((m) => m.type === 'opener.say').map((m) => m.line) };
}

describe('TypeSafe opener', () => {
  it('parses options with alternatives and reads a pick', () => {
    const opts = parseChoiceOptions('# c\nOh no. | Oh dear. = bad news\n- = else\nbroken line\n');
    expect(opts.map((o) => [o.key, o.lines])).toEqual([['Oh no.', ['Oh no.', 'Oh dear.']], ['-', []]]);
    const res = { answers: { pick: { choice: 'Oh no.', confidence: 0.61234 } } };
    expect(readChoice(res, opts, 0.5)?.option?.key).toBe('Oh no.');
    expect(readChoice(res, opts, 0.7)?.option).toBeNull();
    expect(readChoice({ answers: { pick: { choice: '-', confidence: 1 } } }, opts, 0)?.option).toBeNull();
    expect(readChoice({}, opts, 0)).toBeNull();
  });

  it("asks with the user's words as soon as a draft has them, and says the pick when the reply has no words after afterMs", () => {
    const h = opener();
    expect(h.out[0]).toEqual({ type: 'opener.lines', lines: ['Good question.', 'Oh no.', 'Oh dear.'] });
    h.send(0, 'turn', { type: 'turn.draft', draftId: 'd1', text: 'How does it work?' });
    expect((h.ios[0].request as { body: { state: string } }).body.state).toBe('How does it work?');
    h.answer(0, 'Good question.');
    h.send(300, 'turn', { type: 'turn.user', turnId: 'u1', text: 'How does it work?' });
    h.send(300, 'text', { type: 'llm.start', rid: 'r1', turnId: 'u1' });
    expect(h.ios).toHaveLength(1);
    expect(h.said()).toEqual([]);
    h.advance(500);
    expect(h.said()).toEqual(['Good question.']);
    expect(h.out.find((m) => m.type === 'opener.say')).toMatchObject({ rid: 'r1', text: 'Good question. ' });
  });

  it('says nothing once the reply has words, on "-", or for a reply nobody asked for', () => {
    const h = opener();
    h.send(0, 'turn', { type: 'turn.user', turnId: 'u1', text: 'How does it work?' });
    h.send(0, 'text', { type: 'llm.start', rid: 'r1', turnId: 'u1' });
    h.send(100, 'text', { type: 'llm.delta', rid: 'r1', text: 'It ' });
    h.answer(0, 'Good question.');
    h.advance(400);
    h.send(1000, 'turn', { type: 'turn.user', turnId: 'u2', text: 'Hi there.' });
    h.send(1000, 'text', { type: 'llm.start', rid: 'r2', turnId: 'u2' });
    h.answer(1, '-');
    h.advance(1500);
    h.send(2000, 'text', { type: 'llm.start', rid: 'r3', turnId: null });
    h.advance(2500);
    expect(h.said()).toEqual([]);
    expect(h.ios).toHaveLength(2);
  });

  it('says a late pick if the reply is still silent, and takes the lines of an option in turn', () => {
    const h = opener();
    // The reply starts before the committed turn reaches the opener.
    h.send(0, 'text', { type: 'llm.start', rid: 'r1', turnId: 'u1' });
    h.send(0, 'turn', { type: 'turn.user', turnId: 'u1', text: 'My flight got cancelled.' });
    h.advance(400);
    h.answer(0, 'Oh no.');
    h.send(2000, 'turn', { type: 'turn.user', turnId: 'u2', text: 'And they lost my bag.' });
    h.send(2000, 'text', { type: 'llm.start', rid: 'r2', turnId: 'u2' });
    h.answer(1, 'Oh no.');
    h.advance(2300);
    expect(h.said()).toEqual(['Oh no.', 'Oh dear.']);
  });
});
