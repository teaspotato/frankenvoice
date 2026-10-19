import { describe, expect, it } from 'vitest';
import { heardOfReply, llmNode } from '../src/nodes/llm';
import { fillerOpenerNode } from '../src/nodes/opener';
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
  const advance = (t: number) => {
    now = t;
    const due = timers.filter((x) => x.at <= t);
    for (const x of due) timers.splice(timers.indexOf(x), 1), x.fn();
  };
  const cancel = (t: number) => ((now = t), node.onInput?.('control', { type: 'cancel' }));
  return { send, cancel, of, ios, chat, advance };
}

const tok = (text: string) => ({ choices: [{ index: 0, delta: { content: text } }] });
const done = { $done: true };
const body = (r: Json) => (r as { body: { messages: { role: string; content: string }[] } }).body;

describe('fillers while a reasoning model thinks', () => {
  const fparams = { provider: 'openai-compatible', model: 'big', system: 'sys' };
  const think = { choices: [{ index: 0, delta: { reasoning: 'hmm' } }] };

  /** The opener stage on its own, fed the LLM's stream. */
  function opener(params: Record<string, unknown> = { lines: 'Let me see.\nOne moment.', afterMs: 500 }) {
    let now = 0;
    const out: Msg[] = [];
    const timers: { at: number; fn: () => void; dead: boolean }[] = [];
    const ctx = {
      id: 'opener',
      params: resolveParams(fillerOpenerNode, params as never),
      now: () => now,
      emit: (_port: string, msg: Msg) => out.push(msg),
      setState() {},
      log() {},
      setTimer: (ms: number, fn: () => void) => {
        const x = { at: now + ms, fn, dead: false };
        timers.push(x);
        return { clear: () => void (x.dead = true) };
      },
    };
    const node = fillerOpenerNode.create(ctx as never);
    node.start?.();
    const advance = (t: number) => {
      now = t;
      for (const x of timers.filter((y) => y.at <= t && !y.dead)) (x.dead = true), x.fn();
    };
    const send = (t: number, msg: Msg) => (advance(t), node.onInput?.('text', msg));
    return { out, send, advance, said: () => out.filter((m) => m.type === 'opener.say').map((m) => m.line) };
  }

  it('says a filler when the model is still thinking 500 ms after the reply started', () => {
    const o = opener();
    expect(o.out).toEqual([{ type: 'opener.lines', lines: ['Let me see.', 'One moment.'] }]);
    o.send(1000, { type: 'llm.start', rid: 'r0' });
    o.send(1100, { type: 'llm.thinking', rid: 'r0' });
    o.advance(1499);
    expect(o.said()).toEqual([]);
    o.advance(1500);
    expect(o.out[1]).toMatchObject({ type: 'opener.say', rid: 'r0', text: 'Let me see. ', reason: 'still thinking 500 ms after the turn' });
  });

  it('says nothing when the model answers without thinking, starts talking in time, or is cut off', () => {
    const o = opener();
    o.send(1000, { type: 'llm.start', rid: 'r0' });
    o.advance(3000);
    o.send(4000, { type: 'llm.start', rid: 'r1' });
    o.send(4100, { type: 'llm.thinking', rid: 'r1' });
    o.send(4300, { type: 'llm.delta', rid: 'r1', text: 'Because.' });
    o.advance(5000);
    o.send(6000, { type: 'llm.start', rid: 'r2' });
    o.send(6100, { type: 'llm.thinking', rid: 'r2' });
    o.send(6200, { type: 'llm.end', rid: 'r2', status: 'cancelled' });
    o.advance(7000);
    expect(o.said()).toEqual([]);
  });

  it('fills at once when thinking starts late, and does not repeat the last filler', () => {
    const o = opener();
    o.send(1000, { type: 'llm.start', rid: 'r0' });
    o.send(2000, { type: 'llm.thinking', rid: 'r0' });
    o.send(2500, { type: 'llm.end', rid: 'r0' });
    o.send(3000, { type: 'llm.start', rid: 'r1' });
    o.send(3100, { type: 'llm.thinking', rid: 'r1' });
    o.advance(3500);
    expect(o.said()).toEqual(['Let me see.', 'One moment.']);
  });

  it('the LLM reports thinking once and its requests never mention the opener', () => {
    const h = harness(fparams);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'plan a week in Japan' });
    const main = h.chat()[0];
    main.h.onRecv!(think, undefined);
    main.h.onRecv!(think, undefined);
    expect(h.of('llm.thinking')).toMatchObject([{ rid: 'llm-r0' }]);
    main.h.onRecv!(tok('Day one: Tokyo.'), undefined);
    main.h.onRecv!(done, undefined);
    h.send(5000, { type: 'turn.user', turnId: 'u1', text: 'and then?' });
    expect(body(h.chat()[1].request).messages.map((m) => m.content)).toEqual(['sys', 'plan a week in Japan', 'Day one: Tokyo.', 'and then?']);
  });

  it('keeps only the reply\'s own words of what was heard', () => {
    expect(heardOfReply('Let me see. Day one: Tokyo', 'Day one: Tokyo. Day two: Kyoto.')).toBe('Day one: Tokyo');
    expect(heardOfReply('Let me see.', 'Day one: Tokyo.')).toBe('');
    expect(heardOfReply('Hmm, let me think. Sure', 'Hmm, let me think. Sure, here.')).toBe('Hmm, let me think. Sure');
    const h = harness(fparams);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'plan a week' });
    h.chat()[0].h.onRecv!(tok('Day one: Tokyo. Day two: Kyoto.'), undefined);
    h.cancel(2000);
    h.send(2000, { type: 'turn.heard', turnId: 'u0', text: 'One moment. Day one: Tokyo.' });
    h.send(3000, { type: 'turn.user', turnId: 'u1', text: 'wait' });
    expect(body(h.chat()[1].request).messages.map((m) => m.content)).toEqual(['sys', 'plan a week', 'Day one: Tokyo.—', 'wait']);
  });
});
