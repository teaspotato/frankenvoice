import { describe, expect, it } from 'vitest';
import { cutToHeard, heardOfReply, llmNode } from '../src/nodes/llm';
import { resolveParams, type IOHandlers, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function harness(params: Record<string, unknown>) {
  let now = 0;
  const out: Msg[] = [];
  const timers: { at: number; fn: () => void }[] = [];
  const ios: { channel: string; request: Json; h: IOHandlers; closed: boolean }[] = [];
  const ctx: NodeContext = {
    id: 'llm',
    params: resolveParams(llmNode, { utterances: 'plain', ...params } as never),
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


describe('LLM history', () => {
  const fparams = { model: 'big', system: 'sys' };
  const think = { choices: [{ index: 0, delta: { reasoning: 'hmm' } }] };

  it('the LLM reports thinking once and keeps its history to the turns and replies', () => {
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

  it('cuts a reply spanning several turns to what was heard', () => {
    const history = [
      { role: 'user' as const, text: 'q', turnId: 'u0' },
      { role: 'assistant' as const, text: 'Let me look that up.', rid: 'r0' },
      { role: 'assistant' as const, text: 'It will take a moment.', rid: 'r0' },
      { role: 'user' as const, text: 'ok', turnId: 'u1' },
    ];
    expect(cutToHeard(history, [1, 2], 'Let me look that up. It will').map((t) => t.text)).toEqual(['q', 'Let me look that up.', 'It will—', 'ok']);
    expect(cutToHeard(history, [1, 2], 'Let me look').map((t) => t.text)).toEqual(['q', 'Let me look—', 'ok']);
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
