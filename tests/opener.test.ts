import { describe, expect, it } from 'vitest';
import { llmNode } from '../src/nodes/llm';
import { resolveParams, type IOHandlers, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function harness(params: Record<string, unknown>) {
  let now = 0;
  const out: Msg[] = [];
  const ios: { request: Json; h: IOHandlers; closed: boolean }[] = [];
  const ctx: NodeContext = {
    id: 'llm',
    params: resolveParams(llmNode, params as never),
    now: () => now,
    cause: undefined,
    emit: (_port, msg) => out.push(msg),
    setState() {},
    log() {},
    setTimer: () => ({ clear() {} }),
    io: {
      open(_channel, request, h) {
        const x = { request, h, closed: false };
        ios.push(x);
        return { xid: String(ios.length), send() {}, close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: { append: () => { throw new Error('no media'); } },
  };
  const node = llmNode.create(ctx as never);
  const send = (t: number, msg: Msg) => ((now = t), node.onInput?.('turn', msg));
  const of = (type: string) => out.filter((m) => m.type === type);
  return { send, of, ios, at: (t: number) => void (now = t) };
}

const params = { provider: 'openai-compatible', model: 'big', openerModel: 'fast', system: 'sys' };
const tok = (text: string) => ({ choices: [{ index: 0, delta: { content: text } }] });
const done = { $done: true };
const body = (r: Json) => (r as { body: { model: string; messages: { role: string; content: string }[] } }).body;

describe('the opener', () => {
  it('speaks first when it is ready before the reply, and the reply carries on after it', () => {
    const h = harness(params);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'what is 17 times 23' });
    const [main, op] = h.ios;
    expect(body(main.request).model).toBe('big');
    expect(body(main.request).messages[0].content).toContain('Do not open with an acknowledgement');
    expect(body(op.request).model).toBe('fast');
    op.h.onRecv!(tok('Hmm, let me'), undefined);
    op.h.onRecv!(tok(' see.'), undefined);
    op.h.onRecv!(done, undefined);
    expect(h.of('llm.delta')).toMatchObject([{ text: 'Hmm, let me see. ', opener: true }]);
    expect(h.of('llm.opener_end')).toMatchObject([{ status: 'spoken', text: 'Hmm, let me see.' }]);
    main.h.onRecv!(tok('It is 391.'), undefined);
    main.h.onRecv!(done, undefined);
    expect(h.of('llm.end')).toMatchObject([{ text: 'Hmm, let me see. It is 391.' }]);
    h.send(5000, { type: 'turn.user', turnId: 'u1', text: 'thanks' });
    expect(body(h.ios[2].request).messages.map((m) => m.content)).toContain('Hmm, let me see. It is 391.');
  });

  it('is skipped when the reply talks first', () => {
    const h = harness(params);
    h.send(1000, { type: 'turn.user', turnId: 'u0', text: 'hi' });
    const [main, op] = h.ios;
    main.h.onRecv!(tok('Hi there!'), undefined);
    expect(op.closed).toBe(true);
    expect(h.of('llm.opener_end')).toMatchObject([{ status: 'skipped' }]);
    op.h.onRecv!(tok('Yeah.'), undefined);
    op.h.onRecv!(done, undefined);
    expect(h.of('llm.delta').map((d) => d.text)).toEqual(['Hi there!']);
  });

  it('runs with a draft and is spoken when the kept draft has nothing yet', () => {
    const h = harness(params);
    h.send(1000, { type: 'turn.draft', draftId: 'd0', text: 'plan my trip' });
    const [, op] = h.ios;
    op.h.onRecv!(tok('Sure, one moment.'), undefined);
    op.h.onRecv!(done, undefined);
    expect(h.of('llm.delta')).toHaveLength(0);
    h.send(1400, { type: 'turn.user', turnId: 'u0', text: 'plan my trip', draft: 'd0' });
    expect(h.ios).toHaveLength(2);
    expect(h.of('llm.delta')).toMatchObject([{ text: 'Sure, one moment. ', opener: true }]);
  });

  it('goes with a dropped draft', () => {
    const h = harness(params);
    h.send(1000, { type: 'turn.draft', draftId: 'd0', text: 'plan my' });
    h.send(1400, { type: 'turn.user', turnId: 'u0', text: 'plan my trip to Osaka' });
    expect(h.ios[1].closed).toBe(true);
    expect(h.of('llm.opener_end')[0]).toMatchObject({ status: 'cancelled' });
    expect(h.ios).toHaveLength(4);
  });
});
