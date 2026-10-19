import { describe, expect, it } from 'vitest';
import { llmNode, cutToHeard } from '../src/nodes/llm';
import { agentNode } from '../src/nodes/agent';
import { FLOOR } from '../src/nodes/turn';
import { DEFAULT_DELEGATE_PROMPT } from '../src/nodes/llm/delegate';
import { buildGraph, defaultConfig } from '../src/pipeline';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, JsonObject, Msg } from '../src/core/types';
import type { ChatTurn } from '../src/nodes/llm/types';

/** Drives one node by hand: a manual clock, timers, fake IO, and a list of what it emitted. */
function harness(def: NodeDef, params: Record<string, unknown> = {}) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ios: { request: JsonObject; h: IOHandlers; closed: boolean }[] = [];
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
      open(_channel, request, h) {
        const x = { request: request as JsonObject, h, closed: false };
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

/** A Messages API stream: text, then optional blocks given whole (tool_use input is streamed as JSON). */
function stream(io: { h: IOHandlers }, text: string, blocks: JsonObject[] = [], stopReason = blocks.some((b) => b.type === 'tool_use') ? 'tool_use' : 'end_turn') {
  const r = (e: Json) => io.h.onRecv?.(e, undefined);
  let i = 0;
  if (text) {
    r({ type: 'content_block_start', index: i, content_block: { type: 'text', text: '' } });
    r({ type: 'content_block_delta', index: i, delta: { type: 'text_delta', text } });
    r({ type: 'content_block_stop', index: i++ });
  }
  for (const b of blocks) {
    const { input, ...rest } = b;
    r({ type: 'content_block_start', index: i, content_block: input !== undefined ? { ...rest, input: {} } : b });
    if (input !== undefined) r({ type: 'content_block_delta', index: i, delta: { type: 'input_json_delta', partial_json: JSON.stringify(input) } });
    r({ type: 'content_block_stop', index: i++ });
  }
  r({ type: 'message_delta', delta: { stop_reason: stopReason } });
  r({ type: 'message_stop' });
  io.h.onClose?.(null);
}

const delegate = { prompt: DEFAULT_DELEGATE_PROMPT, progress: 'context', quietMs: 800 };
const turn = (turnId: string, text: string) => ({ type: 'turn.user', turnId, text });
const floor = (to: string) => ({ type: 'turn.transition', to });

describe('the conversation LLM delegates', () => {
  it('starts a task with a tool call and continues the same reply with the result', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'What is the weather in Kyoto?'));
    const req = h.ios[0].request;
    expect((req.tools as JsonObject[]).map((t) => t.name)).toEqual(['task_start', 'task_send', 'task_cancel']);
    expect(String(req.system)).toContain('task_update');
    stream(h.ios[0], 'Let me look that up. ', [{ type: 'tool_use', id: 'tu1', name: 'task_start', input: { task: 'Weather in Kyoto today' } }]);
    expect(h.of('task.start')).toMatchObject([{ taskId: 't1', task: 'Weather in Kyoto today' }]);
    expect(h.out.find((o) => o.msg.type === 'task.start')!.port).toBe('tasks');
    // Same reply, a second request carrying the tool result.
    expect(h.ios).toHaveLength(2);
    const msgs = h.ios[1].request.messages as JsonObject[];
    expect(msgs.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect((msgs[2].content as JsonObject[])[0]).toMatchObject({ type: 'tool_result', tool_use_id: 'tu1' });
    expect(h.of('llm.end')).toHaveLength(0);
    stream(h.ios[1], 'It should only take a moment.');
    expect(h.of('llm.start')).toHaveLength(1);
    expect(h.of('llm.end')).toMatchObject([{ status: 'done', text: 'Let me look that up. It should only take a moment.' }]);
  });

  it('speaks a result once the floor is free and the user has paused', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Find me a ramen place.'));
    stream(h.ios[0], 'On it. ', [{ type: 'tool_use', id: 'tu1', name: 'task_start', input: { task: 'ramen' } }]);
    stream(h.ios[1], '');
    h.send(100, 'turn', floor('speaking'));
    h.send(2000, 'agent', { type: 'agent.done', taskId: 't1', text: 'Ichiran in Gion, open until 2 am.' });
    expect(h.of('llm.start')).toHaveLength(1);
    h.send(2400, 'vad', { type: 'vad.speech_start' });
    h.send(2500, 'turn', floor('listening'));
    h.send(3000, 'vad', { type: 'vad.speech_end' });
    h.advance(3700);
    expect(h.of('llm.start')).toHaveLength(1);
    h.advance(3800);
    expect(h.of('llm.start')).toMatchObject([{}, { turnId: null, tasks: ['t1'] }]);
    const msgs = h.ios[2].request.messages as JsonObject[];
    const last = msgs[msgs.length - 1].content as JsonObject[];
    expect(String(last[last.length - 1].text)).toMatch(/<task_update id="t1" kind="result">\nIchiran in Gion/);
  });

  it('keeps progress notes for the next request instead of speaking them', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Plan a trip.'));
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'tu1', name: 'task_start', input: { task: 'trip' } }]);
    stream(h.ios[1], 'Working on it.');
    h.send(5000, 'agent', { type: 'agent.progress', taskId: 't1', text: 'Found three hotels.' });
    h.advance(9000);
    expect(h.ios).toHaveLength(2);
    h.send(9000, 'turn', turn('u1', 'How is it going?'));
    const text = JSON.stringify(h.ios[2].request.messages);
    expect(text).toContain('kind=\\"progress\\">\\nFound three hotels.');
    expect(text.indexOf('Found three hotels')).toBeLessThan(text.indexOf('How is it going?'));
  });

  it('answers a bad task id with an error result', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Cancel it.'));
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'tu1', name: 'task_cancel', input: { task_id: 't9' } }]);
    expect(h.of('task.cancel')).toHaveLength(0);
    expect(((h.ios[1].request.messages as JsonObject[])[2].content as JsonObject[])[0]).toMatchObject({ is_error: true });
  });

  it('has no tools without the background agent', () => {
    const h = harness(llmNode, {});
    h.send(0, 'turn', turn('u0', 'Hi'));
    expect(h.ios[0].request.tools).toBeUndefined();
  });

  it('cuts a reply that called a tool to what was heard, keeping the call', () => {
    const history: ChatTurn[] = [
      { role: 'user' as const, text: 'q', turnId: 'u0' },
      { role: 'assistant' as const, text: 'Let me look that up.', rid: 'r0', native: { provider: 'anthropic', content: [{ type: 'text', text: 'Let me look that up.' }, { type: 'tool_use', id: 'tu1', name: 'task_start', input: {} }] } },
      { role: 'user' as const, text: '', native: { provider: 'anthropic', content: [{ type: 'tool_result', tool_use_id: 'tu1', content: 'ok' }] } },
      { role: 'assistant' as const, text: 'It will take a moment.', rid: 'r0' },
    ];
    const cut = cutToHeard(history, [1, 3], 'Let me look');
    expect(cut).toHaveLength(3);
    expect(cut[1].text).toBe('Let me look—');
    expect((cut[1].native!.content as JsonObject[]).map((b) => b.type)).toEqual(['text', 'tool_use']);
  });
});

describe('the background agent', () => {
  it('runs a task on Sonnet with reasoning and web tools, and reports the result', () => {
    const h = harness(agentNode);
    h.send(0, 'tasks', { type: 'task.start', taskId: 't1', task: 'Weather in Kyoto' });
    const req = h.ios[0].request;
    expect(req).toMatchObject({ model: 'claude-sonnet-5-5', output_config: { effort: 'medium' }, thinking: { type: 'adaptive' } });
    expect((req.tools as JsonObject[]).map((t) => t.name)).toEqual(['web_search', 'web_fetch', 'report_progress']);
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'p1', name: 'report_progress', input: { note: 'Checking the forecast.' } }]);
    expect(h.of('agent.progress')).toMatchObject([{ taskId: 't1', text: 'Checking the forecast.' }]);
    // Steering while a request is out goes with the next one.
    h.send(500, 'tasks', { type: 'task.send', taskId: 't1', message: 'Tomorrow, not today.', mode: 'steer' });
    stream(h.ios[1], '', [], 'pause_turn');
    expect(h.ios).toHaveLength(3);
    expect(JSON.stringify(h.ios[2].request.messages)).not.toContain('Tomorrow');
    stream(h.ios[2], 'Sunny today.');
    expect(h.of('agent.done')).toHaveLength(0);
    expect(JSON.stringify(h.ios[3].request.messages)).toContain('Tomorrow, not today.');
    h.advance(4000);
    stream(h.ios[3], 'Tomorrow: rain, 18 degrees.');
    expect(h.of('agent.done')).toMatchObject([{ taskId: 't1', text: 'Tomorrow: rain, 18 degrees.', rounds: 4, ms: 4000 }]);
  });

  it('answers a side question without touching the task', () => {
    const h = harness(agentNode);
    h.send(0, 'tasks', { type: 'task.start', taskId: 't1', task: 'Compare three laptops' });
    h.send(100, 'tasks', { type: 'task.send', taskId: 't1', message: 'How far along are you?', mode: 'ask' });
    expect(h.ios[1].request).toMatchObject({ tool_choice: { type: 'none' }, output_config: { effort: 'low' } });
    stream(h.ios[1], 'Just started.');
    expect(h.of('agent.answer')).toMatchObject([{ taskId: 't1', question: 'How far along are you?', text: 'Just started.' }]);
    stream(h.ios[0], 'The X1 wins.');
    expect(h.of('agent.done')).toMatchObject([{ text: 'The X1 wins.', rounds: 1 }]);
  });

  it('stops on cancel', () => {
    const h = harness(agentNode);
    h.send(0, 'tasks', { type: 'task.start', taskId: 't1', task: 'x' });
    h.send(10, 'tasks', { type: 'task.cancel', taskId: 't1' });
    expect(h.ios[0].closed).toBe(true);
    stream(h.ios[0], 'late');
    expect(h.of('agent.done')).toHaveLength(0);
    expect(h.of('agent.cancelled')).toHaveLength(1);
  });
});

describe('wiring', () => {
  it('gives the LLM delegation only while the agent stage is on', () => {
    const c = defaultConfig();
    const llm = (g: ReturnType<typeof buildGraph>) => g.nodes.find((n) => n.id === 'llm')!.params.delegate;
    expect(llm(buildGraph(c))).toMatchObject({ progress: 'context', quietMs: 800 });
    expect(buildGraph(c).edges).toContainEqual({ from: { node: 'llm', port: 'tasks' }, to: { node: 'agent', port: 'tasks' } });
    c.agent = { variant: 'off', params: {} };
    expect(llm(buildGraph(c))).toBeNull();
  });

  it('lets the agent take the floor unprompted', () => {
    expect(FLOOR.listening.playback_start).toBe('speaking');
  });
});
