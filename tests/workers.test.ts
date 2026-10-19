import { describe, expect, it } from 'vitest';
import { llmNode, cutToHeard } from '../src/nodes/llm';
import { mockWorkersNode, workersNode } from '../src/nodes/workers';
import { FLOOR, halfDuplexTurnNode } from '../src/nodes/turn';
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
  const ios: { request: JsonObject; h: IOHandlers; closed: boolean; sent: JsonObject[] }[] = [];
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
        const x = { request: request as JsonObject, h, closed: false, sent: [] as JsonObject[] };
        ios.push(x);
        return { xid: String(ios.length), send: (d: Json) => void x.sent.push(d as JsonObject), close: () => void (x.closed = true), get closed() { return x.closed; } };
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

const delegate = { prompt: DEFAULT_DELEGATE_PROMPT, progress: 'context', quietMs: 800, maxWorkers: 2 };
const turn = (turnId: string, text: string) => ({ type: 'turn.user', turnId, text });
const floor = (to: string) => ({ type: 'turn.transition', to });

describe('the conversation LLM delegates', () => {
  it('starts a worker with a tool call and continues the same reply with the result', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'What is the weather in Kyoto?'));
    const req = h.ios[0].request;
    expect((req.tools as JsonObject[]).map((t) => t.name)).toEqual(['spawn_worker', 'message_worker', 'check_worker', 'stop_worker']);
    expect(String(req.system)).toContain('worker_update');
    stream(h.ios[0], 'Let me look that up. ', [{ type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: { task: 'Weather in Kyoto today' } }]);
    expect(h.of('worker.start')).toMatchObject([{ workerId: 'w1', task: 'Weather in Kyoto today' }]);
    expect(h.out.find((o) => o.msg.type === 'worker.start')!.port).toBe('workers');
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
    stream(h.ios[0], 'On it. ', [{ type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: { task: 'ramen' } }]);
    stream(h.ios[1], '');
    h.send(100, 'turn', floor('speaking'));
    h.send(2000, 'updates', { type: 'worker.done', workerId: 'w1', text: 'Ichiran in Gion, open until 2 am.' });
    expect(h.of('llm.start')).toHaveLength(1);
    h.send(2400, 'vad', { type: 'vad.speech_start' });
    h.send(2500, 'turn', floor('listening'));
    h.send(3000, 'vad', { type: 'vad.speech_end' });
    h.advance(3700);
    expect(h.of('llm.start')).toHaveLength(1);
    h.advance(3800);
    expect(h.of('llm.start')).toMatchObject([{}, { turnId: null, workers: ['w1'] }]);
    const msgs = h.ios[2].request.messages as JsonObject[];
    const last = msgs[msgs.length - 1].content as JsonObject[];
    expect(String(last[last.length - 1].text)).toMatch(/<worker_update id="w1" kind="result">\nIchiran in Gion/);
  });

  /** A worker w1 started on the first turn, its reply spoken, the floor back with the user. */
  const withWorker = () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Find me a ramen place.'));
    stream(h.ios[0], 'On it. ', [{ type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: { task: 'ramen' } }]);
    stream(h.ios[1], '');
    h.send(100, 'turn', floor('speaking'));
    h.send(200, 'turn', floor('listening'));
    return h;
  };

  it('holds news while turn-taking has words of the user not yet committed', () => {
    const h = withWorker();
    h.send(1000, 'turn', { type: 'turn.pending', words: true });
    h.send(2000, 'updates', { type: 'worker.done', workerId: 'w1', text: 'Ichiran.' });
    h.advance(5000);
    expect(h.of('llm.start')).toHaveLength(1);
    // The turn is committed: the news goes with it, not on its own.
    h.send(5000, 'turn', turn('u1', 'Somewhere quiet please.'));
    expect(h.of('llm.start')).toMatchObject([{}, { turnId: 'u1' }]);
    expect(JSON.stringify(h.ios[2].request.messages)).toContain('Ichiran.');
  });

  it('drops a reply to news the user spoke up before, and tries again after', () => {
    const h = withWorker();
    h.send(2000, 'updates', { type: 'worker.done', workerId: 'w1', text: 'Ichiran.' });
    expect(h.of('llm.start')).toHaveLength(2);
    h.send(2100, 'vad', { type: 'vad.speech_start' });
    expect(h.ios[2].closed).toBe(true);
    expect(h.of('llm.end')[1]).toMatchObject({ status: 'cancelled' });
    // Only a cough: once they are quiet again, the news is told.
    h.send(2300, 'vad', { type: 'vad.speech_end' });
    h.advance(3200);
    expect(h.of('llm.start')).toHaveLength(3);
    const msgs = JSON.stringify(h.ios[3].request.messages);
    expect(msgs.split('Ichiran.').length).toBe(2);
  });

  it('stops the workers of a reply whose turn was replaced', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Find me a ramen'));
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: { task: 'ramen' } }]);
    stream(h.ios[1], 'On it.');
    h.send(500, 'turn', { ...turn('u1', 'Find me a ramen place in Gion.'), replaces: 'u0' });
    expect(h.of('worker.stop')).toMatchObject([{ workerId: 'w1' }]);
    h.send(600, 'updates', { type: 'worker.done', workerId: 'w1', text: 'Ichiran.' });
    expect(JSON.stringify(h.ios.map((io) => io.request.messages))).not.toContain('Ichiran.');
  });

  it('caps follow-ups at the most workers at once', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'a and b and c'));
    stream(h.ios[0], '', ['a', 'b'].map((task, i) => ({ type: 'tool_use', id: `tu${i}`, name: 'spawn_worker', input: { task } })));
    h.send(10, 'updates', { type: 'worker.done', workerId: 'w1', text: 'A' });
    stream(h.ios[1], '', [
      { type: 'tool_use', id: 'tu2', name: 'spawn_worker', input: { task: 'c' } },
      { type: 'tool_use', id: 'tu3', name: 'message_worker', input: { worker_id: 'w1', message: 'more' } },
    ]);
    const results = (h.ios[2].request.messages as JsonObject[]).at(-1)!.content as JsonObject[];
    expect(results.map((r) => !!r.is_error)).toEqual([false, true]);
    expect(h.of('worker.send')).toHaveLength(0);
  });

  it('half-duplex turn-taking tells the LLM about the floor', () => {
    const h = harness(halfDuplexTurnNode);
    h.send(0, 'transcript', { type: 'asr.final', text: 'hi', start: 0, end: 1, speechFinal: true });
    h.send(10, 'playback', { type: 'playback.start' });
    h.send(20, 'playback', { type: 'playback.end' });
    expect(h.of('turn.transition').map((m) => m.to)).toEqual(['thinking', 'speaking', 'listening']);
    expect(h.of('turn.pending')).toMatchObject([{ words: true }]);
  });

  it('keeps progress notes for the next request instead of speaking them', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Plan a trip.'));
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: { task: 'trip' } }]);
    stream(h.ios[1], 'Working on it.');
    h.send(5000, 'updates', { type: 'worker.progress', workerId: 'w1', text: 'Found three hotels.' });
    h.advance(9000);
    expect(h.ios).toHaveLength(2);
    h.send(9000, 'turn', turn('u1', 'How is it going?'));
    const text = JSON.stringify(h.ios[2].request.messages);
    expect(text).toContain('kind=\\"progress\\">\\nFound three hotels.');
    expect(text.indexOf('Found three hotels')).toBeLessThan(text.indexOf('How is it going?'));
  });

  it('answers a bad worker id with an error result', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Stop it.'));
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'tu1', name: 'stop_worker', input: { worker_id: 'w9' } }]);
    expect(h.of('worker.stop')).toHaveLength(0);
    expect(((h.ios[1].request.messages as JsonObject[])[2].content as JsonObject[])[0]).toMatchObject({ is_error: true });
  });

  it('runs workers in parallel, up to the most at once', () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Compare flights and hotels and trains.'));
    stream(h.ios[0], '', ['flights', 'hotels', 'trains'].map((task, i) => ({ type: 'tool_use', id: `tu${i}`, name: 'spawn_worker', input: { task } })));
    expect(h.of('worker.start').map((m) => m.workerId)).toEqual(['w1', 'w2']);
    const results = (h.ios[1].request.messages as JsonObject[])[2].content as JsonObject[];
    expect(results.map((r) => !!r.is_error)).toEqual([false, false, true]);
    // One finishes: room for another.
    h.send(100, 'updates', { type: 'worker.done', workerId: 'w1', text: 'BA 7 at 9 am.' });
    stream(h.ios[1], '', [{ type: 'tool_use', id: 'tu9', name: 'spawn_worker', input: { task: 'trains' } }]);
    expect(h.of('worker.start').map((m) => m.workerId)).toEqual(['w1', 'w2', 'w3']);
  });

  it("answers check_worker at once from the worker's updates", () => {
    const h = harness(llmNode, { delegate });
    h.send(0, 'turn', turn('u0', 'Find a hotel.'));
    stream(h.ios[0], '', [{ type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: { task: 'hotel in Kyoto' } }]);
    stream(h.ios[1], 'On it.');
    h.send(1000, 'updates', { type: 'worker.tool', workerId: 'w1', title: 'Search: kyoto hotels', kind: 'fetch' });
    h.send(2000, 'updates', { type: 'worker.progress', workerId: 'w1', text: '[x] search\n[>] compare prices' });
    h.send(3000, 'turn', turn('u1', 'How is it going?'));
    stream(h.ios[2], '', [{ type: 'tool_use', id: 'tu2', name: 'check_worker', input: { worker_id: 'w1' } }]);
    expect(h.ios).toHaveLength(4);
    const msgs = h.ios[3].request.messages as JsonObject[];
    const status = String(((msgs[msgs.length - 1].content as JsonObject[])[0] as JsonObject).content);
    expect(status).toContain('Worker w1: running, 3 s');
    expect(status).toContain('[>] compare prices');
    expect(status).toContain('- Search: kyoto hotels');
  });

  it('has no tools without workers', () => {
    const h = harness(llmNode, {});
    h.send(0, 'turn', turn('u0', 'Hi'));
    expect(h.ios[0].request.tools).toBeUndefined();
  });

  it('cuts a reply that called a tool to what was heard, keeping the call', () => {
    const history: ChatTurn[] = [
      { role: 'user' as const, text: 'q', turnId: 'u0' },
      { role: 'assistant' as const, text: 'Let me look that up.', rid: 'r0', native: { provider: 'anthropic', content: [{ type: 'text', text: 'Let me look that up.' }, { type: 'tool_use', id: 'tu1', name: 'spawn_worker', input: {} }] } },
      { role: 'user' as const, text: '', native: { provider: 'anthropic', content: [{ type: 'tool_result', tool_use_id: 'tu1', content: 'ok' }] } },
      { role: 'assistant' as const, text: 'It will take a moment.', rid: 'r0' },
    ];
    const cut = cutToHeard(history, [1, 3], 'Let me look');
    expect(cut).toHaveLength(3);
    expect(cut[1].text).toBe('Let me look—');
    expect((cut[1].native!.content as JsonObject[]).map((b) => b.type)).toEqual(['text', 'tool_use']);
  });
});

/** Plays the agent's side of a worker's ACP connection. */
function agent(io: { h: IOHandlers; sent: JsonObject[] }) {
  const recv = (m: Json) => io.h.onRecv?.(m, undefined);
  const req = (method: string) => [...io.sent].reverse().find((m) => m.method === method)!;
  const ok = (method: string, result: Json) => recv({ jsonrpc: '2.0', id: req(method).id!, result });
  const update = (u: JsonObject) => recv({ jsonrpc: '2.0', method: 'session/update', params: { sessionId: 's1', update: u } });
  return { recv, req, ok, update };
}

/** A worker up to its first prompt. */
function started(params: Record<string, unknown> = {}) {
  const h = harness(workersNode, params);
  h.send(0, 'workers', { type: 'worker.start', workerId: 'w1', task: 'Weather in Kyoto' });
  const a = agent(h.ios[0]);
  a.recv({ $open: { cwd: '/work' } });
  a.ok('initialize', { protocolVersion: 1 });
  a.ok('session/new', { sessionId: 's1', configOptions: ['mode', 'model', 'effort'].map((id) => ({ id, name: id, type: 'select', currentValue: 'default', options: [] })) });
  a.ok('session/set_config_option', { configOptions: [] });
  h.advance(400);
  a.ok('session/set_config_option', { configOptions: [] });
  return { h, a };
}

describe('workers', () => {
  it('sets up an ACP session with the model and effort, then prompts the task', () => {
    const { h, a } = started();
    expect(h.ios[0].request).toEqual({ worker: 'w1' });
    expect(a.req('initialize').params).toMatchObject({ protocolVersion: 1, clientCapabilities: { terminal: false } });
    expect(a.req('session/new').params).toMatchObject({ cwd: '/work', mcpServers: [] });
    expect(JSON.stringify(a.req('session/new').params)).toContain('"systemPrompt":{"append":"You are a worker for a voice assistant.')
    expect(h.ios[0].sent.filter((m) => m.method === 'session/set_config_option').map((m) => m.params)).toEqual([
      { sessionId: 's1', configId: 'model', value: 'claude-sonnet-5-5' },
      { sessionId: 's1', configId: 'effort', value: 'medium' },
    ]);
    expect(a.req('session/prompt').params).toEqual({ sessionId: 's1', prompt: [{ type: 'text', text: 'Weather in Kyoto' }] });
    expect(h.of('worker.ready')).toMatchObject([{ workerId: 'w1', sessionId: 's1', ms: 400 }]);
  });

  it('reports thinking, tool calls and the plan, and the text after the last tool call as the result', () => {
    const { h, a } = started();
    a.update({ sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'hmm' } });
    a.update({ sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: ' more' } });
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Let me search.' } });
    a.update({ sessionUpdate: 'tool_call', toolCallId: 'tc1', title: 'Web search: kyoto weather', kind: 'fetch', status: 'pending' });
    a.update({ sessionUpdate: 'plan', entries: [{ content: 'search', status: 'completed' }, { content: 'answer', status: 'in_progress' }] });
    a.update({ sessionUpdate: 'tool_call_update', toolCallId: 'tc1', status: 'completed' });
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Sunny, ' } });
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: '25 degrees.' } });
    h.advance(5000);
    a.ok('session/prompt', { stopReason: 'end_turn' });
    expect(h.of('worker.thinking')).toHaveLength(1);
    expect(h.of('worker.text')).toMatchObject([{ text: 'Let me search.' }]);
    expect(h.of('worker.tool')).toMatchObject([{ toolCallId: 'tc1', title: 'Web search: kyoto weather', kind: 'fetch' }]);
    expect(h.of('worker.tool_done')).toMatchObject([{ toolCallId: 'tc1', status: 'completed' }]);
    expect(h.of('worker.progress')).toMatchObject([{ text: '[x] search\n[>] answer' }]);
    expect(h.of('worker.done')).toMatchObject([{ workerId: 'w1', text: 'Sunny, 25 degrees.', stopReason: 'end_turn', ms: 5000 }]);
  });

  it('answers permission requests by policy', () => {
    const ask = (kind: string) => ({
      jsonrpc: '2.0',
      id: 77,
      method: 'session/request_permission',
      params: { sessionId: 's1', toolCall: { toolCallId: 'x', title: kind, kind }, options: [{ optionId: 'yes', name: 'Allow', kind: 'allow_once' }, { optionId: 'no', name: 'Reject', kind: 'reject_once' }] },
    });
    const answer = (policy: string, kind: string) => {
      const { h, a } = started({ permissions: policy });
      a.recv(ask(kind));
      return (h.ios[0].sent.find((m) => m.id === 77)!.result as JsonObject).outcome;
    };
    expect(answer('read-only', 'fetch')).toEqual({ outcome: 'selected', optionId: 'yes' });
    expect(answer('read-only', 'execute')).toEqual({ outcome: 'selected', optionId: 'no' });
    expect(answer('allow', 'edit')).toEqual({ outcome: 'selected', optionId: 'yes' });
    expect(answer('reject', 'read')).toEqual({ outcome: 'selected', optionId: 'no' });
    // No option of the kind wanted: an error, not a made-up cancel.
    const { h, a } = started({ permissions: 'allow' });
    a.recv({ ...ask('edit'), params: { ...ask('edit').params, options: [{ optionId: 'no', name: 'Reject', kind: 'reject_once' }] } });
    expect(h.ios[0].sent.find((m) => m.id === 77 && m.error)).toBeTruthy();
  });

  it('refuses client methods it did not offer', () => {
    const { h, a } = started();
    a.recv({ jsonrpc: '2.0', id: 5, method: 'fs/read_text_file', params: {} });
    expect(h.ios[0].sent.find((m) => m.id === 5 && m.error)).toMatchObject({ error: { code: -32601 } });
  });

  it('sends a message to a running worker as another prompt, and finishes on the latest one', () => {
    const { h, a } = started();
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Today is' } });
    h.send(1000, 'workers', { type: 'worker.send', workerId: 'w1', message: 'Tomorrow, not today.' });
    // The agent finishes the sentence it was on before it takes the message in.
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: ' sunny.' } });
    const prompts = h.ios[0].sent.filter((m) => m.method === 'session/prompt');
    expect(prompts).toHaveLength(2);
    expect(JSON.stringify(prompts[1].params)).toContain('Message from the voice assistant: Tomorrow, not today.');
    // The first prompt returns when the agent takes in the second: not done yet.
    a.recv({ jsonrpc: '2.0', id: prompts[0].id!, result: { stopReason: 'end_turn' } });
    expect(h.of('worker.done')).toHaveLength(0);
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Tomorrow: rain.' } });
    a.recv({ jsonrpc: '2.0', id: prompts[1].id!, result: { stopReason: 'end_turn' } });
    expect(h.of('worker.done')).toMatchObject([{ text: 'Tomorrow: rain.' }]);
    // A follow-up on a finished worker: same session, running again.
    h.send(2000, 'workers', { type: 'worker.send', workerId: 'w1', message: 'And the day after?' });
    expect(h.of('worker.steer')).toMatchObject([{ followUp: false }, { followUp: true }]);
    const third = h.ios[0].sent.filter((m) => m.method === 'session/prompt');
    expect(third).toHaveLength(3);
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Thursday: clear.' } });
    a.recv({ jsonrpc: '2.0', id: third[2].id!, result: { stopReason: 'end_turn' } });
    expect(h.of('worker.done')).toMatchObject([{ text: 'Tomorrow: rain.' }, { text: 'Thursday: clear.' }]);
  });

  it('keeps a finished result when its agent exits, and says so if messaged afterwards', () => {
    const { h, a } = started();
    a.update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Done.' } });
    a.ok('session/prompt', { stopReason: 'end_turn' });
    h.ios[0].h.onClose?.({ code: 1011, error: 'agent exited (1)' });
    expect(h.of('worker.error')).toHaveLength(0);
    h.send(100, 'workers', { type: 'worker.send', workerId: 'w1', message: 'more' });
    expect(h.of('worker.error')).toMatchObject([{ text: expect.stringContaining('session has ended') }]);
  });

  it('keeps at most maxWorkers finished agents around, closing the oldest', () => {
    const h = harness(workersNode, { maxWorkers: 1, model: '', effort: '' });
    for (const id of ['w1', 'w2']) {
      h.send(0, 'workers', { type: 'worker.start', workerId: id, task: id });
      const io = h.ios[h.ios.length - 1];
      const a = agent(io);
      a.ok('initialize', {});
      a.ok('session/new', { sessionId: id });
    }
    for (const io of h.ios) agent(io).ok('session/prompt', { stopReason: 'end_turn' });
    expect(h.ios.map((io) => io.closed)).toEqual([true, false]);
  });

  it('fails on a prompt error or a refusal, and stops cleanly while starting', () => {
    const { h, a } = started();
    a.recv({ jsonrpc: '2.0', id: a.req('session/prompt').id!, error: { code: -32603, message: 'Internal error' } });
    expect(h.of('worker.error')).toMatchObject([{ text: 'Internal error' }]);
    const r = started();
    r.a.ok('session/prompt', { stopReason: 'refusal' });
    expect(r.h.of('worker.error')).toMatchObject([{ text: 'the agent declined the task' }]);
    const s = harness(workersNode);
    s.send(0, 'workers', { type: 'worker.start', workerId: 'w1', task: 'x' });
    s.send(5, 'workers', { type: 'worker.stop', workerId: 'w1' });
    agent(s.ios[0]).ok('initialize', {});
    expect(s.ios[0].sent.map((m) => m.method)).toEqual(['initialize']);
    expect(s.of('worker.stopped')).toHaveLength(1);
  });

  it('holds messages that come before the session is ready', () => {
    const h = harness(workersNode, { model: '' });
    h.send(0, 'workers', { type: 'worker.start', workerId: 'w1', task: 'x' });
    h.send(10, 'workers', { type: 'worker.send', workerId: 'w1', message: 'also y' });
    const a = agent(h.ios[0]);
    a.ok('initialize', {});
    a.ok('session/new', { sessionId: 's1' });
    expect(h.ios[0].sent.filter((m) => m.method === 'session/set_config_option')).toHaveLength(0);
    expect(h.ios[0].sent.filter((m) => m.method === 'session/prompt').map((m) => JSON.stringify(m.params))).toEqual([
      expect.stringContaining('"text":"x"'),
      expect.stringContaining('also y'),
    ]);
  });

  it('cancels and ends the agent on stop', () => {
    const { h, a } = started();
    h.send(10, 'workers', { type: 'worker.stop', workerId: 'w1' });
    expect(h.ios[0].sent[h.ios[0].sent.length - 1]).toMatchObject({ method: 'session/cancel', params: { sessionId: 's1' } });
    expect(h.ios[0].closed).toBe(true);
    a.ok('session/prompt', { stopReason: 'cancelled' });
    expect(h.of('worker.done')).toHaveLength(0);
    expect(h.of('worker.stopped')).toHaveLength(1);
  });

  it('fails when the agent process ends, with its last stderr', () => {
    const { h, a } = started();
    a.recv({ $stderr: 'Invalid API key' });
    h.ios[0].h.onClose?.({ code: 1011, reason: 'agent exited (1)', error: 'agent exited (1)' });
    expect(h.of('worker.error')).toMatchObject([{ text: expect.stringContaining('Invalid API key') }]);
  });
});

describe('mock workers', () => {
  it('return a dummy result after a seeded random delay, the same every run', () => {
    const run = (seed: number) => {
      const h = harness(mockWorkersNode, { minMs: 1000, maxMs: 5000, seed });
      h.send(0, 'workers', { type: 'worker.start', workerId: 'w1', task: 'weather' });
      h.send(0, 'workers', { type: 'worker.start', workerId: 'w2', task: 'trains' });
      h.send(500, 'workers', { type: 'worker.send', workerId: 'w2', message: 'tomorrow' });
      h.advance(10_000);
      return h;
    };
    const h = run(0);
    const done = h.of('worker.done');
    expect(done.map((d) => d.text)).toEqual(expect.arrayContaining(['Mock result for "weather".', 'Mock result for "trains". Noted: "tomorrow".']));
    for (const d of done) expect(Number(d.ms)).toBeGreaterThanOrEqual(1000);
    for (const d of done) expect(Number(d.ms)).toBeLessThanOrEqual(5000);
    expect(h.of('worker.ready')).toHaveLength(2);
    expect(h.of('worker.tool')).toHaveLength(2);
    expect(run(0).of('worker.done').map((d) => d.ms)).toEqual(done.map((d) => d.ms));
    expect(run(1).of('worker.done').map((d) => d.ms)).not.toEqual(done.map((d) => d.ms));
  });

  it('stop without a result, and pick up again on a follow-up', () => {
    const h = harness(mockWorkersNode, { minMs: 100, maxMs: 200 });
    h.send(0, 'workers', { type: 'worker.start', workerId: 'w1', task: 'a' });
    h.send(0, 'workers', { type: 'worker.start', workerId: 'w2', task: 'b' });
    h.send(10, 'workers', { type: 'worker.stop', workerId: 'w1' });
    h.advance(1000);
    expect(h.of('worker.done').map((d) => d.workerId)).toEqual(['w2']);
    h.send(1000, 'workers', { type: 'worker.send', workerId: 'w2', message: 'more' });
    h.advance(2000);
    expect(h.of('worker.done').map((d) => d.workerId)).toEqual(['w2', 'w2']);
  });
});

describe('wiring', () => {
  it('gives the LLM delegation only while the workers stage is on', () => {
    const c = defaultConfig();
    const llm = (g: ReturnType<typeof buildGraph>) => g.nodes.find((n) => n.id === 'llm')!.params.delegate;
    expect(llm(buildGraph(c))).toMatchObject({ progress: 'context', quietMs: 800, maxWorkers: 4 });
    expect(buildGraph(c).edges).toContainEqual({ from: { node: 'llm', port: 'workers' }, to: { node: 'workers', port: 'workers' } });
    c.workers = { variant: 'mock', params: {} };
    expect(llm(buildGraph(c))).toMatchObject({ maxWorkers: 4 });
    expect(buildGraph(c).nodes.find((n) => n.id === 'workers')!.type).toBe('workers.mock');
    c.workers = { variant: 'off', params: {} };
    expect(llm(buildGraph(c))).toBeNull();
  });

  it('lets the agent take the floor unprompted', () => {
    expect(FLOOR.listening.playback_start).toBe('speaking');
  });
});
