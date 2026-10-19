import type { NodeDef } from '../core/node';
import type { Json, JsonObject } from '../core/types';
import { anthropicAdapter, thinkingOff } from './llm/anthropic';
import { DEFAULT_DELEGATE_PROMPT } from './llm/delegate';
import type { LlmStreamParser } from './llm/types';

export const DEFAULT_AGENT_SYSTEM = `You are the background agent of a voice assistant. The voice assistant talks with a user in real time and hands you tasks it cannot answer on the spot. Work the task out properly: reason, search and read the web when it helps, and check what you find.

Your final message is read by the voice assistant, which tells the user. Make it short and spoken-friendly: the answer first, then only the details that matter, no markdown or links unless asked for.

When a step takes a while or turns up something worth knowing, call report_progress with a one-line note; the assistant may pass it on. Don't use it for the final answer.`;

const PROGRESS_TOOL: JsonObject = {
  name: 'report_progress',
  description: 'Send the voice assistant a one-line progress note: what you found or are doing now. Not for the final answer.',
  input_schema: { type: 'object', properties: { note: { type: 'string' } }, required: ['note'], additionalProperties: false },
};

type Io = { close(reason?: Json): void; readonly closed: boolean };

interface Task {
  id: string;
  messages: JsonObject[];
  status: 'running' | 'done' | 'cancelled' | 'error';
  /** Steering messages that arrived while a request was out: they go with the next one. */
  steers: string[];
  rounds: number;
  t0: number;
  io?: Io;
  parser?: LlmStreamParser;
}

const textOf = (content: Json) =>
  Array.isArray(content)
    ? (content as JsonObject[])
        .filter((b) => b?.type === 'text')
        .map((b) => String(b.text ?? ''))
        .join('')
        .trim()
    : '';

/**
 * The background agent: a slower model with reasoning and web tools, run on
 * the tasks the conversation LLM delegates (task.start / task.send /
 * task.cancel). Each task is its own conversation, on the same recorded
 * `anthropic` channel as the LLM, so a replay reproduces it exactly. It runs
 * the request loop itself: server tools (web search, web fetch) run inside a
 * request; `report_progress` is a client tool answered at once; `pause_turn`
 * is resumed. Steering waits for the request in flight, then goes with the
 * next one (on a finished task it starts a follow-up). A side question ("ask")
 * is a separate one-off request over the task so far, without tools, that
 * leaves the task alone.
 *
 * Emits agent.started, agent.round, agent.thinking, agent.step (each request's
 * text and tool calls), agent.progress, agent.answer, agent.done, agent.error
 * and agent.cancelled on `agent`, which goes back to the LLM.
 */
export const agentNode: NodeDef = {
  type: 'agent',
  title: 'Background agent',
  category: 'llm',
  description: 'A slower model with reasoning and web tools that works on tasks the conversation LLM delegates, while the conversation goes on.',
  inputs: [{ name: 'tasks', type: 'tasks', description: 'task.start / task.send / task.cancel from the LLM' }],
  outputs: [{ name: 'agent', type: 'agent', description: 'Progress, answers and results, for the LLM' }],
  params: [
    { key: 'model', label: 'Model', type: 'string', default: 'claude-sonnet-5-5' },
    { key: 'effort', label: 'Effort', type: 'select', options: ['low', 'medium', 'high', 'xhigh', 'max'], default: 'medium' },
    {
      key: 'thinking',
      label: 'Reasoning',
      type: 'select',
      options: ['summarized', 'omitted', 'off'],
      default: 'summarized',
      description: 'Adaptive thinking; summarized returns a readable summary (shown on the lane), omitted returns none. Off: between_tools (Sonnet) or disabled (Haiku).',
    },
    { key: 'webSearch', label: 'Web search', type: 'boolean', default: true },
    { key: 'webFetch', label: 'Web fetch', type: 'boolean', default: true },
    { key: 'maxTokens', label: 'Max tokens', type: 'number', default: 32000 },
    { key: 'maxRounds', label: 'Max requests per task', type: 'number', default: 12 },
    { key: 'fallbacks', label: 'Server-side refusal fallback', type: 'boolean', default: true },
    { key: 'system', label: 'System prompt', type: 'text', default: DEFAULT_AGENT_SYSTEM },
    { key: 'askEffort', label: 'Side question effort', type: 'select', options: ['low', 'medium', 'high'], default: 'low' },
    {
      key: 'delegatePrompt',
      label: 'Conversation LLM: delegation prompt',
      type: 'text',
      default: DEFAULT_DELEGATE_PROMPT,
      description: "Added to the conversation LLM's system prompt along with the task tools.",
    },
    {
      key: 'progress',
      label: 'Conversation LLM: progress notes',
      type: 'select',
      options: ['context', 'speak', 'off'],
      default: 'context',
      description: 'context: added to the next request, so the LLM knows if asked. speak: the LLM replies to each one. Results and answers are always spoken.',
    },
    {
      key: 'quietMs',
      label: 'Conversation LLM: wait for a pause (ms)',
      type: 'number',
      default: 800,
      description: 'A result is spoken once the floor is free and the user has been quiet this long.',
    },
  ],
  channels: [{ name: 'anthropic', description: 'Messages API stream (raw SDK stream events)' }],
  requires: () => ['anthropic'],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const tasks = new Map<string, Task>();
    const tools: JsonObject[] = [
      ...(p.webSearch ? [{ type: 'web_search_20260209', name: 'web_search' }] : []),
      ...(p.webFetch ? [{ type: 'web_fetch_20260209', name: 'web_fetch' }] : []),
    ];
    const thinking = (): JsonObject =>
      p.thinking === 'off' ? thinkingOff({ thinking: 'off', model: p.model }) : { thinking: { type: 'adaptive', display: p.thinking === 'omitted' ? 'omitted' : 'summarized' } };
    const base = (): JsonObject => ({
      model: p.model,
      max_tokens: p.maxTokens,
      ...(p.fallbacks ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' } : {}),
    });

    const fail = (t: Task, error: Json) => {
      t.status = 'error';
      ctx.emit('agent', { type: 'agent.error', taskId: t.id, error, text: typeof error === 'string' ? error : JSON.stringify(error) });
      ctx.log('error', `task ${t.id} failed`, error);
    };

    /** One request of the task's loop; what follows depends on why it stopped. */
    const run = (t: Task) => {
      if (t.rounds >= Number(p.maxRounds)) return fail(t, `stopped after ${t.rounds} requests`);
      const n = ++t.rounds;
      const request: Json = {
        ...base(),
        system: p.system,
        // A copy: the recorded request must not grow with the task.
        messages: [...t.messages] as Json,
        tools: [...tools, PROGRESS_TOOL] as Json,
        output_config: { effort: p.effort },
        ...thinking(),
      };
      const parser = anthropicAdapter.parser();
      t.parser = parser;
      let thought = false;
      ctx.emit('agent', { type: 'agent.round', taskId: t.id, n });
      const onEvents = (events: ReturnType<LlmStreamParser['recv']>) => {
        if (t.parser !== parser || t.status !== 'running') return;
        for (const ev of events) {
          if (ev.type === 'thinking' && !thought) {
            thought = true;
            ctx.emit('agent', { type: 'agent.thinking', taskId: t.id, n });
          } else if (ev.type === 'error') {
            t.parser = undefined;
            return fail(t, ev.error);
          } else if (ev.type === 'end') {
            t.parser = undefined;
            return stopped(t, n, parser.native(), ev.stopReason);
          }
        }
      };
      t.io = ctx.io.open('anthropic', request, { onRecv: (d) => onEvents(parser.recv(d)), onClose: (d) => onEvents(parser.close(d)) });
    };

    const stopped = (t: Task, n: number, content: Json, stopReason: string | null) => {
      const blocks = (Array.isArray(content) ? content : []) as JsonObject[];
      t.messages.push({ role: 'assistant', content: blocks as Json });
      const calls = blocks.filter((b) => b.type === 'tool_use' || b.type === 'server_tool_use').map((b) => ({ name: b.name as string, input: b.input ?? null }));
      const thoughts = blocks
        .filter((b) => b.type === 'thinking')
        .map((b) => String(b.thinking ?? ''))
        .join('\n')
        .trim();
      ctx.emit('agent', { type: 'agent.step', taskId: t.id, n, stopReason, text: textOf(blocks as Json), tools: calls as Json, ...(thoughts ? { thinking: thoughts } : {}) });
      // The server paused its own tool loop: send the conversation back as it is to resume.
      if (stopReason === 'pause_turn') return run(t);
      const steers = t.steers.splice(0).map((s) => ({ type: 'text', text: `Message from the voice assistant: ${s}` }));
      if (stopReason === 'tool_use') {
        const results = blocks
          .filter((b) => b.type === 'tool_use')
          .map((b) => {
            const known = b.name === PROGRESS_TOOL.name;
            if (known) ctx.emit('agent', { type: 'agent.progress', taskId: t.id, text: String((b.input as JsonObject | undefined)?.note ?? '') });
            return { type: 'tool_result', tool_use_id: b.id, content: known ? 'Sent.' : `Unknown tool "${String(b.name)}".`, ...(known ? {} : { is_error: true }) };
          });
        t.messages.push({ role: 'user', content: [...results, ...steers] as Json });
        return run(t);
      }
      if (stopReason === 'refusal') return fail(t, 'the model declined the task');
      if (steers.length) {
        t.messages.push({ role: 'user', content: steers as Json });
        return run(t);
      }
      t.status = 'done';
      ctx.emit('agent', {
        type: 'agent.done',
        taskId: t.id,
        text: textOf(blocks as Json) || '(no answer)',
        stopReason,
        rounds: t.rounds,
        ms: Math.round(ctx.now() - t.t0),
      });
    };

    /** A side question over the task so far: one request, no tools used, the task untouched. */
    const ask = (t: Task, question: string) => {
      const q = { type: 'text', text: `Side question from the voice assistant (don't change course, and answer from what you know so far in a sentence or two, without tools): ${question}` };
      const messages = t.messages.map((m) => ({ ...m }));
      const last = messages[messages.length - 1];
      if (last?.role === 'user') last.content = [...(Array.isArray(last.content) ? last.content : []), q];
      else messages.push({ role: 'user', content: [q] });
      const parser = anthropicAdapter.parser();
      let done = false;
      const t0 = ctx.now();
      const onEvents = (events: ReturnType<LlmStreamParser['recv']>) => {
        for (const ev of events) {
          if (done) return;
          if (ev.type === 'end' || ev.type === 'error') {
            done = true;
            const text = ev.type === 'end' ? textOf(parser.native()) : '';
            ctx.emit('agent', { type: 'agent.answer', taskId: t.id, question, text: text || '(no answer)', ms: Math.round(ctx.now() - t0), ...(ev.type === 'error' ? { error: ev.error } : {}) });
          }
        }
      };
      ctx.io.open(
        'anthropic',
        { ...base(), system: p.system, messages: messages as Json, tools: [...tools, PROGRESS_TOOL] as Json, tool_choice: { type: 'none' }, output_config: { effort: p.askEffort } },
        { onRecv: (d) => onEvents(parser.recv(d)), onClose: (d) => onEvents(parser.close(d)) },
      );
    };

    const stopTask = (t: Task) => {
      t.parser = undefined;
      if (t.io && !t.io.closed) t.io.close({ status: 'cancelled' });
    };

    return {
      onInput(_port, msg) {
        const id = msg.taskId as string;
        const t = tasks.get(id);
        if (msg.type === 'task.start') {
          const task: Task = { id, messages: [{ role: 'user', content: [{ type: 'text', text: String(msg.task ?? '') }] }], status: 'running', steers: [], rounds: 0, t0: ctx.now() };
          tasks.set(id, task);
          ctx.emit('agent', { type: 'agent.started', taskId: id, task: String(msg.task ?? '') });
          ctx.setState('running', [...tasks.values()].filter((x) => x.status === 'running').length);
          return run(task);
        }
        if (!t) return void ctx.log('warn', `no task ${id}`);
        if (msg.type === 'task.cancel') {
          if (t.status !== 'running') return;
          stopTask(t);
          t.status = 'cancelled';
          ctx.emit('agent', { type: 'agent.cancelled', taskId: id });
        } else if (msg.type === 'task.send') {
          const text = String(msg.message ?? '');
          if (msg.mode === 'ask') return ask(t, text);
          if (t.status === 'running') {
            t.steers.push(text);
            ctx.emit('agent', { type: 'agent.steer', taskId: id, text, queued: true });
          } else {
            // A finished task picks up again: a follow-up in the same conversation.
            t.messages.push({ role: 'user', content: [{ type: 'text', text: `Message from the voice assistant: ${text}` }] });
            t.status = 'running';
            t.t0 = ctx.now();
            ctx.emit('agent', { type: 'agent.steer', taskId: id, text, queued: false });
            run(t);
          }
        }
        ctx.setState('running', [...tasks.values()].filter((x) => x.status === 'running').length);
      },
      stop() {
        for (const t of tasks.values()) stopTask(t);
      },
    };
  },
};

export const noAgentNode: NodeDef<Record<string, never>> = {
  type: 'agent.off',
  title: 'No background agent',
  category: 'llm',
  description: 'The conversation LLM answers everything itself, with no task tools.',
  inputs: agentNode.inputs,
  outputs: agentNode.outputs,
  params: [],
  create() {
    return {};
  },
};
