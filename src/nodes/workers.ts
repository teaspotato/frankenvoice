import type { NodeDef } from '../core/node';
import type { Json, JsonObject } from '../core/types';
import { DEFAULT_DELEGATE_PROMPT } from './llm/delegate';

export const DEFAULT_WORKER_SYSTEM = `You are a worker for a voice assistant. The voice assistant talks with a user in real time and hands you tasks it cannot answer on the spot. Work the task out properly: reason, search and read the web when it helps, and check what you find.

Your final message is read by the voice assistant, which tells the user. Make it short and spoken-friendly: the answer first, then only the details that matter, no markdown or links unless asked for. Messages that start with "Message from the voice assistant:" change or add to your task.`;

/** ACP protocol version this client speaks. */
const PROTOCOL_VERSION = 1;
/** Tool kinds the read-only policy allows. */
const READ_KINDS = new Set(['read', 'search', 'fetch', 'think']);

type Io = { send(data: Json): void; close(reason?: Json): void; readonly closed: boolean };

interface Worker {
  id: string;
  task: string;
  status: 'starting' | 'running' | 'done' | 'stopped' | 'error';
  io: Io;
  t0: number;
  rpc: number;
  /** Our requests waiting for a response, by JSON-RPC id. */
  pending: Map<number, (res: JsonObject) => void>;
  sessionId?: string;
  cwd?: string;
  /** The latest session/prompt: the worker is done when this one returns. */
  prompt?: number;
  /** Messages that came before the session was ready. */
  queued: string[];
  /** The agent's text since its last tool call or prompt: the last of it is the answer. */
  text: string;
  thinking: boolean;
  /** The agent's last stderr lines, for errors. */
  stderr: string[];
}

const chunkText = (u: JsonObject) => {
  const c = u.content as JsonObject | undefined;
  return c?.type === 'text' ? String(c.text ?? '') : '';
};

/** A plan update, as one line per entry. */
const planText = (entries: Json) =>
  (Array.isArray(entries) ? (entries as JsonObject[]) : [])
    .map((e) => `${e.status === 'completed' ? '[x]' : e.status === 'in_progress' ? '[>]' : '[ ]'} ${String(e.content ?? '')}`)
    .join('\n');

/**
 * Workers: each task the conversation LLM delegates (worker.start) runs as its
 * own session on a real agent harness, spoken to over the Agent Client
 * Protocol (JSON-RPC). The local proxy runs one agent process per worker
 * (Claude Code through its ACP adapter by default) and passes the JSON-RPC
 * lines through; every line is recorded on the `acp` channel, so a replay
 * reproduces each worker without running an agent.
 *
 * Per worker: initialize, session/new (with the system prompt in `_meta`, for
 * Claude Code), session/set_config_option for the model and effort, then
 * session/prompt with the task. A message for a running worker is another
 * session/prompt, which the agent takes in mid-turn; for a finished one it
 * starts a follow-up. worker.stop sends session/cancel and ends the process.
 * Permission requests are answered by a policy, since nobody is there to ask.
 *
 * Emits worker.started, worker.ready, worker.thinking, worker.text,
 * worker.tool, worker.tool_done, worker.permission, worker.progress (the
 * agent's plan), worker.steer, worker.done, worker.error and worker.stopped
 * on `updates`, which goes back to the LLM.
 */
export const workersNode: NodeDef = {
  type: 'workers',
  title: 'Workers',
  category: 'llm',
  description: 'Agent sessions (Claude Code over ACP by default) that work on the tasks the conversation LLM delegates, several at once, while the conversation goes on.',
  inputs: [{ name: 'workers', type: 'workers', description: 'worker.start / worker.send / worker.stop from the LLM' }],
  outputs: [{ name: 'updates', type: 'workers', description: 'Progress and results, for the LLM' }],
  params: [
    {
      key: 'model',
      label: 'Model',
      type: 'string',
      default: 'claude-sonnet-5-5',
      description: "Set with the agent's model option after each session starts; Claude Code also takes aliases like sonnet. Empty: the agent's own default.",
    },
    {
      key: 'effort',
      label: 'Effort',
      type: 'select',
      options: ['', 'low', 'medium', 'high', 'xhigh', 'max'],
      default: 'medium',
      description: "Set with the agent's effort option, when it has one. Empty: the agent's own default.",
    },
    {
      key: 'permissions',
      label: 'Permission requests',
      type: 'select',
      options: ['read-only', 'allow', 'reject'],
      default: 'read-only',
      description:
        "Nobody is there to answer the agent's permission requests. read-only: allow reading, searching and fetching, reject edits and commands. allow: allow everything. reject: reject everything. Only what the agent asks about: Claude Code's own settings (~/.claude) can allow tools without asking.",
    },
    { key: 'maxWorkers', label: 'Most workers at once', type: 'number', default: 4 },
    { key: 'system', label: 'Added to the system prompt', type: 'text', default: DEFAULT_WORKER_SYSTEM },
    {
      key: 'delegatePrompt',
      label: 'Conversation LLM: delegation prompt',
      type: 'text',
      default: DEFAULT_DELEGATE_PROMPT,
      description: "Added to the conversation LLM's system prompt along with the worker tools.",
    },
    {
      key: 'progress',
      label: 'Conversation LLM: progress notes',
      type: 'select',
      options: ['context', 'speak', 'off'],
      default: 'context',
      description: "A worker's plan updates. context: added to the next request, so the LLM knows if asked. speak: the LLM replies to each one. Results are always spoken.",
    },
    {
      key: 'quietMs',
      label: 'Conversation LLM: wait for a pause (ms)',
      type: 'number',
      default: 800,
      description: 'A result is spoken once the floor is free and the user has been quiet this long.',
    },
  ],
  channels: [{ name: 'acp', description: 'One agent process per worker: ACP JSON-RPC messages, both ways' }],
  requires: () => ['acp'],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const workers = new Map<string, Worker>();
    const emit = (w: Worker, type: string, data: JsonObject = {}) => ctx.emit('updates', { type, workerId: w.id, ...data });
    const count = () => ctx.setState('running', [...workers.values()].filter((w) => w.status === 'starting' || w.status === 'running').length);

    const call = (w: Worker, method: string, params: JsonObject, then: (res: JsonObject) => void) => {
      const id = ++w.rpc;
      w.pending.set(id, then);
      w.io.send({ jsonrpc: '2.0', id, method, params });
    };
    const respond = (w: Worker, id: Json, body: JsonObject) => w.io.send({ jsonrpc: '2.0', id, ...body });

    const fail = (w: Worker, error: Json) => {
      if (w.status === 'stopped' || w.status === 'error') return;
      w.status = 'error';
      const text = typeof error === 'string' ? error : JSON.stringify(error);
      emit(w, 'worker.error', { error, text });
      ctx.log('error', `worker ${w.id} failed`, error);
      if (!w.io.closed) w.io.close({ status: 'error' });
      count();
    };
    /** A JSON-RPC response's error, if any, as text. */
    const rpcError = (res: JsonObject) => {
      const e = res.error as JsonObject | undefined;
      return e ? `${String(e.message ?? 'error')}${e.data ? `: ${JSON.stringify(e.data)}` : ''}` : null;
    };

    const prompt = (w: Worker, text: string) => {
      w.text = '';
      w.thinking = false;
      const id = w.rpc + 1;
      w.prompt = id;
      call(w, 'session/prompt', { sessionId: w.sessionId!, prompt: [{ type: 'text', text }] }, (res) => {
        // An earlier prompt returns when the agent takes in the next one: only the latest one ends the work.
        if (w.prompt !== id || w.status !== 'running') return;
        const err = rpcError(res);
        if (err) return fail(w, err);
        const stopReason = String((res.result as JsonObject | undefined)?.stopReason ?? '');
        if (stopReason === 'refusal') return fail(w, 'the agent declined the task');
        w.status = 'done';
        emit(w, 'worker.done', { text: w.text.trim() || '(no answer)', stopReason, ms: Math.round(ctx.now() - w.t0) });
        count();
      });
    };

    const ready = (w: Worker) => {
      w.status = 'running';
      emit(w, 'worker.ready', { sessionId: w.sessionId!, ms: Math.round(ctx.now() - w.t0) });
      prompt(w, w.task);
      for (const m of w.queued.splice(0)) prompt(w, `Message from the voice assistant: ${m}`);
    };

    /** Set the session's config options we have values for, one after another, then start. */
    const configure = (w: Worker, offered: Set<string>) => {
      const want = (['model', 'effort'] as const).filter((k) => p[k]);
      for (const k of want) if (!offered.has(k)) ctx.log('warn', `the agent has no ${k} option: worker ${w.id} runs on its default`);
      const next = (rest: string[]) => {
        if (w.status !== 'starting') return;
        const [k, ...more] = rest;
        if (!k) return ready(w);
        call(w, 'session/set_config_option', { sessionId: w.sessionId!, configId: k, value: String(p[k]) }, (res) => {
          const err = rpcError(res);
          if (err) ctx.log('warn', `could not set ${k} to ${String(p[k])}: ${err}`);
          next(more);
        });
      };
      next(want.filter((k) => offered.has(k)));
    };

    const newSession = (w: Worker) => {
      const meta: JsonObject = p.system ? { systemPrompt: { append: String(p.system) } } : {};
      call(w, 'session/new', { cwd: w.cwd ?? '', mcpServers: [], _meta: meta }, (res) => {
        const err = rpcError(res);
        if (err) return fail(w, err);
        const r = res.result as JsonObject;
        w.sessionId = String(r.sessionId);
        const options = (Array.isArray(r.configOptions) ? r.configOptions : []) as JsonObject[];
        configure(w, new Set(options.map((o) => String(o.id))));
      });
    };

    /** The agent asks to run a tool: answer for the user, by policy. */
    const permission = (w: Worker, id: Json, params: JsonObject) => {
      const tool = (params.toolCall ?? {}) as JsonObject;
      const options = (Array.isArray(params.options) ? params.options : []) as JsonObject[];
      const kind = String(tool.kind ?? 'other');
      const allow = p.permissions === 'allow' || (p.permissions === 'read-only' && READ_KINDS.has(kind));
      const pick = allow ? ['allow_once', 'allow_always'] : ['reject_once', 'reject_always'];
      const option = pick.map((k) => options.find((o) => o.kind === k)).find(Boolean);
      emit(w, 'worker.permission', { title: String(tool.title ?? ''), kind, allowed: allow });
      respond(w, id, { result: { outcome: option ? { outcome: 'selected', optionId: option.optionId } : { outcome: 'cancelled' } } });
    };

    const update = (w: Worker, u: JsonObject) => {
      switch (u.sessionUpdate) {
        case 'agent_message_chunk':
          w.text += chunkText(u);
          w.thinking = false;
          break;
        case 'agent_thought_chunk':
          if (!w.thinking) emit(w, 'worker.thinking');
          w.thinking = true;
          break;
        case 'user_message_chunk':
          // The agent took in our latest message: what it says next answers that.
          w.text = '';
          break;
        case 'tool_call':
          if (w.text.trim()) emit(w, 'worker.text', { text: w.text.trim() });
          w.text = '';
          w.thinking = false;
          emit(w, 'worker.tool', { toolCallId: String(u.toolCallId), title: String(u.title ?? ''), kind: String(u.kind ?? 'other'), status: String(u.status ?? 'pending') });
          break;
        case 'tool_call_update':
          if (u.status === 'completed' || u.status === 'failed') emit(w, 'worker.tool_done', { toolCallId: String(u.toolCallId), status: u.status as string });
          break;
        case 'plan':
          emit(w, 'worker.progress', { text: planText(u.entries ?? null) });
          break;
      }
    };

    const recv = (w: Worker, msg: JsonObject) => {
      if (w.status === 'stopped' || w.status === 'error') return;
      if (msg.$open) {
        w.cwd = String((msg.$open as JsonObject).cwd ?? '');
        return;
      }
      if (msg.$stderr !== undefined) {
        w.stderr = [...w.stderr, String(msg.$stderr)].slice(-5);
        return;
      }
      if (msg.method === undefined && typeof msg.id === 'number') {
        const then = w.pending.get(msg.id);
        w.pending.delete(msg.id);
        return then?.(msg);
      }
      if (msg.method === 'session/update') return update(w, ((msg.params as JsonObject)?.update ?? {}) as JsonObject);
      if (msg.id !== undefined) {
        if (msg.method === 'session/request_permission') return permission(w, msg.id, (msg.params ?? {}) as JsonObject);
        // fs/* and terminal/*: not offered in our capabilities.
        respond(w, msg.id, { error: { code: -32601, message: `${String(msg.method)} is not supported by this client` } });
      }
    };

    const start = (id: string, task: string) => {
      const w: Worker = { id, task, status: 'starting', t0: ctx.now(), rpc: 0, pending: new Map(), queued: [], text: '', thinking: false, stderr: [], io: undefined as never };
      workers.set(id, w);
      emit(w, 'worker.started', { task });
      w.io = ctx.io.open(
        'acp',
        { worker: id },
        {
          onRecv: (d) => recv(w, (d ?? {}) as JsonObject),
          onClose: (d) => {
            if (w.status === 'stopped' || w.status === 'error') return;
            const why = (d as JsonObject | null)?.error ?? (d as JsonObject | null)?.reason;
            fail(w, `the agent process ended${why ? ` (${String(why)})` : ''}${w.stderr.length ? `: ${w.stderr.join(' | ')}` : ''}`);
          },
        },
      );
      call(w, 'initialize', { protocolVersion: PROTOCOL_VERSION, clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false }, clientInfo: { name: 'frankenvoice', version: '0' } }, (res) => {
        const err = rpcError(res);
        if (err) return fail(w, err);
        newSession(w);
      });
      count();
    };

    const stop = (w: Worker) => {
      if (w.sessionId && !w.io.closed) w.io.send({ jsonrpc: '2.0', method: 'session/cancel', params: { sessionId: w.sessionId } });
      if (!w.io.closed) w.io.close({ status: 'stopped' });
    };

    return {
      onInput(_port, msg) {
        const id = msg.workerId as string;
        if (msg.type === 'worker.start') return start(id, String(msg.task ?? ''));
        const w = workers.get(id);
        if (!w) return void ctx.log('warn', `no worker ${id}`);
        if (msg.type === 'worker.stop') {
          if (w.status === 'stopped' || w.status === 'error') return;
          stop(w);
          w.status = 'stopped';
          emit(w, 'worker.stopped');
        } else if (msg.type === 'worker.send') {
          const text = String(msg.message ?? '');
          if (w.status === 'stopped' || w.status === 'error') return void ctx.log('warn', `worker ${id} is ${w.status}`);
          const followUp = w.status === 'done';
          emit(w, 'worker.steer', { text, followUp });
          if (w.status === 'starting') return void w.queued.push(text);
          if (followUp) {
            w.status = 'running';
            w.t0 = ctx.now();
          }
          prompt(w, `Message from the voice assistant: ${text}`);
        }
        count();
      },
      stop() {
        for (const w of workers.values()) stop(w);
      },
    };
  },
};

export const noWorkersNode: NodeDef<Record<string, never>> = {
  type: 'workers.off',
  title: 'No workers',
  category: 'llm',
  description: 'The conversation LLM answers everything itself, with no worker tools.',
  inputs: workersNode.inputs,
  outputs: workersNode.outputs,
  params: [],
  create() {
    return {};
  },
};
