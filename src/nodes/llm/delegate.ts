import type { Json, JsonObject } from '../../core/types';

// Delegation: the conversation model keeps the floor and hands slow work to
// workers through these tools, one worker per task, as many as it needs at
// once. Every call is answered at once (the worker runs on its own); what a
// worker finds comes back later as a <worker_update> in a user turn, which
// the model then tells the user about.

export const WORKER_TOOLS: Json = [
  {
    name: 'spawn_worker',
    description:
      'Start a worker on a task: a full agent (a slower, smarter model with its own tools) that reasons and can search and read the web. Use it for anything that needs looking up, current information, or careful multi-step work. Workers run in parallel while you keep talking; for independent parts of a request, start one worker each. A worker\'s result arrives later as a worker_update.',
    input_schema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'What to do, with everything the worker needs from the conversation. It cannot hear the user.' },
      },
      required: ['task'],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: 'message_worker',
    description:
      'Send a worker a message: it changes or adds to what the worker is doing. On a finished worker it starts a follow-up in the same session, so the worker still knows what it did.',
    input_schema: {
      type: 'object',
      properties: {
        worker_id: { type: 'string' },
        message: { type: 'string' },
      },
      required: ['worker_id', 'message'],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: 'check_worker',
    description: "See how a worker is doing: its status, its plan and its latest steps. Answered at once; the worker isn't interrupted.",
    input_schema: {
      type: 'object',
      properties: { worker_id: { type: 'string' } },
      required: ['worker_id'],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: 'stop_worker',
    description: 'Stop a worker, e.g. when the user no longer wants what it is doing.',
    input_schema: {
      type: 'object',
      properties: { worker_id: { type: 'string' } },
      required: ['worker_id'],
      additionalProperties: false,
    },
    strict: true,
  },
];

export const DEFAULT_DELEGATE_PROMPT = `You can hand work to workers with the worker tools. Delegate anything you would otherwise guess at: facts to look up, current information, calculations, or questions that need careful thought. Split independent parts across several workers so they run in parallel. Before calling a worker tool, say one short line to the user, like "Let me look that up." Then keep the conversation going; don't wait in silence and don't invent the answer.

Messages wrapped in <worker_update> come from the system, not the user. When one arrives with a result, tell the user what matters in a sentence or two, as speech. Progress notes are for you: mention them only if the user asks how it is going, and use check_worker when they ask and you have nothing recent.`;

export type ToolCall = { id: string; name: string; input: JsonObject };

/** The client tool calls in a reply's content blocks. */
export function toolCalls(blocks: Json): ToolCall[] {
  if (!Array.isArray(blocks)) return [];
  return (blocks as JsonObject[])
    .filter((b) => b?.type === 'tool_use')
    .map((b) => ({ id: b.id as string, name: b.name as string, input: (b.input as JsonObject) ?? {} }));
}

/** A worker's event, as the text the conversation model reads. */
export function workerUpdate(msg: JsonObject): string {
  const attr = `id="${String(msg.workerId)}" kind="${String(msg.kind)}"`;
  const body = String(msg.text ?? msg.error ?? '');
  return `<worker_update ${attr}>\n${body.trim()}\n</worker_update>`;
}

/** What the conversation model knows about one worker, kept from its updates. */
export interface WorkerInfo {
  status: 'running' | 'done' | 'stopped' | 'error';
  task: string;
  /** The reply that started it. */
  rid: string;
  t0: number;
  /** The latest tool steps, newest last. */
  steps: string[];
  plan: string;
}

/** check_worker's answer. */
export function workerStatus(id: string, w: WorkerInfo, now: number): string {
  const lines = [`Worker ${id}: ${w.status}, ${Math.round((now - w.t0) / 1000)} s since it started.`, `Task: ${w.task}`];
  if (w.plan) lines.push(`Plan:\n${w.plan}`);
  lines.push(w.steps.length ? `Latest steps:\n${w.steps.map((s) => `- ${s}`).join('\n')}` : 'No steps yet.');
  return lines.join('\n');
}
