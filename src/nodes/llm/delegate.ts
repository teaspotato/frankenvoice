import type { Json, JsonObject } from '../../core/types';

// Delegation: the conversation model keeps the floor and hands slow work to
// the background agent through these tools. Every call is answered at once
// (the task runs on its own); what the agent finds comes back later as a
// <task_update> in a user turn, which the model then tells the user about.

export const TASK_TOOLS: Json = [
  {
    name: 'task_start',
    description:
      'Hand a task to the background agent: a slower, smarter model that reasons and can search and read the web. Use it for anything that needs looking up, current information, or careful multi-step thinking. It runs while you keep talking; its result arrives later as a task_update.',
    input_schema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'What to do, with everything the agent needs from the conversation. It cannot hear the user.' },
      },
      required: ['task'],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: 'task_send',
    description:
      'Send a running or finished task a message. mode "steer" changes or adds to what the agent is doing (on a finished task, it starts a follow-up). mode "ask" is a side question about its progress that leaves the work alone; the answer arrives as a task_update.',
    input_schema: {
      type: 'object',
      properties: {
        task_id: { type: 'string' },
        message: { type: 'string' },
        mode: { type: 'string', enum: ['steer', 'ask'] },
      },
      required: ['task_id', 'message', 'mode'],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: 'task_cancel',
    description: 'Stop a task, e.g. when the user no longer wants it.',
    input_schema: {
      type: 'object',
      properties: { task_id: { type: 'string' } },
      required: ['task_id'],
      additionalProperties: false,
    },
    strict: true,
  },
];

export const DEFAULT_DELEGATE_PROMPT = `You can delegate to a background agent with the task tools. Delegate anything you would otherwise guess at: facts to look up, current information, calculations, or questions that need careful thought. Before calling a task tool, say one short line to the user, like "Let me look that up." Then keep the conversation going; don't wait in silence and don't invent the answer.

Messages wrapped in <task_update> come from the system, not the user. When one arrives with a result or an answer, tell the user what matters in a sentence or two, as speech. Progress notes are for you: mention them only if the user asks how it is going.`;

export type TaskCall = { id: string; name: string; input: JsonObject };

/** The client tool calls in a reply's content blocks. */
export function taskCalls(blocks: Json): TaskCall[] {
  if (!Array.isArray(blocks)) return [];
  return (blocks as JsonObject[])
    .filter((b) => b?.type === 'tool_use')
    .map((b) => ({ id: b.id as string, name: b.name as string, input: (b.input as JsonObject) ?? {} }));
}

/** An agent event, as the text the conversation model reads. */
export function taskUpdate(msg: JsonObject): string {
  const attr = `id="${String(msg.taskId)}" kind="${String(msg.kind)}"`;
  const body =
    msg.kind === 'answer' ? `Question: ${String(msg.question ?? '')}\nAnswer: ${String(msg.text ?? '')}` : String(msg.text ?? msg.error ?? '');
  return `<task_update ${attr}>\n${body.trim()}\n</task_update>`;
}
