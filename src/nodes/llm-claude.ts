import type { NodeDef } from '../core/node';
import type { Json, JsonObject } from '../core/types';

interface ClaudeParams {
  model: string;
  system: string;
  effort: string;
  maxTokens: number;
  fallbacks: boolean;
}

type Block = JsonObject & { type: string };
interface ApiMessage {
  role: 'user' | 'assistant';
  content: Block[];
}

const DEFAULT_SYSTEM =
  'You are a voice assistant in a live spoken conversation. Your words are converted to speech, so answer in short, natural spoken sentences. No markdown, lists, or emoji.';

/**
 * Claude via the Messages API, streamed. The request (full history included) is
 * the io.open payload and every stream event is an io.recv, so a replay sees
 * the identical token stream with identical timing.
 */
export const claudeLlmNode: NodeDef<ClaudeParams> = {
  type: 'llm.claude',
  title: 'Claude',
  category: 'llm',
  description: 'Streams a reply for each user turn; keeps the conversation history.',
  inputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'text', type: 'text' }],
  params: [
    { key: 'model', label: 'Model', type: 'string', default: 'claude-opus-5-5' },
    { key: 'effort', label: 'Effort', type: 'select', options: ['low', 'medium', 'high', 'xhigh', 'max'], default: 'low' },
    { key: 'maxTokens', label: 'Max tokens', type: 'number', default: 4096 },
    { key: 'fallbacks', label: 'Server-side refusal fallback', type: 'boolean', default: true },
    { key: 'system', label: 'System prompt', type: 'text', default: DEFAULT_SYSTEM },
  ],
  channels: [{ name: 'anthropic', description: 'Messages API stream (raw stream events)' }],
  create(ctx) {
    const p = ctx.params;
    const history: ApiMessage[] = [];
    let n = 0;
    let active:
      | { io: ReturnType<typeof ctx.io.open>; rid: string; blocks: Block[]; json: Record<number, string>; text: string; stopReason: string | null }
      | undefined;

    const finish = (status: 'done' | 'cancelled' | 'error', error?: Json) => {
      if (!active) return;
      const a = active;
      active = undefined;
      if (!a.io.closed) a.io.close({ status });
      // Keep whatever the model produced, so the next request continues from it.
      const content = a.blocks.filter((b) => b.type !== 'text' || (b.text as string));
      if (content.length) history.push({ role: 'assistant', content });
      ctx.emit('text', { type: 'llm.end', rid: a.rid, status, stopReason: a.stopReason, text: a.text, ...(error ? { error } : {}) });
      ctx.setState('status', status === 'error' ? 'error' : 'idle');
      ctx.setState('turns', history.length);
    };

    const onEvent = (ev: JsonObject) => {
      const a = active!;
      switch (ev.type) {
        case 'content_block_start':
          a.blocks[ev.index as number] = { ...(ev.content_block as Block) };
          break;
        case 'content_block_delta': {
          const b = a.blocks[ev.index as number];
          const d = ev.delta as JsonObject;
          if (!b) break;
          if (d.type === 'text_delta') {
            b.text = ((b.text as string) ?? '') + (d.text as string);
            a.text += d.text as string;
            ctx.emit('text', { type: 'llm.delta', rid: a.rid, text: d.text as string });
          } else if (d.type === 'thinking_delta') {
            b.thinking = ((b.thinking as string) ?? '') + (d.thinking as string);
          } else if (d.type === 'signature_delta') {
            b.signature = d.signature as string;
          } else if (d.type === 'input_json_delta') {
            a.json[ev.index as number] = (a.json[ev.index as number] ?? '') + (d.partial_json as string);
          }
          break;
        }
        case 'content_block_stop': {
          const i = ev.index as number;
          if (a.json[i] !== undefined && a.blocks[i]) {
            try {
              a.blocks[i].input = JSON.parse(a.json[i] || '{}') as Json;
            } catch {
              a.blocks[i].input = {};
            }
          }
          break;
        }
        case 'message_delta':
          a.stopReason = ((ev.delta as JsonObject).stop_reason as string) ?? a.stopReason;
          break;
        case 'message_stop':
          finish('done');
          break;
        case 'error':
          finish('error', ev.error ?? null);
          break;
      }
    };

    return {
      onInput(port, msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') finish('cancelled');
          return;
        }
        if (msg.type !== 'turn.user') return;
        finish('cancelled');
        const text = msg.text as string;
        const last = history[history.length - 1];
        if (last?.role === 'user') last.content.push({ type: 'text', text });
        else history.push({ role: 'user', content: [{ type: 'text', text }] });

        const rid = `${ctx.id}-r${n++}`;
        const request: JsonObject = {
          model: p.model,
          max_tokens: p.maxTokens,
          system: p.system,
          messages: JSON.parse(JSON.stringify(history)) as Json,
          output_config: { effort: p.effort },
          ...(p.fallbacks ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' } : {}),
        };
        ctx.emit('text', { type: 'llm.start', rid, turnId: (msg.turnId as string) ?? null });
        ctx.setState('status', 'generating');
        const io = ctx.io.open('anthropic', request, {
          onRecv(data) {
            if (active?.io === io) onEvent(data as JsonObject);
          },
          onClose(data) {
            if (active?.io !== io) return;
            const d = data as JsonObject | null;
            if (d?.error) finish('error', d.error);
            else finish('done');
          },
        });
        active = { io, rid, blocks: [], json: {}, text: '', stopReason: null };
      },
      stop() {
        finish('cancelled');
      },
    };
  },
};
