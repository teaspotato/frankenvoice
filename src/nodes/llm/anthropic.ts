import type { Json, JsonObject } from '../../core/types';
import type { ChatTurn, LlmAdapter, LlmStreamEvent } from './types';

type Block = JsonObject & { type: string };

/** How a model turns thinking off; Opus models cannot, so nothing is sent for them. */
export function thinkingOff(p: JsonObject): JsonObject {
  if (p.thinking !== 'off') return {};
  const model = String(p.model ?? '');
  if (model.includes('haiku')) return { thinking: { type: 'disabled' } };
  if (model.includes('sonnet')) return { thinking: { type: 'between_tools' } };
  return {};
}

/** Claude Messages API. Payloads are the SDK's raw stream events. */
export const anthropicAdapter: LlmAdapter = {
  provider: 'anthropic',
  channel: 'anthropic',
  defaultModel: 'claude-haiku-5-5',
  buildRequest(history: ChatTurn[], p: JsonObject): Json {
    const messages: { role: string; content: Json }[] = [];
    for (const t of history) {
      // A copy: same-role turns are merged into it below, and the history must not change.
      const native = t.native?.provider === 'anthropic' ? t.native.content : null;
      const content: Json = Array.isArray(native) ? [...native] : (native ?? [{ type: 'text', text: t.text }]);
      const last = messages[messages.length - 1];
      if (last?.role === t.role && t.role === 'user') (last.content as Json[]).push(...(content as Json[]));
      else messages.push({ role: t.role, content });
    }
    return {
      model: p.model,
      max_tokens: p.maxTokens,
      system: p.system,
      messages: messages as Json,
      ...(Array.isArray(p.tools) && p.tools.length ? { tools: p.tools } : {}),
      output_config: { effort: p.effort },
      ...thinkingOff(p),
      ...(p.fallbacks ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' } : {}),
    };
  },
  parser() {
    const blocks: Block[] = [];
    const json: Record<number, string> = {};
    let stopReason: string | null = null;
    return {
      recv(data): LlmStreamEvent[] {
        const ev = data as JsonObject;
        switch (ev.type) {
          case 'content_block_start': {
            const b = { ...(ev.content_block as Block) };
            blocks[ev.index as number] = b;
            return b.type === 'thinking' || b.type === 'redacted_thinking' ? [{ type: 'thinking' }] : [];
          }
          case 'content_block_delta': {
            const b = blocks[ev.index as number];
            const d = ev.delta as JsonObject;
            if (!b) return [];
            if (d.type === 'text_delta') {
              b.text = ((b.text as string) ?? '') + (d.text as string);
              return [{ type: 'delta', text: d.text as string }];
            }
            if (d.type === 'thinking_delta') b.thinking = ((b.thinking as string) ?? '') + (d.thinking as string);
            else if (d.type === 'signature_delta') b.signature = d.signature as string;
            else if (d.type === 'input_json_delta') json[ev.index as number] = (json[ev.index as number] ?? '') + (d.partial_json as string);
            return [];
          }
          case 'content_block_stop': {
            const i = ev.index as number;
            if (json[i] !== undefined && blocks[i]) {
              try {
                blocks[i].input = JSON.parse(json[i] || '{}') as Json;
              } catch {
                blocks[i].input = {};
              }
            }
            return [];
          }
          case 'message_delta':
            stopReason = ((ev.delta as JsonObject).stop_reason as string) ?? stopReason;
            return [];
          case 'message_stop':
            return [{ type: 'end', stopReason }];
          case 'error':
            return [{ type: 'error', error: ev.error ?? null }];
          default:
            return [];
        }
      },
      close(data) {
        const d = data as JsonObject | null;
        return d?.error ? [{ type: 'error', error: d.error }] : [{ type: 'end', stopReason }];
      },
      native() {
        return blocks.filter((b) => b && (b.type !== 'text' || (b.text as string))) as Json;
      },
    };
  },
};
