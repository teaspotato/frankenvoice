import { describe, expect, it } from 'vitest';
import { anthropicAdapter } from '../src/nodes/llm/anthropic';

const req = (model: string, thinking: string) =>
  anthropicAdapter.buildRequest([{ role: 'user', text: 'hi' }], { model, thinking, effort: 'low', maxTokens: 100, system: '' }) as Record<string, unknown>;

describe('anthropic thinking param', () => {
  it('turns thinking off the way each model accepts', () => {
    expect(req('claude-haiku-5-5', 'off').thinking).toEqual({ type: 'disabled' });
    expect(req('claude-sonnet-5-5', 'off').thinking).toEqual({ type: 'between_tools' });
    expect(req('claude-opus-5-5', 'off').thinking).toBeUndefined();
    expect(req('claude-haiku-5-5', 'model default').thinking).toBeUndefined();
  });

  it('reports a thinking block as thinking, before any text', () => {
    const parser = anthropicAdapter.parser();
    expect(parser.recv({ type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } })).toEqual([{ type: 'thinking' }]);
    expect(parser.recv({ type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } })).toEqual([]);
  });
});
