import { describe, expect, it } from 'vitest';
import { chatOf, reassembleAnthropic, reassembleOpenAI } from '../src/ui/SpanDetails';

describe('span details', () => {
  it('reads an OpenAI-compatible request with tool calls', () => {
    const c = chatOf({
      endpoint: 'openrouter',
      body: {
        model: 'm',
        stream: true,
        tools: [{}, {}],
        messages: [
          { role: 'system', content: 'sys' },
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: null, tool_calls: [{ function: { name: 'clock', arguments: '{}' } }] },
          { role: 'tool', content: '12:00' },
        ],
      },
    })!;
    expect(c.turns.map((t) => t.role)).toEqual(['user', 'assistant', 'tool']);
    expect(c.turns[1].blocks).toEqual([{ type: 'tool_use', text: 'clock {}' }]);
    expect(c.params).toMatchObject({ endpoint: 'openrouter', model: 'm' });
  });

  it('reassembles OpenAI streams, tool calls, other choices and errors', () => {
    const r = reassembleOpenAI([
      { choices: [{ index: 1, delta: { content: 'NO' } }] },
      { choices: [{ index: 0, delta: { content: 'Hel' } }] },
      { choices: [{ index: 0, delta: { content: 'lo', tool_calls: [{ index: 0, function: { name: 'clo', arguments: '{"a"' } }] } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { name: 'ck', arguments: ':1}' } }] } }] },
      { error: { message: 'boom' } },
    ])!;
    expect(r.blocks.map((b) => b.type)).toEqual(['text', 'tool_use', 'error']);
    expect(r.blocks[0].text).toBe('Hello');
    expect(r.blocks[1].text).toBe('clock {"a":1}');
    expect(r.stop).toBe('error');
    expect(reassembleOpenAI([{ choices: [{ message: { content: 'whole' }, finish_reason: 'stop' }] }])!.blocks).toEqual([{ type: 'text', text: 'whole' }]);
  });

  it('keeps the partial Anthropic reply when the stream errors', () => {
    const r = reassembleAnthropic([
      { type: 'message_start', message: { usage: {} } },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { text: 'part' } },
      { type: 'error', error: { type: 'overloaded_error' } },
    ])!;
    expect(r.blocks.map((b) => b.type)).toEqual(['text', 'error']);
    expect(r.blocks[0].text).toBe('part');
  });
});
