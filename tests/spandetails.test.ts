import { describe, expect, it } from 'vitest';
import { chatOf, reassembleOpenAI } from '../src/ui/SpanDetails';

describe('span details', () => {
  it('reads an OpenAI-compatible request with tool calls', () => {
    const c = chatOf({
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
    expect(c.params).toMatchObject({ model: 'm' });
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
});
