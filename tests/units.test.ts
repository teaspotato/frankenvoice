import { describe, expect, it } from 'vitest';
import { SentenceSplitter } from '../src/nodes/segment';
import { Framer } from '../src/nodes/audio';
import { SimScheduler } from '../src/core/scheduler';
import type { AudioMsg } from '../src/core/types';

describe('SentenceSplitter', () => {
  it('emits sentences as soon as their boundary is certain', () => {
    const s = new SentenceSplitter();
    expect(s.push('Hello there')).toEqual([]);
    expect(s.push('! How are')).toEqual(['Hello there!']);
    expect(s.push(' you? Pi is 3.')).toEqual(['How are you?']);
    expect(s.push('14.')).toEqual([]);
    expect(s.flush()).toBe('Pi is 3.14.');
  });
});

describe('Framer', () => {
  it('re-chunks audio into fixed frames with exact sample ranges', () => {
    const f = new Framer(4);
    const frames: [number, number][] = [];
    const msg = (start: number, n: number): AudioMsg => ({ type: 'audio', track: 'm', start, end: start + n, sampleRate: 16000, samples: new Float32Array(n) });
    f.push(msg(0, 3), (_x, _t, a, b) => frames.push([a, b]));
    f.push(msg(3, 6), (_x, _t, a, b) => frames.push([a, b]));
    expect(frames).toEqual([
      [0, 4],
      [4, 8],
    ]);
  });
});

describe('SimScheduler', () => {
  it('runs items in time order, breaking ties by order', () => {
    const sim = new SimScheduler();
    const out: string[] = [];
    sim.at(5, () => out.push('c'));
    sim.at(1, () => out.push('b'), 20);
    sim.at(1, () => out.push('a'), 10);
    const cancel = sim.at(3, () => out.push('x'));
    cancel();
    sim.runFast();
    expect(out).toEqual(['a', 'b', 'c']);
    expect(sim.now()).toBe(5);
  });
});

describe('migrateGraph', () => {
  it('upgrades format-1 per-provider node types to provider params', async () => {
    const { migrateGraph, registry } = await import('../src/nodes');
    const old: import('../src/core/graph').GraphSpec = {
      format: 'frankenvoice-graph',
      formatVersion: 1,
      name: 'old',
      nodes: [
        { id: 'asr', type: 'asr.deepgram', params: { model: 'nova-2' } },
        { id: 'llm', type: 'llm.claude', params: { effort: 'high' } },
        { id: 'tts', type: 'tts.cartesia', params: { modelId: 'sonic-x', minSegmentChars: 5 } },
      ],
      edges: [],
    };
    const g = migrateGraph(old);
    expect(g.nodes.map((n) => n.type)).toEqual(['asr', 'llm', 'tts']);
    expect(g.nodes[0].params).toEqual({ provider: 'deepgram', 'deepgram.model': 'nova-2' });
    expect(g.nodes[1].params).toEqual({ provider: 'anthropic', effort: 'high' });
    expect(g.nodes[2].params).toEqual({ provider: 'cartesia', 'cartesia.model': 'sonic-x', minSegmentChars: 5 });
    for (const n of g.nodes) expect(registry[n.type]).toBeDefined();
  });
});

describe('openai-compatible adapter', () => {
  it('builds a chat request and parses the SSE payloads', async () => {
    const { openaiCompatAdapter } = await import('../src/nodes/llm/openai');
    const req = openaiCompatAdapter.buildRequest(
      [
        { role: 'user', text: 'hi' },
        { role: 'assistant', text: 'hello' },
        { role: 'user', text: 'how are you' },
      ],
      { endpoint: 'openrouter', model: 'm', system: 'sys', maxTokens: 50, extraBody: '{"temperature":0.2}' },
    ) as { endpoint: string; body: Record<string, unknown> };
    expect(req.endpoint).toBe('openrouter');
    expect(req.body).toEqual({
      model: 'm',
      stream: true,
      max_tokens: 50,
      temperature: 0.2,
      messages: [
        { role: 'system', content: 'sys' },
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello' },
        { role: 'user', content: 'how are you' },
      ],
    });
    const p = openaiCompatAdapter.parser();
    expect(p.recv({ choices: [{ index: 0, delta: { role: 'assistant', content: '' } }] })).toEqual([]);
    expect(p.recv({ choices: [{ index: 0, delta: { content: 'Hi' } }] })).toEqual([{ type: 'delta', text: 'Hi' }]);
    expect(p.recv({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })).toEqual([]);
    expect(p.recv({ $done: true })).toEqual([{ type: 'end', stopReason: 'stop' }]);
    expect(p.close(null)).toEqual([]);
    expect(openaiCompatAdapter.parser().recv({ error: { message: 'bad model' } })).toEqual([{ type: 'error', error: { message: 'bad model' } }]);
  });
});
