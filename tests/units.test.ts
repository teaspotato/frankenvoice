import { describe, expect, it } from 'vitest';
import { buildGraph, configFromGraph, defaultConfig } from '../src/pipeline';
import { SentenceSplitter } from '../src/nodes/segment';
import { Framer } from '../src/nodes/audio';
import { SimScheduler } from '../src/core/scheduler';
import { TrackStore } from '../src/core/tracks';
import { EchoCanceller } from '../src/nodes/cleanup';
import type { AudioMsg, TrajEvent } from '../src/core/types';
import { ActivityTracker } from '../src/ui/activity';
import { TimelineModel } from '../src/ui/timeline';
import { Trajectory, TRAJECTORY_FORMAT } from '../src/core/trajectory';

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

  it('settles a real-time run when stopped mid-way', async () => {
    const sim = new SimScheduler();
    const out: number[] = [];
    for (let t = 0; t <= 10_000; t += 100) sim.at(t, () => out.push(t));
    const done = sim.runRealtime(1);
    setTimeout(() => sim.stop(), 120);
    await done;
    expect(out.length).toBeGreaterThan(0);
    expect(out.length).toBeLessThan(10);
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
    expect(g.nodes[1].params).toEqual({ provider: 'anthropic', effort: 'high', model: 'claude-opus-5-5', opener: false, thinking: 'model default' });
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

describe('TrackStore', () => {
  it('reads sample ranges across chunks without merging', () => {
    const ts = new TrackStore();
    const all: number[] = [];
    for (let c = 0; c < 7; c++) {
      const chunk = new Float32Array(3 + c).map((_, i) => c * 100 + i);
      all.push(...chunk);
      ts.appendF32('a', chunk, 16000);
    }
    for (const [s, e] of [[0, 3], [2, 9], [5, 40], [10, 11], [40, 100], [-3, 2]]) {
      expect([...ts.readF32Range('a', s, e)]).toEqual(all.slice(Math.max(0, s), e));
    }
    ts.bytes('a'); // merge, then keep appending
    ts.appendF32('a', new Float32Array([9, 8]), 16000);
    expect([...ts.readF32Range('a', all.length - 1, all.length + 2)]).toEqual([all[all.length - 1], 9, 8]);
  });
});

describe('EchoCanceller', () => {
  const rate = 16000;
  const params = { filterMs: 64, stepSize: 0.5, delayMs: 0, doubleTalk: 0.6, suppressionDb: 0, adapt: true };
  // Deterministic white-ish noise.
  let seed = 1;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;

  it('learns a delayed, filtered echo of what was played', () => {
    const aec = new EchoCanceller(rate, params);
    const played = new Float32Array(rate * 4).map(() => rnd() * 0.3);
    aec.addReference(500, played, rate); // playback starts at 0.5 s
    const lag = Math.round(0.03 * rate) + 500 * 16; // heard 30 ms later
    const mic = new Float32Array(rate * 5);
    for (let j = 0; j < mic.length; j++) {
      const k = j - lag;
      const a = k >= 0 && k < played.length ? played[k] : 0;
      const b = k - 5 >= 0 && k - 5 < played.length ? played[k - 5] : 0;
      mic[j] = 0.3 * a + 0.12 * b + rnd() * 0.001;
    }
    const out = new Float32Array(mic.length);
    for (let s = 0; s + 512 <= mic.length; s += 512) out.set(aec.process(mic.subarray(s, s + 512), s, 0).out, s);
    const e = (x: Float32Array, a: number, b: number) => x.subarray(a, b).reduce((acc, v) => acc + v * v, 0);
    const erle = 10 * Math.log10(e(mic, rate * 3, rate * 4) / e(out, rate * 3, rate * 4));
    expect(erle).toBeGreaterThan(20);
    // Before playback it passes audio through untouched.
    expect([...out.subarray(0, 4000)]).toEqual([...mic.subarray(0, 4000)]);
  });
});

describe('ActivityTracker', () => {
  const ev = (seq: number, node: string, port: string, data: Record<string, unknown>, kind: TrajEvent['kind'] = 'msg'): TrajEvent => ({ seq, t: seq, node, kind, port, data: data as never });
  it('marks connections carrying data and stages switched on', () => {
    const a = new ActivityTracker();
    a.consume(ev(1, 'vad', 'vad', { type: 'vad.frame', p: 0.9 }), 0);
    expect(a.snapshot(10).edges.has('vad.vad')).toBe(false); // telemetry is not data
    a.consume(ev(2, 'vad', 'vad', { type: 'vad.speech_start' }), 0);
    expect(a.snapshot(10).stages.get('vad')).toBe('speech');
    expect(a.snapshot(10).edges.has('vad.vad')).toBe(true);
    expect(a.snapshot(1000).edges.has('vad.vad')).toBe(false);
    a.consume(ev(3, 'llm', 'text', { type: 'llm.start', rid: 'r1' }), 0);
    expect(a.snapshot(5000).edges.has('llm.text')).toBe(true); // stream open
    a.consume(ev(4, 'llm', 'text', { type: 'llm.end', rid: 'r1' }), 5000);
    expect(a.snapshot(6000).edges.has('llm.text')).toBe(false);
    expect(a.snapshot(6000).stages.has('llm')).toBe(false);
    a.consume(ev(5, 'turn', '', { floor: 'thinking' }, 'state'), 6000);
    expect(a.snapshot(6000).stages.get('turn')).toBe('thinking');
  });
});

describe('SimScheduler pause', () => {
  it('freezes the clock while paused and resumes from where it stopped', async () => {
    let wall = 0;
    const perf = { now: () => wall };
    const s = new SimScheduler();
    const fired: number[] = [];
    for (const t of [10, 20, 30]) s.at(t, () => fired.push(t));
    const done = s.runRealtime(1, perf);
    wall = 15;
    await new Promise((r) => setTimeout(r, 60));
    expect(fired).toEqual([10]);
    s.pause();
    wall = 1000;
    await new Promise((r) => setTimeout(r, 60));
    expect(fired).toEqual([10]);
    expect(s.now()).toBeLessThan(20);
    s.resume();
    wall = 1010;
    await new Promise((r) => setTimeout(r, 60));
    expect(fired).toEqual([10, 20]);
    wall = 1100;
    await done;
    expect(fired).toEqual([10, 20, 30]);
  });
});

describe('TimelineModel.related', () => {
  it('walks causes up and effects down, never sideways', () => {
    const traj = new Trajectory({ format: TRAJECTORY_FORMAT, formatVersion: 1, id: 't', createdAt: '', mode: 'live', graph: { nodes: [], edges: [] } as never, graphHash: '' });
    const add = (seq: number, cause?: number[]) => traj.append({ seq, t: seq, node: 'n', kind: 'msg', port: 'p', data: { type: 'x' }, ...(cause ? { cause } : {}) });
    add(0); // shared connection
    add(1, [0]); // result A
    add(2, [0]); // result B (sibling of A through the connection)
    add(3, [1]); // produced by A
    add(4, [3]);
    const m = new TimelineModel(traj);
    m.update();
    const r = m.related([1]);
    expect([...r.up]).toEqual([0]);
    expect([...r.down].sort()).toEqual([3, 4]);
    expect(r.down.has(2)).toBe(false);
  });
});

describe('SimScheduler seek', () => {
  it('runs everything before the target at once, then keeps pacing', async () => {
    let wall = 0;
    const perf = { now: () => wall };
    const s = new SimScheduler();
    const fired: number[] = [];
    for (const t of [10, 500, 520]) s.at(t, () => fired.push(t));
    const done = s.runRealtime(1, perf);
    s.seek(505);
    expect(fired).toEqual([10, 500]);
    expect(s.now()).toBe(505);
    wall = 20;
    await done;
    expect(fired).toEqual([10, 500, 520]);
  });
});

describe('configFromGraph', () => {
  it('turns the turn model off for sessions recorded without one', () => {
    const c = defaultConfig();
    c.turnmodel.variant = 'off';
    const g = buildGraph(c);
    g.nodes = g.nodes.filter((n) => n.id !== 'turnmodel');
    expect(configFromGraph(g).turnmodel.variant).toBe('off');
    expect(configFromGraph(buildGraph(defaultConfig())).turnmodel.variant).toBe('ssl_turn');
  });
});
