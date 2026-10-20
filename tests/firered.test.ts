import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import { Runtime } from '../src/core/runtime';
import { LiveScheduler } from '../src/core/scheduler';
import { LiveWorld } from '../src/core/world';
import { prepareReplay } from '../src/core/replay';
import { registry } from '../src/nodes';
import { buildGraph } from '../src/pipeline';
import { FireRedStream, KaldiFbank, MEL_BINS } from '../src/drivers/firered/stream';
import { apiConfig, fakeDrivers } from './fake-drivers';

describe('FireRedVAD features', () => {
  it('produces Kaldi fbank frames: 25 ms windows every 10 ms, snip_edges, 80 bins', () => {
    const fb = new KaldiFbank();
    const sine = Float32Array.from({ length: 800 }, (_, n) => 0.5 * Math.sin((2 * Math.PI * 1000 * n) / 16000));
    // Split across pushes: frames come out as soon as their window is complete.
    const a = fb.push(sine.subarray(0, 399));
    const b = fb.push(sine.subarray(399));
    expect(a).toHaveLength(0);
    expect(b).toHaveLength(3); // 1 + floor((800 - 400) / 160)
    expect(b[0]).toHaveLength(MEL_BINS);
    // kaldi-native-fbank 1.22 on the same signal: peak at bin 27 = 27.0539, bin 0 = 6.0378, bin 79 = -1.3341.
    const f = Array.from(b[0]);
    expect(f.indexOf(Math.max(...f))).toBe(27);
    expect(f[27]).toBeCloseTo(27.0539, 3);
    expect(f[0]).toBeCloseTo(6.0378, 3);
    // Near-empty bins differ a little more: kaldi-native-fbank computes in float32.
    expect(f[79]).toBeCloseTo(-1.3341, 2);
  });

  it('runs the bundled streaming model: one probability per frame, low on silence', async () => {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(new Uint8Array(readFileSync('src/drivers/firered/stream_vad.onnx')));
    const vad = new FireRedStream(ort, session);
    const probs: number[] = [];
    for (let i = 0; i < 10; i++) probs.push(...(await vad.push(new Float32Array(480))));
    expect(probs).toHaveLength(28); // 4800 samples: 1 + floor((4800 - 400) / 160)
    for (const p of probs) expect(p).toBeLessThan(0.2);
  }, 20_000);
});

describe('vad.firered node', () => {
  it('segments speech from per-frame probabilities and replays exactly', async () => {
    const c = apiConfig();
    c.vad.variant = 'firered';
    const g = buildGraph(c);
    expect(g.nodes.find((n) => n.id === 'vad')!.type).toBe('vad.firered');
    const scheduler = new LiveScheduler(performance);
    const rt = new Runtime({ graph: g, registry, scheduler, world: new LiveWorld(fakeDrivers, scheduler), mode: 'live' });
    rt.start();
    await new Promise((r) => setTimeout(r, 1500));
    rt.stop();
    const live = rt.trajectory;
    const vad = live.events.filter((e) => e.node === 'vad' && e.kind === 'msg').map((e) => e.data as { type: string; start?: number; end?: number; at?: number });
    const frames = vad.filter((m) => m.type === 'vad.frame');
    expect(frames.length).toBeGreaterThan(50);
    // 10 ms hop, 25 ms windows, on the mic's sample clock.
    expect(frames[1].start! - frames[0].start!).toBe(160);
    expect(frames[0].end! - frames[0].start!).toBe(400);
    expect(vad.some((m) => m.type === 'vad.speech_start')).toBe(true);
    expect(live.events.some((e) => e.kind === 'io.open' && e.node === 'vad' && (e.data as { channel: string }).channel === 'firered')).toBe(true);
    expect(live.events.filter((e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toEqual([]);
    const { runtime, sim } = prepareReplay(live, { registry });
    sim.runFast();
    expect(runtime.trajectory.events).toEqual(live.events);
  }, 10_000);
});
