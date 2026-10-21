import { describe, expect, it } from 'vitest';
import { duplexTurnNode } from '../src/nodes/turn';
import { speakerNode } from '../src/nodes/speaker';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';
import { Runtime } from '../src/core/runtime';
import { LiveScheduler } from '../src/core/scheduler';
import { LiveWorld, type LiveDriver } from '../src/core/world';
import { prepareReplay } from '../src/core/replay';
import { decodeBundle, encodeBundle } from '../src/core/bundle';
import { registry } from '../src/nodes';
import { buildGraph } from '../src/pipeline';
import { fakeDrivers, testConfig } from './fake-drivers';

/** Drives one node by hand: a manual clock, timers, what it emitted and sent, and its IO handlers. */
function harness<P>(def: NodeDef<P>, params: Record<string, unknown>) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const sent: Json[] = [];
  const ios: IOHandlers[] = [];
  const ctx: NodeContext = {
    id: def.type === 'speaker' ? 'speaker' : 'turn',
    params: resolveParams(def, params as never),
    now: () => now,
    cause: undefined,
    emit: (port, msg) => (out.push({ port, msg }), out.length),
    setState() {},
    log() {},
    setTimer(ms, fn) {
      const t = { at: now + ms, fn, live: true };
      timers.push(t);
      return { clear: () => void (t.live = false) };
    },
    io: {
      open(_channel, _request, h) {
        ios.push(h);
        return { xid: String(ios.length), send: (d: Json) => void sent.push(d), close() {}, closed: false };
      },
    },
    media: {
      append: () => {
        throw new Error('no media');
      },
    },
  };
  const node = def.create(ctx as never);
  node.start?.();
  const advance = (to: number) => {
    for (;;) {
      const t = timers.filter((x) => x.live && x.at <= to).sort((a, b) => a.at - b.at)[0];
      if (!t) break;
      t.live = false;
      now = t.at;
      t.fn();
    }
    now = to;
  };
  const send = (t: number, port: string, msg: Msg) => (advance(t), node.onInput?.(port, msg));
  const recv = (t: number, data: Json) => (advance(t), ios[0].onRecv?.(data, undefined));
  const of = (type: string) => out.filter((o) => o.msg.type === type).map((o) => o.msg);
  return { send, recv, advance, out, sent, of };
}

const sr = 16;

describe('page gate: turn-taking', () => {
  const turn = (params: Record<string, unknown> = {}) => {
    const h = harness(duplexTurnNode, { overlap: 'classify', eotCheck: 'off', yieldWithinMs: 1000, ...params });
    h.send(1000, 'transcript', { type: 'asr.final', text: 'tell me about Tokyo', start: 500 * sr, end: 1000 * sr, speechFinal: true });
    h.send(1500, 'playback', { type: 'playback.start', rid: 'r0' });
    return h;
  };
  const controls = (h: ReturnType<typeof turn>) => h.out.filter((o) => o.port === 'control').map((o) => o.msg);
  const decisions = (h: ReturnType<typeof turn>) => h.of('turn.decision').map((m) => m.decision);
  const states = (h: ReturnType<typeof turn>) => h.of('turn.transition').map((m) => m.to);

  it('arms the page for the quick-yield window of a reply and disarms when the agent stops holding the floor', () => {
    const h = turn();
    expect(controls(h)).toEqual([{ type: 'gate', until: 2500 }]);
    h.send(3000, 'playback', { type: 'playback.end', rid: 'r0' });
    expect(controls(h)).toEqual([
      { type: 'gate', until: 2500 },
      { type: 'gate', until: null },
    ]);
  });

  it('leaves the page alone when off, or without a quick yield', () => {
    expect(controls(turn({ pageGate: false }))).toEqual([]);
    expect(controls(turn({ yieldWithinMs: 0 }))).toEqual([]);
    expect(controls(turn({ bargeIn: false }))).toEqual([]);
  });

  it('yields once the VAD here hears the speech the page stopped the agent on', () => {
    const h = turn();
    h.send(1820, 'playback', { type: 'playback.paused', rid: 'r0', at: 1840, from: 1780, gate: true, cut: 1800, pos: 1700 * sr });
    expect(decisions(h)).toEqual(['candidate']);
    expect(states(h).at(-1)).toBe('speaking+user');
    // Nothing more goes to the speaker: it already stopped.
    expect(controls(h).slice(1)).toEqual([]);
    h.send(1900, 'vad', { type: 'vad.speech_start', at: 1700 * sr });
    expect(decisions(h)).toEqual(['candidate', 'interrupt']);
    expect(controls(h).slice(1)).toEqual([{ type: 'cancel' }]);
    expect(states(h).at(-1)).toBe('listening');
    h.send(1905, 'playback', { type: 'playback.interrupted', rid: 'r0', at: 1905, heard: 'Tokyo' });
    expect(h.of('turn.heard')).toEqual([{ type: 'turn.heard', turnId: 'turn-u0', rid: 'r0', text: 'Tokyo' }]);
    // The page's timer never fires a resume.
    h.advance(5000);
    expect(controls(h).slice(1)).toEqual([{ type: 'cancel' }]);
  });

  it('plays on from where it stopped when the VAD here hears nothing (echo or noise in the page)', () => {
    const h = turn();
    h.send(1820, 'playback', { type: 'playback.paused', rid: 'r0', at: 1840, from: 1780, gate: true, cut: 1800, pos: 1700 * sr });
    h.advance(2419);
    expect(controls(h).slice(1)).toEqual([]);
    h.advance(2420);
    expect(controls(h).slice(1)).toEqual([{ type: 'resume' }]);
    expect(decisions(h)).toEqual(['candidate', 'noise']);
    expect(states(h).at(-1)).toBe('speaking');
    // Speech later in the reply goes by the usual rules (and the page is not armed again).
    h.send(4000, 'vad', { type: 'vad.speech_start', at: 4000 * sr });
    expect(decisions(h).at(-1)).toBe('candidate');
    expect(controls(h).filter((c) => c.type === 'gate')).toEqual([{ type: 'gate', until: 2500 }]);
  });

  it('yields when the VAD here heard the speech first but too late for the quick yield', () => {
    const h = turn({ holdAfterMs: 1000 });
    // The page heard it at 2450 (in the window); the VAD here reported it at 2550 (past it).
    h.send(2550, 'vad', { type: 'vad.speech_start', at: 2400 * sr });
    expect(decisions(h)).toEqual(['candidate']);
    h.send(2560, 'playback', { type: 'playback.paused', rid: 'r0', at: 2490, from: 2400, gate: true, cut: 2450, pos: 2400 * sr });
    expect(decisions(h)).toEqual(['candidate', 'interrupt']);
    expect(controls(h).at(-1)).toEqual({ type: 'cancel' });
  });
});

describe('page gate: speaker', () => {
  const speaker = () => {
    const h = harness(speakerNode, {});
    let pos = 0;
    const play = (t: number, seg = 0, loud = 0) => {
      const samples = new Float32Array(2400).map((_, i) => loud * Math.sin(i / 3));
      h.send(t, 'audio', { type: 'audio', track: 'tts.audio', start: pos, end: pos + samples.length, sampleRate: 24000, samples, rid: 'r0', seg } as never);
      pos += samples.length;
    };
    const plays = () => h.sent.filter((d) => (d as { cmd: string }).cmd === 'play').map((d) => (d as { at: number }).at);
    return { ...h, play, plays };
  };

  it('passes the arm on to the page with how to cut off', () => {
    const s = speaker();
    s.play(0);
    s.send(0, 'control', { type: 'gate', until: 1000 });
    expect(s.sent.at(-1)).toEqual({ cmd: 'gate', until: 1000, cutoff: 'syllable', maxFinishMs: 300, tailMs: 60, releaseMs: 40 });
  });

  it("takes the page's cut-off: heard up to the syllable's end, silent after the tail", () => {
    const s = speaker();
    s.play(0);
    s.play(0);
    s.recv(190, { type: 'gate', at: 150, pos: 2000, stop: 210, silent: 270 });
    expect(s.of('playback.paused')).toEqual([{ type: 'playback.paused', rid: 'r0', at: 270, from: 150, gate: true, cut: 210, pos: 2000 }]);
    expect(s.of('played.flush')).toEqual([{ type: 'played.flush', at: 270 }]);
  });

  it('gives the echo reference the tail the page played, not the reply past where it took over', () => {
    const s = speaker();
    for (let i = 0; i < 4; i++) s.play(0, 0, 0.5);
    // The page heard the user at 150; the sound ended at 300, the tail took over 90 ms before.
    s.recv(190, { type: 'gate', at: 150, pos: 3600, stop: 300, silent: 360 });
    expect(s.of('played.flush')).toEqual([{ type: 'played.flush', at: 210 }]);
    const tail = s.out.filter((o) => o.port === 'played' && o.msg.type === 'audio').at(-1)?.msg as unknown as { at: number; start: number; end: number; samples: Float32Array };
    expect(tail).toMatchObject({ at: 210, start: 5040 });
    expect(tail.end - tail.start).toBe(tail.samples.length);
    // The 90 ms before the sound's end, sustained over the 60 ms tail.
    expect(tail.samples.length).toBeGreaterThan(0.09 * 24000);
  });

  it("takes the page's stop as a pause where the page stopped, acks it, and reports what was heard up to there", () => {
    const s = speaker();
    s.send(0, 'audio', { type: 'tts.segment_start', rid: 'r0', seg: 0, text: 'Tokyo is big.' });
    s.play(0);
    s.send(0, 'audio', { type: 'tts.segment_start', rid: 'r0', seg: 1, text: 'It has trains.' });
    s.play(0, 1);
    s.play(30, 1);
    // The page stopped at 150; its report lands at 190.
    s.recv(190, { type: 'gate', at: 150, pos: 2000 });
    expect(s.sent.at(-1)).toEqual({ cmd: 'flush' });
    expect(s.of('playback.paused')).toEqual([{ type: 'playback.paused', rid: 'r0', at: 190, from: 150, gate: true, cut: 150, pos: 2000 }]);
    expect(s.of('played.flush')).toEqual([{ type: 'played.flush', at: 190 }]);
    s.send(400, 'control', { type: 'cancel' });
    expect(s.of('playback.interrupted')).toEqual([{ type: 'playback.interrupted', rid: 'r0', at: 400, heard: 'Tokyo is big. It' }]);
  });

  it('plays the rest from where the page stopped on resume', () => {
    const s = speaker();
    s.play(0);
    s.play(0);
    s.recv(90, { type: 'gate', at: 50 });
    s.send(700, 'control', { type: 'resume' });
    expect(s.of('playback.resumed')).toEqual([{ type: 'playback.resumed', rid: 'r0', at: 700, from: 50, shift: 650 }]);
    expect(s.plays().slice(2)).toEqual([700, 750]);
  });

  it('only acks a stop that comes after the reply ended or was already cut here', () => {
    const s = speaker();
    s.play(0);
    s.send(20, 'control', { type: 'cancel' });
    s.recv(60, { type: 'gate', at: 30 });
    expect(s.sent.at(-1)).toEqual({ cmd: 'flush' });
    expect(s.of('playback.paused')).toEqual([]);
  });
});

describe('page gate: record and replay', () => {
  it('replays a session with page stops bit-for-bit', async () => {
    // A page that stops playback 30 ms into each armed window.
    const speaker: LiveDriver = (_req, sink, env) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      setTimeout(() => sink.recv({ type: 'ready' }), 1);
      return {
        send(d) {
          const c = d as { cmd: string; until?: number | null };
          if (c.cmd !== 'gate') return;
          clearTimeout(timer);
          if (c.until != null) timer = setTimeout(() => sink.recv({ type: 'gate', at: env.now() - 10, pos: 0 }), 30);
        },
        close: () => clearTimeout(timer),
      };
    };
    const c = testConfig();
    c.turn.params['full-duplex'] = { yieldWithinMs: 1000, pageGateWaitMs: 50 };
    // Plain replies: the fake session's timing has the agent start before the user's next words.
    c.llm.params.api = { utterances: 'plain' };
    const scheduler = new LiveScheduler();
    const rt = new Runtime({ graph: buildGraph(c), registry, scheduler, world: new LiveWorld({ ...fakeDrivers, speaker }, scheduler), mode: 'live' });
    rt.start();
    await new Promise((r) => setTimeout(r, 1500));
    rt.stop();
    const live = rt.trajectory;
    const gated = live.events.filter((e) => e.kind === 'msg' && (e.data as { type: string; gate?: boolean }).type === 'playback.paused' && (e.data as { gate?: boolean }).gate);
    expect(gated.length).toBeGreaterThan(0);
    const { runtime, sim } = prepareReplay(decodeBundle(encodeBundle(live)), { registry });
    sim.runFast();
    expect(runtime.trajectory.events).toEqual(live.events);
    expect(runtime.trajectory.events.filter((e) => e.kind === 'io.diverge')).toHaveLength(0);
  }, 10_000);
});
