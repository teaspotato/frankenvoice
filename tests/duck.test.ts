import { describe, expect, it } from 'vitest';
import { speakerNode } from '../src/nodes/speaker';
import { resolveParams, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

describe('speaker ducking', () => {
  function speaker() {
    let now = 0;
    const out: Msg[] = [];
    const sent: Json[] = [];
    const ctx = {
      id: 'speaker',
      params: resolveParams(speakerNode, {}),
      now: () => now,
      emit: (_port: string, msg: Msg) => (out.push(msg), out.length),
      setState() {},
      log() {},
      setTimer: () => ({ clear() {} }),
      io: { open: () => ({ xid: '1', send: (d: Json) => void sent.push(d), close() {}, closed: false }) },
    } as unknown as NodeContext;
    const node = speakerNode.create(ctx as never);
    node.start?.();
    const play = (t: number) => {
      now = t;
      node.onInput?.('audio', { type: 'audio', track: 'tts.audio', start: 0, end: 2400, sampleRate: 24000, samples: new Float32Array(2400), rid: 'r0', seg: 0 } as never);
    };
    const control = (t: number, msg: Msg) => ((now = t), node.onInput?.('control', msg));
    const gains = () => sent.filter((d) => (d as { cmd: string }).cmd === 'gain').map((d) => (d as { gain: number }).gain);
    return { play, control, gains, out };
  }

  it('turns the agent down while ducked and back up after', () => {
    const s = speaker();
    s.play(0);
    s.control(20, { type: 'duck', db: 12 });
    s.control(60, { type: 'unduck' });
    expect(s.gains().map((g) => +g.toFixed(3))).toEqual([0.251, 1]);
    expect(s.out.filter((m) => m.type === 'playback.gain').map((m) => m.at)).toEqual([20, 60]);
  });

  it('restores the volume when the reply is cancelled, and ignores a duck with nothing playing', () => {
    const s = speaker();
    s.control(0, { type: 'duck', db: 12 });
    expect(s.gains()).toEqual([]);
    s.play(10);
    s.control(20, { type: 'duck', db: 12 });
    s.control(30, { type: 'cancel' });
    expect(s.gains().map((g) => +g.toFixed(3))).toEqual([0.251, 1]);
  });
});
