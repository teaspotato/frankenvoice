import { describe, expect, it } from 'vitest';
import { cutAt, streamingTtsNode } from '../src/nodes/tts/stream';
import { resolveParams, type IOHandlers, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function harness(params: Record<string, unknown> = {}) {
  let now = 0;
  const out: Msg[] = [];
  const ios: { request: Json; h: IOHandlers; closed: boolean; sent: Json[] }[] = [];
  let media = 0;
  const ctx: NodeContext = {
    id: 'tts',
    params: resolveParams(streamingTtsNode, params as never),
    now: () => now,
    cause: undefined,
    emit: (_port, msg) => (out.push(msg), out.length),
    setState() {},
    log() {},
    setTimer: () => ({ clear() {} }),
    io: {
      open(_channel, request, h) {
        const x = { request, h, closed: false, sent: [] as Json[] };
        ios.push(x);
        return { xid: String(ios.length), send: (d) => void x.sent.push(d), close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: {
      append: (track, samples) => ({ track, start: media, end: (media += samples.length) }),
    },
  };
  const node = streamingTtsNode.create(ctx as never);
  const send = (t: number, port: string, msg: Msg) => ((now = t), node.onInput?.(port, msg));
  const llm = (t: number, type: string, text = '') => send(t, 'text', { type: `llm.${type}`, rid: 'r0', text });
  const pcm = (n: number) => new Uint8Array(n * 2);
  const types = () => out.map((m) => m.type);
  return { send, llm, ios, out, pcm, types };
}

describe('streaming TTS (Deepgram WebSocket)', () => {
  it('cuts at sentence ends, and at the first clause of a reply once it is long enough', () => {
    expect(cutAt('Hey, nice', 20)).toBe(-1);
    expect(cutAt('Sure, I can help with that, ', 20)).toBe('Sure, I can help with that, '.length);
    expect(cutAt('Hi there. How', 0)).toBe('Hi there. '.length);
    expect(cutAt('It costs 3.', 0)).toBe(-1);
  });

  it('opens the socket when the LLM starts, then speaks and flushes as text streams in', () => {
    const h = harness();
    h.llm(0, 'start');
    expect(h.ios).toHaveLength(1);
    expect(h.ios[0].request).toEqual({ path: '/v1/speak', query: { model: 'aura-2-thalia-en', encoding: 'linear16', sample_rate: 24000 } });
    h.llm(100, 'delta', 'Sure, I can help with that');
    expect(h.ios[0].sent).toEqual([]);
    h.llm(150, 'delta', ', and more. Then');
    expect(h.ios[0].sent).toEqual([{ type: 'Speak', text: 'Sure, I can help with that, and more.' }, { type: 'Flush' }]);
    h.ios[0].h.onRecv!(null, h.pcm(100));
    h.ios[0].h.onRecv!(null, new Uint8Array(3)); // odd bytes carry over
    h.ios[0].h.onRecv!(null, new Uint8Array(1));
    h.llm(300, 'delta', ' bye.');
    h.llm(310, 'end');
    expect(h.ios[0].sent.slice(2)).toEqual([{ type: 'Speak', text: 'Then bye.' }, { type: 'Flush' }]);
    h.ios[0].h.onRecv!({ type: 'Flushed', sequence_id: 0 }, undefined);
    h.ios[0].h.onRecv!(null, h.pcm(50));
    h.ios[0].h.onRecv!({ type: 'Flushed', sequence_id: 1 }, undefined);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'audio', 'audio', 'tts.segment_end', 'tts.segment_start', 'audio', 'tts.segment_end', 'tts.response_end']);
    expect(h.out.filter((m) => m.type === 'audio').map((m) => (m.samples as Float32Array).length)).toEqual([100, 1, 1, 50]);
    expect(h.out[0]).toMatchObject({ rid: 'r0', seg: 0, text: 'Sure, I can help with that, and more.' });
    expect(h.ios[0].sent.at(-1)).toEqual({ type: 'Close' });
    expect(h.ios[0].closed).toBe(true);
  });

  it('speaks an opener at once, and the reply goes on the same socket after it', () => {
    const h = harness();
    h.llm(0, 'start');
    h.send(100, 'text', { type: 'llm.delta', rid: 'r0', text: 'Hmm, let me see ', opener: true });
    expect(h.ios[0].sent).toEqual([{ type: 'Speak', text: 'Hmm, let me see' }, { type: 'Flush' }]);
    h.llm(900, 'delta', 'It is 391. ');
    expect(h.ios).toHaveLength(1);
    expect(h.ios[0].sent.slice(2)).toEqual([{ type: 'Speak', text: 'It is 391.' }, { type: 'Flush' }]);
  });

  it('synthesizes canned opener lines ahead and plays one without a request', () => {
    const h = harness();
    h.send(0, 'text', { type: 'llm.opener_lines', lines: ['Hmm, let me think.', 'Let me see.'] });
    const pre = h.ios[0];
    expect(pre.sent).toEqual([{ type: 'Speak', text: 'Hmm, let me think.' }, { type: 'Flush' }, { type: 'Speak', text: 'Let me see.' }, { type: 'Flush' }]);
    pre.h.onRecv!(null, h.pcm(30));
    pre.h.onRecv!({ type: 'Flushed', sequence_id: 0 }, undefined);
    pre.h.onRecv!(null, h.pcm(20));
    pre.h.onRecv!({ type: 'Flushed', sequence_id: 1 }, undefined);
    expect(pre.closed).toBe(true);
    expect(h.out).toEqual([]);
    h.llm(1000, 'start');
    h.send(1000, 'text', { type: 'llm.delta', rid: 'r0', text: 'Hmm, let me think. ', opener: true });
    expect(h.ios[1].sent).toEqual([]);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.segment_end']);
    expect((h.out[1].samples as Float32Array).length).toBe(30);
    h.llm(1500, 'delta', 'Because of scattering. ');
    expect(h.ios[1].sent).toEqual([{ type: 'Speak', text: 'Because of scattering.' }, { type: 'Flush' }]);
    expect(h.out.at(-1)).toMatchObject({ type: 'tts.segment_end', seg: 0 });
    h.ios[1].h.onRecv!(null, h.pcm(5));
    expect(h.out.at(-2)).toMatchObject({ type: 'tts.segment_start', seg: 1 });
  });

  it('stops on cancel and drops audio still in flight', () => {
    const h = harness();
    h.llm(0, 'start');
    h.llm(100, 'delta', 'Hello there. ');
    h.ios[0].h.onRecv!(null, h.pcm(10));
    h.send(200, 'control', { type: 'cancel' });
    h.ios[0].h.onRecv!(null, h.pcm(10));
    expect(h.ios[0].closed).toBe(true);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.cancelled']);
  });

  it('holds sentences back past the flush limit and sends them with the next flush', () => {
    const h = harness({ maxFlushesPerMin: 1 });
    h.llm(0, 'start');
    h.llm(100, 'delta', 'One. ');
    h.llm(200, 'delta', 'Two. ');
    h.llm(300, 'delta', 'Three.');
    h.llm(400, 'end');
    expect(h.ios[0].sent).toEqual([
      { type: 'Speak', text: 'One.' },
      { type: 'Flush' },
      { type: 'Speak', text: 'Two. Three.' },
      { type: 'Flush' },
    ]);
  });
});
