import { describe, expect, it } from 'vitest';
import { asrNode } from '../src/nodes/asr';
import { resolveParams, type IOHandlers, type NodeContext } from '../src/core/node';
import type { AudioMsg, Json } from '../src/core/types';

function harness() {
  let now = 0;
  const conns: { handlers: IOHandlers; sent: [number, number][]; closed: boolean }[] = [];
  const logs: string[] = [];
  const ctx: NodeContext = {
    id: 'asr',
    params: resolveParams(asrNode, {}),
    now: () => now,
    cause: undefined,
    emit: () => 0,
    setState() {},
    log: (_l, m) => void logs.push(m),
    setTimer: () => ({ clear() {} }),
    io: {
      open(_ch, _req, handlers) {
        const c = { handlers, sent: [] as [number, number][], closed: false };
        conns.push(c);
        return {
          xid: `asr#${conns.length - 1}`,
          send: (_d: Json, o) => void (o?.media && c.sent.push([o.media.start, o.media.end])),
          close: () => void (c.closed = true),
          get closed() {
            return c.closed;
          },
        };
      },
    },
    media: { append: () => ({ track: 'x', start: 0, end: 0 }) },
  };
  const node = asrNode.create(ctx as never);
  const frame = (i: number) => {
    now = i * 32;
    const msg: AudioMsg = { type: 'audio', track: 'mic', start: i * 512, end: (i + 1) * 512, sampleRate: 16000, samples: new Float32Array(512) };
    node.onInput?.('audio', msg);
  };
  const result = (c: number, start: number, end: number, isFinal: boolean) =>
    conns[c].handlers.onRecv?.(
      { type: 'Results', is_final: isFinal, speech_final: isFinal, start, end, text: 'hi', words: [] },
      undefined,
    );
  return { conns, logs, frame, result, node };
}

describe('ASR connection', () => {
  it('reconnects after a drop and resends the audio since the last final result', () => {
    const h = harness();
    for (let i = 0; i < 100; i++) h.frame(i);
    h.result(0, 0, 1.6, true); // final up to 1.6 s = sample 25600 = frame 50
    for (let i = 100; i < 110; i++) h.frame(i);
    h.conns[0].handlers.onClose?.({ code: 1006, error: 'websocket closed (1006)' });
    expect(h.logs.at(-1)).toMatch(/reconnecting/);
    h.frame(110); // too soon: waits 500 ms
    expect(h.conns).toHaveLength(1);
    h.frame(130);
    expect(h.conns).toHaveLength(2);
    // Everything after the final result, up to now, goes out again first.
    expect(h.conns[1].sent[0]).toEqual([50 * 512, 51 * 512]);
    expect(h.conns[1].sent.at(-1)).toEqual([130 * 512, 131 * 512]);
  });

  it('gives up after repeated failures to connect', () => {
    const h = harness();
    h.frame(0);
    for (let k = 0; k < 6; k++) {
      h.conns.at(-1)!.handlers.onClose?.({ code: 1011, error: 'bad key' });
      for (let i = 1; i < 200; i++) h.frame(k * 200 + i);
    }
    expect(h.conns).toHaveLength(4);
    expect(h.logs.at(-1)).toMatch(/gave up/);
  });

  it('does not reconnect when the session stops', () => {
    const h = harness();
    h.frame(0);
    h.node.stop?.();
    h.conns[0].handlers.onClose?.(null);
    h.frame(100);
    expect(h.conns).toHaveLength(1);
  });
});

describe('GPU box ASR adapter', () => {
  it('maps the server messages onto ASR events', async () => {
    const { gpuboxAsrAdapter: a } = await import('../src/nodes/asr/gpubox');
    expect(a.buildRequest({ language: 'English', partialMs: 250, context: '' }, 16000)).toEqual({
      path: '/v1/asr',
      query: { language: 'English', sample_rate: 16000, partial_ms: 250 },
    });
    expect(a.parse({ type: 'SpeechStarted', at: 1.2 })).toEqual([{ type: 'speech_started', at: 1.2 }]);
    expect(
      a.parse({ type: 'Results', is_final: true, speech_final: true, from_finalize: true, start: 1, end: 2, text: 'hi there', words: [{ word: 'hi', start: 1.1, end: 1.3 }] }),
    ).toEqual([
      { type: 'final', text: 'hi there', confidence: null, speechFinal: true, fromFinalize: true, start: 1, end: 2, words: [{ word: 'hi', start: 1.1, end: 1.3 }] },
    ]);
    expect(a.parse({ type: 'Results', is_final: false, start: 0, end: 0.5, text: 'hi' })[0]).toMatchObject({ type: 'partial', words: [] });
    expect(a.finalizeMessage()).toEqual({ type: 'Finalize' });
  });
});
