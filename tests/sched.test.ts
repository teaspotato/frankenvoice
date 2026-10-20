import { describe, expect, it } from 'vitest';
import { schedNode } from '../src/nodes/sched';
import { llmNode } from '../src/nodes/llm';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function harness(def: NodeDef, params: Record<string, Json> = {}) {
  let now = 0;
  const out: { port: string; msg: Msg }[] = [];
  const ios: { request: Json; h: IOHandlers; closed: boolean }[] = [];
  const timers: { at: number; fn: () => void; done: boolean }[] = [];
  const ctx: NodeContext = {
    id: def.type,
    params: resolveParams(def, params),
    now: () => now,
    cause: undefined,
    emit: (port, msg) => (out.push({ port, msg }), out.length),
    setState() {},
    log() {},
    setTimer(delay, fn) {
      const t = { at: now + delay, fn, done: false };
      timers.push(t);
      return { clear: () => void (t.done = true) };
    },
    io: {
      open(_channel, request, h) {
        const x = { request, h, closed: false };
        ios.push(x);
        return { xid: String(ios.length), send() {}, close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: { append: (track, samples) => ({ track, start: 0, end: samples.length }) },
  };
  const node = def.create(ctx as never);
  node.start?.();
  const send = (port: string, msg: Msg) => node.onInput?.(port, msg);
  /** Run the timers due up to `t`. */
  const until = (t: number) => {
    for (;;) {
      const next = timers.filter((x) => !x.done && x.at <= t).sort((a, b) => a.at - b.at)[0];
      if (!next) break;
      next.done = true;
      now = next.at;
      next.fn();
    }
    now = t;
  };
  const of = (type: string) => out.filter((o) => o.msg.type === type).map((o) => o.msg);
  return { send, until, out, of, ios, at: (t: number) => (now = t) };
}

const audio = (rid: string, ms: number, at?: number): Msg => ({
  type: 'audio',
  track: 'tts.audio',
  start: 0,
  end: 0,
  sampleRate: 1000,
  samples: new Float32Array(ms),
  rid,
  ...(at !== undefined ? { at } : {}),
});

describe('tick scheduler', () => {
  it('bursts to the lookahead, holds for the first audio, then paces by the TTS queue and playback', () => {
    const h = harness(schedNode);
    expect(h.of('sched.config')).toEqual([{ type: 'sched.config', pace: true, firstTokens: 16 }]);
    h.send('text', { type: 'llm.start', rid: 'r' });
    h.until(0);
    expect(h.of('sched.tick')).toHaveLength(1);
    // The first request stopped at its limit with two words for TTS: short of the lookahead.
    h.send('audio', { type: 'tts.sent', rid: 'r', words: 2 });
    h.send('text', { type: 'llm.chunk', rid: 'r', stop: 'length' });
    h.until(80);
    expect(h.of('sched.grant')).toEqual([{ type: 'sched.grant', rid: 'r', tokens: 8 }]);
    // Lookahead reached, no audio yet: hold.
    h.send('audio', { type: 'tts.sent', rid: 'r', words: 3 });
    h.send('text', { type: 'llm.chunk', rid: 'r', stop: 'length' });
    h.until(160);
    expect(h.of('sched.grant')).toHaveLength(1);
    expect(h.of('sched.tick').at(-1)).toMatchObject({ llm: { decision: 'hold', why: 'first audio first' }, tts: { queuedWords: 5 } });
    // Audio out, but not far enough ahead of playback.
    h.send('audio', audio('r', 200));
    h.send('played', audio('r', 200, 160));
    h.send('audio', { type: 'tts.consumed', rid: 'r', words: 1, queued: 4 });
    h.until(240);
    expect(h.of('sched.tick').at(-1)).toMatchObject({ llm: { decision: 'hold', why: 'TTS not far enough ahead' }, tts: { bufferedMs: 120, queuedWords: 4 } });
    // Far enough ahead: a chunk.
    h.send('played', audio('r', 400, 360));
    h.until(320);
    expect(h.of('sched.grant')).toHaveLength(2);
    expect(h.of('sched.tick').at(-1)).toMatchObject({ llm: { decision: 'grant', why: 'TTS ahead', granted: 8, pending: 8 } });
    // The ticks stay on the 80 ms grid and stop once the reply has been written, spoken and played.
    h.send('text', { type: 'llm.end', rid: 'r' });
    h.send('audio', { type: 'tts.response_end', rid: 'r' });
    h.until(2000);
    const ticks = h.of('sched.tick');
    expect(ticks.map((t) => t.ms)).toEqual(ticks.map((_, i) => i * 80));
    expect(ticks.at(-1)!.ms).toBe(800);
  });

  it('grants when playback is about to run dry, however many words TTS holds', () => {
    const h = harness(schedNode);
    h.send('text', { type: 'llm.start', rid: 'r' });
    h.send('audio', { type: 'tts.sent', rid: 'r', words: 9 });
    h.send('text', { type: 'llm.chunk', rid: 'r', stop: 'length' });
    h.send('audio', audio('r', 200));
    h.send('played', audio('r', 200, 0));
    h.until(80);
    expect(h.of('sched.grant')).toEqual([]);
    h.until(160);
    expect(h.of('sched.tick').at(-1)).toMatchObject({ llm: { decision: 'grant', why: 'playback about to run dry' }, tts: { queuedWords: 9, bufferedMs: 40 } });
  });

  it('only logs ticks when pacing is off', () => {
    const h = harness(schedNode, { pace: false });
    h.send('text', { type: 'llm.start', rid: 'r' });
    h.send('text', { type: 'llm.chunk', rid: 'r', stop: 'length' });
    h.until(400);
    expect(h.of('sched.grant')).toEqual([]);
    expect(h.of('sched.tick').every((t) => (t.llm as { decision: string }).decision === 'free')).toBe(true);
  });
});

describe('paced LLM', () => {
  const chunk = (text: string) => ({ choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
  const stop = (why: string) => ({ choices: [{ index: 0, delta: {}, finish_reason: why }] });

  it('writes a reply in granted chunks, continuing its text as the assistant turn', () => {
    const h = harness(llmNode, { system: '' });
    h.send('sched', { type: 'sched.config', pace: true, firstTokens: 4 });
    h.send('turn', { type: 'turn.user', turnId: 'u0', text: 'Hi' });
    const body = (i: number) => (h.ios[i].request as { body: { max_tokens: number; messages: Json[] } }).body;
    expect(body(0).max_tokens).toBe(4);
    for (const t of ['Hello', ' there', ',', ' friend ']) h.ios[0].h.onRecv!(chunk(t), undefined);
    h.ios[0].h.onRecv!(stop('length'), undefined);
    h.ios[0].h.onRecv!({ $done: true }, undefined);
    expect(h.of('llm.chunk')).toEqual([{ type: 'llm.chunk', rid: 'llm-r0', n: 0, tokens: 4, stop: 'length' }]);
    expect(h.of('llm.end')).toEqual([]);
    h.send('sched', { type: 'sched.grant', rid: 'llm-r0', tokens: 8 });
    // The text so far, without its trailing space (llama.cpp would take it as a token of its own).
    expect(body(1)).toMatchObject({ max_tokens: 8, messages: [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello there, friend' }] });
    h.ios[1].h.onRecv!(chunk(' how'), undefined);
    h.ios[1].h.onRecv!(chunk(' are you?'), undefined);
    h.ios[1].h.onRecv!(stop('stop'), undefined);
    h.ios[1].h.onRecv!({ $done: true }, undefined);
    expect(h.of('llm.delta').map((d) => d.text).join('')).toBe('Hello there, friend how are you?');
    expect(h.of('llm.end')).toMatchObject([{ status: 'done', text: 'Hello there, friend how are you?' }]);
    // A grant that comes late does nothing.
    h.send('sched', { type: 'sched.grant', rid: 'llm-r0', tokens: 8 });
    expect(h.ios).toHaveLength(2);
  });

  it('caps a draft at the first burst and paces it once it is kept', () => {
    const h = harness(llmNode, { system: '' });
    h.send('sched', { type: 'sched.config', pace: true, firstTokens: 2 });
    h.send('turn', { type: 'turn.draft', draftId: 'd0', text: 'Hi' });
    expect((h.ios[0].request as { body: { max_tokens: number } }).body.max_tokens).toBe(2);
    h.ios[0].h.onRecv!(chunk('Hello'), undefined);
    h.ios[0].h.onRecv!(chunk(' there'), undefined);
    h.ios[0].h.onRecv!(stop('length'), undefined);
    h.ios[0].h.onRecv!({ $done: true }, undefined);
    h.send('turn', { type: 'turn.user', turnId: 'u0', text: 'Hi', draft: 'd0' });
    expect(h.of('llm.start')).toMatchObject([{ draft: 'd0' }]);
    expect(h.of('llm.chunk')).toMatchObject([{ stop: 'length' }]);
    h.send('sched', { type: 'sched.grant', rid: 'llm-r0', tokens: 8 });
    expect((h.ios[1].request as { body: { messages: Json[] } }).body.messages.at(-1)).toEqual({ role: 'assistant', content: 'Hello there' });
  });
});
