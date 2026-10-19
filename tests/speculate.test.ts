import { describe, expect, it } from 'vitest';
import { duplexTurnNode } from '../src/nodes/turn';
import { llmNode } from '../src/nodes/llm';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

/** Drives one node by hand: a manual clock, timers, fake IO, and a list of what it emitted. */
function harness(def: NodeDef, params: Record<string, unknown> = {}) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ios: { request: Json; h: IOHandlers; closed: boolean; sent: Json[] }[] = [];
  const ctx: NodeContext = {
    id: def.type === 'llm' ? 'llm' : def.type === 'asr' ? 'asr' : 'turn',
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
      open(_channel, request, h) {
        const x = { request, h, closed: false, sent: [] as Json[] };
        ios.push(x);
        return { xid: String(ios.length), send: (d) => void x.sent.push(d), close: () => void (x.closed = true), get closed() { return x.closed; } };
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
  const of = (type: string) => out.filter((o) => o.msg.type === type).map((o) => o.msg);
  return { send, advance, out, of, ios };
}

const sr = 16;
const vad = (on: boolean, t: number) => ({ type: on ? 'vad.speech_start' : 'vad.speech_end', at: t * sr });
const asr = (final: boolean, t: number, text: string, speechFinal = false) => ({
  type: final ? 'asr.final' : 'asr.partial',
  text,
  start: (t - 800) * sr,
  end: t * sr,
  ...(final ? { speechFinal } : {}),
});

describe('drafting replies before the turn is committed', () => {
  it('drafts once ASR catches up with the silence, and keeps it when the final words match', () => {
    const h = harness(duplexTurnNode);
    h.send(0, 'vad', vad(true, 0));
    h.send(800, 'transcript', asr(false, 800, 'what about'));
    h.send(1000, 'vad', vad(false, 1000));
    // The partial from before the silence is stale: no draft yet.
    expect(h.of('turn.draft')).toHaveLength(0);
    h.send(1200, 'transcript', asr(false, 1200, 'What about Osaka?'));
    expect(h.of('turn.draft')).toMatchObject([{ draftId: 'turn-d0', text: 'What about Osaka?' }]);
    // The same words again do not redraft; the final, with other case and punctuation, keeps it.
    h.send(1400, 'transcript', asr(false, 1400, 'What about Osaka?'));
    h.send(1600, 'transcript', asr(true, 1600, 'what about Osaka', true));
    expect(h.of('turn.draft')).toHaveLength(1);
    expect(h.of('turn.draft_drop')).toHaveLength(0);
    expect(h.of('turn.user')).toMatchObject([{ text: 'what about Osaka', draft: 'turn-d0' }]);
  });

  it('redrafts when the words change in the silence, and drops it when the user goes on', () => {
    const h = harness(duplexTurnNode);
    h.send(0, 'vad', vad(true, 0));
    h.send(800, 'transcript', asr(true, 800, 'book a table', false));
    h.send(1000, 'vad', vad(false, 1000));
    // All the words were final before the silence: draft right away.
    expect(h.of('turn.draft')).toMatchObject([{ draftId: 'turn-d0', text: 'book a table' }]);
    h.send(1300, 'transcript', asr(false, 1300, 'for two'));
    expect(h.of('turn.draft_drop')).toMatchObject([{ draftId: 'turn-d0', why: 'words changed' }]);
    expect(h.of('turn.draft').at(-1)).toMatchObject({ draftId: 'turn-d1', text: 'book a table for two' });
    h.send(1500, 'vad', vad(true, 1500));
    h.send(1900, 'transcript', asr(false, 1900, 'for two at seven'));
    expect(h.of('turn.draft_drop').at(-1)).toMatchObject({ draftId: 'turn-d1', why: 'user went on' });
    h.send(2000, 'vad', vad(false, 2000));
    h.send(2300, 'transcript', asr(true, 2300, 'for two at seven please', true));
    expect(h.of('turn.user')).toMatchObject([{ text: 'book a table for two at seven please' }]);
    expect(h.of('turn.user')[0].draft).toBeUndefined();
  });

  it('drops the draft at commit when the final words differ', () => {
    const h = harness(duplexTurnNode);
    h.send(0, 'vad', vad(true, 0));
    h.send(1000, 'vad', vad(false, 1000));
    h.send(1200, 'transcript', asr(false, 1200, "I'm in talk"));
    h.send(1400, 'transcript', asr(true, 1400, "I'm in Tokyo.", true));
    expect(h.of('turn.draft_drop')).toMatchObject([{ draftId: 'turn-d0', why: 'final words differ' }]);
    expect(h.of('turn.user')[0].draft).toBeUndefined();
  });

  it('ignores fillers when matching', () => {
    const h = harness(duplexTurnNode);
    h.send(0, 'vad', vad(true, 0));
    h.send(1000, 'vad', vad(false, 1000));
    h.send(1200, 'transcript', asr(false, 1200, 'so, what now'));
    h.send(1400, 'transcript', asr(true, 1400, 'So, um, what now?', true));
    expect(h.of('turn.user')).toMatchObject([{ draft: 'turn-d0' }]);
  });

  it('with the turn model, drafts only once its end-of-turn score reaches draftAt', () => {
    const h = harness(duplexTurnNode, { draftAt: 0.3 });
    const frame = (t: number, eot: number) => h.send(t, 'scores', { type: 'turnmodel.frame', at: t * sr, eot: [eot, 0], int: [0, 0], silent: [0, 0] });
    h.send(0, 'scores', { type: 'turnmodel.status', status: 'ready' });
    h.send(0, 'vad', vad(true, 0));
    h.send(1000, 'vad', vad(false, 1000));
    h.send(1300, 'transcript', asr(true, 1300, 'tell me about Kyoto', false));
    frame(1400, 0.1);
    expect(h.of('turn.draft')).toHaveLength(0);
    frame(1500, 0.35);
    expect(h.of('turn.draft')).toMatchObject([{ text: 'tell me about Kyoto' }]);
    h.send(2600, 'scores', { type: 'turnmodel.eot', who: 'user', at: 2600 * sr, p: 0.7 });
    expect(h.of('turn.user')).toMatchObject([{ text: 'tell me about Kyoto', draft: 'turn-d0' }]);
  });

  it('does not draft when turned off', () => {
    const h = harness(duplexTurnNode, { speculate: false });
    h.send(1000, 'vad', vad(false, 1000));
    h.send(1200, 'transcript', asr(false, 1200, 'hello'));
    expect(h.of('turn.draft')).toHaveLength(0);
  });
});

describe('the LLM on drafts', () => {
  const params = { provider: 'openai-compatible', model: 'm', system: 'sys' };
  const tok = (text: string) => ({ choices: [{ index: 0, delta: { content: text } }] });
  const messages = (r: Json) => (r as { body: { messages: { role: string; content: string }[] } }).body.messages.map((m) => m.content);

  it('holds a draft’s stream back and releases it when the turn keeps the draft', () => {
    const h = harness(llmNode, params);
    h.send(1000, 'turn', { type: 'turn.draft', draftId: 'd0', text: 'What about Osaka?' });
    expect(h.ios).toHaveLength(1);
    expect(messages(h.ios[0].request)).toEqual(['sys', 'What about Osaka?']);
    h.ios[0].h.onRecv!(tok('Osaka is'), undefined);
    expect(h.of('llm.delta')).toHaveLength(0);
    h.send(1500, 'turn', { type: 'turn.user', turnId: 'u0', text: 'what about Osaka', draft: 'd0' });
    expect(h.ios).toHaveLength(1);
    expect(h.of('llm.start')).toMatchObject([{ rid: 'llm-r0', turnId: 'u0', draft: 'd0', headStartMs: 500 }]);
    expect(h.of('llm.delta')).toMatchObject([{ rid: 'llm-r0', text: 'Osaka is' }]);
    h.ios[0].h.onRecv!(tok(' lovely.'), undefined);
    h.ios[0].h.onRecv!({ $done: true }, undefined);
    expect(h.of('llm.end')).toMatchObject([{ rid: 'llm-r0', status: 'done', text: 'Osaka is lovely.' }]);
    // The next request carries the turn as the draft sent it.
    h.send(3000, 'turn', { type: 'turn.user', turnId: 'u1', text: 'thanks' });
    expect(messages(h.ios[1].request)).toEqual(['sys', 'What about Osaka?', 'Osaka is lovely.', 'thanks']);
  });

  it('discards a draft the turn does not keep, leaving no trace in the history', () => {
    const h = harness(llmNode, params);
    h.send(1000, 'turn', { type: 'turn.draft', draftId: 'd0', text: "I'm in talk" });
    h.ios[0].h.onRecv!(tok('Talk?'), undefined);
    h.send(1200, 'turn', { type: 'turn.user', turnId: 'u0', text: "I'm in Tokyo." });
    expect(h.ios[0].closed).toBe(true);
    expect(h.of('llm.draft_end')).toMatchObject([{ draftId: 'd0', status: 'dropped' }]);
    expect(messages(h.ios[1].request)).toEqual(['sys', "I'm in Tokyo."]);
    // Late tokens from the dropped draft go nowhere.
    h.ios[0].h.onRecv!(tok(' more'), undefined);
    expect(h.of('llm.delta')).toHaveLength(0);
  });

  it('drafts a merged turn against the history before the turn it replaces', () => {
    const h = harness(llmNode, params);
    h.send(1000, 'turn', { type: 'turn.user', turnId: 'u0', text: 'how about' });
    h.ios[0].h.onRecv!(tok('About what?'), undefined);
    h.send(1200, 'control', { type: 'cancel' });
    h.send(2000, 'turn', { type: 'turn.draft', draftId: 'd0', text: 'how about New York', replaces: 'u0' });
    expect(messages(h.ios[1].request)).toEqual(['sys', 'how about New York']);
    h.send(2400, 'turn', { type: 'turn.user', turnId: 'u1', text: 'how about New York', replaces: 'u0', draft: 'd0' });
    h.send(2500, 'turn', { type: 'turn.draft', draftId: 'd1', text: 'and Boston' });
    // No reply came yet, so the adapter merges the two user turns; "how about" is gone.
    expect(messages(h.ios[2].request)).toEqual(['sys', 'how about New York\nand Boston']);
  });
});

describe('sessions recorded before drafts', () => {
  it('replay without them', async () => {
    const { migrateGraph } = await import('../src/nodes');
    const g = migrateGraph({ format: 'frankenvoice-graph', formatVersion: 2, name: 'old', nodes: [{ id: 'turn', type: 'turn.duplex', params: {} }], edges: [] });
    expect(g.nodes[0].params).toEqual({ speculate: false });
  });
});

describe('finalizing ASR on VAD speech end', () => {
  it('commits a final flushed after the VAD heard the end, without waiting for ASR endpointing', () => {
    const h = harness(duplexTurnNode);
    h.send(0, 'vad', vad(true, 0));
    h.send(800, 'transcript', asr(false, 800, 'what about'));
    h.send(1000, 'vad', vad(false, 1000));
    h.send(1150, 'transcript', { ...asr(true, 1150, 'what about Osaka?'), fromFinalize: true });
    expect(h.of('turn.user')).toMatchObject([{ text: 'what about Osaka?' }]);
  });

  it('asks the ASR provider to finalize on speech end', async () => {
    const { asrNode } = await import('../src/nodes/asr');
    const h = harness(asrNode);
    h.send(0, 'audio', { type: 'audio', track: 'mic', start: 0, end: 160, sampleRate: 16000, samples: new Float32Array(160) });
    h.send(500, 'vad', vad(false, 500));
    expect(h.ios[0].sent.filter((d) => d !== null)).toEqual([{ type: 'Finalize' }]);
  });
});

describe('turn-taking fixes from the smoothie session', () => {
  const ready = { type: 'turnmodel.status', status: 'ready' };
  const eot = (t: number) => ({ type: 'turnmodel.eot', who: 'user', at: t * sr, p: 0.65 });
  const int = (t: number) => ({ type: 'turnmodel.int', who: 'user', at: t * sr, p: 0.33 });

  it('does not end the turn on the model while the VAD still hears speech, unless the speech ends right after', () => {
    const h = harness(duplexTurnNode);
    h.send(0, 'scores', ready);
    h.send(0, 'vad', vad(true, 0));
    h.send(500, 'transcript', asr(false, 500, 'I was thinking'));
    h.send(900, 'scores', eot(900));
    h.send(2000, 'vad', vad(false, 2000)); // 1.1 s later: they were not done
    h.advance(3000);
    expect(h.of('turn.user')).toHaveLength(0);
    h.send(3000, 'vad', vad(true, 3000));
    h.send(3500, 'scores', eot(3500));
    h.send(3800, 'vad', vad(false, 3800)); // within the VAD's hangover: it stands
    h.send(3900, 'transcript', asr(true, 3900, 'I was thinking about Osaka', true));
    expect(h.of('turn.user')).toMatchObject([{ text: 'I was thinking about Osaka' }]);
  });

  it('ignores ASR utterance end while the VAD still hears speech', () => {
    const h = harness(duplexTurnNode, { useModel: false, commitOn: 'utterance_end' });
    h.send(0, 'vad', vad(true, 0));
    h.send(800, 'transcript', asr(true, 800, 'I was thinking'));
    h.send(1000, 'transcript', { type: 'asr.utterance_end', at: 1000 * sr });
    expect(h.of('turn.user')).toHaveLength(0);
  });

  it('does not repeat words when a turn committed on a partial is resent', () => {
    const h = harness(duplexTurnNode, { asrGraceMs: 600 });
    h.send(0, 'scores', ready);
    h.send(0, 'vad', vad(true, 0));
    h.send(1000, 'transcript', asr(false, 1000, 'I was thinking'));
    h.send(1100, 'vad', vad(false, 1100));
    h.send(1200, 'scores', eot(1200));
    h.advance(1800); // no final in time: commits on the partial
    expect(h.of('turn.user')).toMatchObject([{ text: 'I was thinking' }]);
    h.send(1900, 'scores', int(1900)); // they go on: the reply is dropped
    h.send(2000, 'vad', vad(true, 2000));
    h.send(2100, 'transcript', { ...asr(true, 1100, 'I was thinking'), fromFinalize: true });
    h.send(2600, 'transcript', asr(false, 2600, 'about'));
    h.send(2700, 'vad', vad(false, 2700));
    h.send(2900, 'transcript', asr(true, 2900, 'about'));
    h.send(3000, 'scores', eot(3000));
    expect(h.of('turn.user').at(-1)).toMatchObject({ text: 'I was thinking about', replaces: 'turn-u0' });
  });
});
