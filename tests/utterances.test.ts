import { describe, expect, it } from 'vitest';
import { rawPrefix, tagGrammar, UtteranceParser, visibleText } from '../src/nodes/llm/utterances';
import { cutToHeard, llmNode } from '../src/nodes/llm';
import { kyutaiTtsNode } from '../src/nodes/tts/stream';
import { speakerNode } from '../src/nodes/speaker';
import { duplexTurnNode } from '../src/nodes/turn';
import type { NodeDef } from '../src/core/node';
import type { JsonObject, Msg } from '../src/core/types';
import { harness } from './node-harness';

const chunk = (content: string, finish: string | null = null) => ({ choices: [{ index: 0, delta: { content }, finish_reason: finish }] });
const body = (io: { request: unknown }) => (io.request as { body: JsonObject }).body;

describe('tagged utterances: parsing', () => {
  it('streams flags before the words, a line break as a space, whatever the chunking', () => {
    const raw = '[i1g0] Sure thing.\n[i0g1] It is K L 7 7.\nThanks';
    const whole = new UtteranceParser().feed(raw);
    const p = new UtteranceParser();
    const bits = [...raw].flatMap((c) => p.feed(c));
    const merge = (es: typeof whole) =>
      es.reduce<typeof whole>((acc, e) => {
        const prev = acc.at(-1);
        if (e.type === 'text' && prev?.type === 'text') prev.text += e.text;
        else acc.push({ ...e });
        return acc;
      }, []);
    expect(merge(bits)).toEqual(merge(whole));
    expect(merge(whole)).toEqual([
      { type: 'utterance', flags: { interruptible: true, gate: false }, tagged: true },
      { type: 'text', text: 'Sure thing.' },
      { type: 'utterance', flags: { interruptible: false, gate: true }, tagged: true },
      { type: 'text', text: ' It is K L 7 7.' },
      { type: 'utterance', flags: { interruptible: true, gate: true }, tagged: false },
      { type: 'text', text: ' Thanks' },
    ]);
    expect(visibleText(raw)).toBe('Sure thing. It is K L 7 7. Thanks');
    expect(visibleText('[listen]')).toBe('[listen]');
  });

  it('cuts a tagged text after its kth word, keeping the tags', () => {
    expect(rawPrefix('[i1g0] Sure thing.\n[i0g1] It is K L.', 3)).toBe('[i1g0] Sure thing.\n[i0g1] It');
    expect(rawPrefix('plain words here', 2)).toBe('plain words');
    const h = [
      { role: 'user' as const, text: 'hi', turnId: 't0' },
      { role: 'assistant' as const, text: '[i1g0] Sure thing.\n[i0g1] It is K L.', turnId: 't0', rid: 'r0' },
    ];
    expect(cutToHeard(h, [1], 'Sure thing. It')[1].text).toBe('[i1g0] Sure thing.\n[i0g1] It—');
  });

  it('writes the grammar for what follows any prefix', () => {
    const rules = 'line ::= "[i" [01] "g" [01] "] " t\nt ::= [^\\n\\[]+\n';
    expect(tagGrammar('[i')).toBe(`root ::= [01] "g" [01] "]" " " t ("\\n" line)*\n${rules}`);
    expect(tagGrammar('[i1g0]')).toBe(`root ::= " " t ("\\n" line)*\n${rules}`);
    expect(tagGrammar('[i1g0] Hello there')).toBe(`root ::= [^\\n\\[]* ("\\n" line)*\n${rules}`);
    expect(tagGrammar('[i1g0] Hi.\n')).toBe(`root ::= line ("\\n" line)*\n${rules}`);
  });
});

describe('tagged utterances: the LLM', () => {
  const llm = (params: Record<string, unknown> = {}) => harness(llmNode, { model: 'gemma', system: 'Be brief.', liveContext: false, ...params }, 'llm');

  it('prefills the first tag, sends the grammar, emits flags before words, and keeps the raw text in the history', () => {
    const h = llm();
    h.send(0, 'turn', { type: 'turn.user', text: 'What is my code?', turnId: 't0' });
    const b = body(h.ios[0]);
    const msgs = b.messages as JsonObject[];
    expect(String(msgs[0].content)).toMatch(/^Be brief\.\n\nWrite your reply as lines/);
    expect(msgs.at(-1)).toEqual({ role: 'assistant', content: '[i' });
    expect(b.grammar).toBe(tagGrammar('[i'));
    expect(b.id_slot).toBe(0);
    for (const c of ['1g0] ', 'Sure.', '\n[i', '0g0] ', 'It is 42.']) h.ios[0].h.onRecv?.(chunk(c), undefined);
    h.ios[0].h.onRecv?.(chunk('', 'stop'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
    const seq = h.out.filter((o) => ['llm.utterance', 'llm.delta'].includes(o.msg.type)).map((o) => (o.msg.type === 'llm.delta' ? o.msg.text : [o.msg.interruptible, o.msg.gate]));
    const merged = seq.reduce<unknown[]>((acc, x) => (typeof x === 'string' && typeof acc.at(-1) === 'string' ? [...acc.slice(0, -1), `${acc.at(-1)}${x}`] : [...acc, x]), []);
    expect(merged).toEqual([[true, false], 'Sure.', [false, false], ' It is 42.']);
    expect(h.of('llm.end')[0]).toMatchObject({ text: 'Sure. It is 42.', raw: '[i1g0] Sure.\n[i0g0] It is 42.' });
    h.send(100, 'turn', { type: 'turn.user', text: 'Thanks.', turnId: 't1' });
    expect((body(h.ios[1]).messages as JsonObject[]).slice(2)).toEqual([
      { role: 'assistant', content: '[i1g0] Sure.\n[i0g0] It is 42.' },
      { role: 'user', content: 'Thanks.' },
      { role: 'assistant', content: '[i' },
    ]);
  });

  it('continues a paced reply mid-tag under the grammar for the rest', () => {
    const h = llm();
    h.send(0, 'sched', { type: 'sched.config', pace: true, firstTokens: 2 });
    h.send(0, 'turn', { type: 'turn.user', text: 'Hi', turnId: 't0' });
    h.ios[0].h.onRecv?.(chunk('1g0] '), undefined);
    h.ios[0].h.onRecv?.(chunk('Hello'), undefined);
    h.ios[0].h.onRecv?.(chunk('', 'length'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
    h.send(10, 'sched', { type: 'sched.grant', rid: 'llm-r0', tokens: 4 });
    const b = body(h.ios[1]);
    expect((b.messages as JsonObject[]).at(-1)).toEqual({ role: 'assistant', content: '[i1g0] Hello' });
    expect(b.grammar).toBe(tagGrammar('[i1g0] Hello'));
    h.ios[1].h.onRecv?.(chunk(' there.\n[i'), undefined);
    h.ios[1].h.onRecv?.(chunk('1g1] Bye.', 'stop'), undefined);
    h.ios[1].h.onRecv?.({ $done: true }, undefined);
    expect(h.of('llm.end')[0]).toMatchObject({ text: 'Hello there. Bye.', raw: '[i1g0] Hello there.\n[i1g1] Bye.' });
    expect(h.of('llm.utterance').map((m) => m.n)).toEqual([0, 1]);
  });

  it('keeps the tags of what was heard when the reply is cut off', () => {
    const h = llm();
    h.send(0, 'turn', { type: 'turn.user', text: 'Hi', turnId: 't0' });
    h.ios[0].h.onRecv?.(chunk('1g0] Hello there.\n[i1g1] How are you?', 'stop'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
    h.send(50, 'turn', { type: 'turn.heard', turnId: 't0', rid: 'llm-r0', text: 'Hello there. How' });
    h.send(100, 'turn', { type: 'turn.user', text: 'Fine.', turnId: 't1' });
    expect((body(h.ios[1]).messages as JsonObject[])[2]).toEqual({ role: 'assistant', content: '[i1g0] Hello there.\n[i1g1] How—' });
  });
});

describe('tagged utterances: TTS and the speaker', () => {
  it('ends a segment at each utterance and gives the next ones its flags', () => {
    const h = harness(kyutaiTtsNode, { alignWords: false }, 'tts');
    h.send(0, 'text', { type: 'llm.start', rid: 'r0' });
    h.send(1, 'text', { type: 'llm.utterance', rid: 'r0', n: 0, interruptible: true, gate: false });
    h.send(2, 'text', { type: 'llm.delta', rid: 'r0', text: 'Sure thing' });
    h.send(3, 'text', { type: 'llm.utterance', rid: 'r0', n: 1, interruptible: false, gate: true });
    h.send(4, 'text', { type: 'llm.delta', rid: 'r0', text: ' Your code is 42.' });
    expect(h.ios[0].sent).toContainEqual({ type: 'Flush' });
    h.ios[0].h.onRecv?.(null, new Uint8Array(960));
    h.ios[0].h.onRecv?.({ type: 'Flushed' }, undefined);
    h.ios[0].h.onRecv?.(null, new Uint8Array(960));
    expect(h.of('tts.segment_start')).toMatchObject([
      { seg: 0, text: 'Sure thing', interruptible: true, gate: false },
      { seg: 1, interruptible: false, gate: true },
    ]);
  });

  it('tells each played segment with its flags', () => {
    const h = harness(speakerNode as NodeDef, {}, 'speaker');
    h.send(0, 'audio', { type: 'tts.segment_start', rid: 'r0', seg: 0, text: 'Your code is 42.', interruptible: false, gate: false });
    expect(h.of('playback.segment')).toMatchObject([{ rid: 'r0', seg: 0, interruptible: false, gate: false }]);
  });
});

describe('tagged utterances: turn-taking', () => {
  const sr = 16;
  const turn = (params: Record<string, unknown> = {}) => {
    const h = harness(duplexTurnNode as NodeDef, { overlap: 'classify', eotCheck: 'off', yieldWithinMs: 1000, ...params }, 'turn');
    h.send(1000, 'transcript', { type: 'asr.final', text: 'what is my code', start: 500 * sr, end: 1000 * sr, speechFinal: true });
    return h;
  };
  const controls = (h: ReturnType<typeof turn>) => h.out.filter((o) => o.port === 'control').map((o) => o.msg);
  const vad = (h: ReturnType<typeof turn>, t: number, on: boolean) => h.send(t, 'vad', { type: on ? 'vad.speech_start' : 'vad.speech_end', at: t * sr } as Msg);

  it("doesn't arm the page gate or yield for an utterance said regardless", () => {
    const h = turn();
    h.send(1400, 'playback', { type: 'playback.segment', rid: 'r0', seg: 0, text: 'Okay.', at: 1500, interruptible: true, gate: false });
    h.send(1500, 'playback', { type: 'playback.start', rid: 'r0', at: 1500 });
    expect(controls(h).filter((c) => c.type === 'gate')).toEqual([]);
    vad(h, 1600, true);
    // Not a quick yield: speech over it is judged as usual.
    expect(h.of('turn.decision').map((d) => d.decision)).toEqual(['candidate']);
    expect(controls(h).filter((c) => c.type === 'cancel')).toEqual([]);
  });

  it('stops the quick-yield window where an utterance without it starts', () => {
    const h = turn();
    h.send(1400, 'playback', { type: 'playback.segment', rid: 'r0', seg: 0, text: 'Sure.', at: 1500, interruptible: true, gate: true });
    h.send(1500, 'playback', { type: 'playback.start', rid: 'r0', at: 1500 });
    h.send(1550, 'playback', { type: 'playback.segment', rid: 'r0', seg: 1, text: 'Your code is 42.', at: 1900, interruptible: false, gate: true });
    expect(controls(h).filter((c) => c.type === 'gate')).toEqual([
      { type: 'gate', until: 2500 },
      { type: 'gate', until: 1900 },
    ]);
  });

  it('lets the user talk over an utterance that must be heard, and counts their speech from the next one', () => {
    const h = turn({ yieldWithinMs: 0 });
    h.send(1400, 'playback', { type: 'playback.segment', rid: 'r0', seg: 0, text: 'Your code is 42.', at: 1500, interruptible: false, gate: false });
    h.send(1500, 'playback', { type: 'playback.start', rid: 'r0', at: 1500 });
    vad(h, 1700, true);
    expect(h.of('turn.decision').map((d) => d.decision)).toEqual(['protected']);
    h.advance(3000);
    expect(controls(h)).toEqual([]);
    expect(h.of('turn.transition').at(-1)?.to).toBe('speaking');
    // The next utterance may be cut off: still talking when it starts, so that's their onset.
    h.send(3100, 'playback', { type: 'playback.segment', rid: 'r0', seg: 1, text: 'Anything else?', at: 3400, interruptible: true, gate: true });
    h.advance(3400);
    expect(h.of('turn.decision').map((d) => d.decision)).toEqual(['protected', 'candidate']);
    expect(h.of('turn.transition').at(-1)?.to).toBe('speaking+user');
  });

  it('forgets speech over a protected utterance that ended before it', () => {
    const h = turn({ yieldWithinMs: 0 });
    h.send(1400, 'playback', { type: 'playback.segment', rid: 'r0', seg: 0, text: 'Your code is 42.', at: 1500, interruptible: false, gate: false });
    h.send(1500, 'playback', { type: 'playback.start', rid: 'r0', at: 1500 });
    vad(h, 1700, true);
    vad(h, 2000, false);
    h.send(2100, 'playback', { type: 'playback.segment', rid: 'r0', seg: 1, text: 'Anything else?', at: 2400, interruptible: true, gate: true });
    h.advance(3000);
    expect(h.of('turn.decision').map((d) => d.decision)).toEqual(['protected']);
    expect(controls(h)).toEqual([]);
  });
});
