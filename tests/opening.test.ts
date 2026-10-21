import { describe, expect, it } from 'vitest';
import { llmNode } from '../src/nodes/llm';
import { contextBlock } from '../src/nodes/llm/context';
import type { Json, JsonObject } from '../src/core/types';
import { duplexTurnNode } from '../src/nodes/turn';
import { harness } from './node-harness';

const chunk = (content: string, finish: string | null = null) => ({ choices: [{ index: 0, delta: { content }, finish_reason: finish }] });
const body = (io: { request: Json }) => (io.request as { body: JsonObject }).body;
const contents = (io: { request: Json }) => (body(io).messages as JsonObject[]).map((m) => `${m.role}: ${m.content}`);

describe('live context', () => {
  it("says when the call started in the user's time zone, and their language", () => {
    const at = Date.UTC(2026, 9, 21, 23, 30);
    expect(contextBlock({ epochMs: at, timeZone: 'America/New_York', locale: 'en-US' })).toBe(
      "Live context:\n- The call started on Wednesday, 21 October 2026 at 19:30, the user's local time (time zone America/New_York, GMT-04:00).\n- The user's browser language is en-US.",
    );
    // Past midnight in Tokyo; an unknown zone is UTC.
    expect(contextBlock({ epochMs: at, timeZone: 'Asia/Tokyo' })).toContain('Thursday, 22 October 2026 at 08:30');
    expect(contextBlock({ epochMs: at, timeZone: 'Nowhere/Land' })).toContain('at 23:30, the user\'s local time (time zone UTC, GMT)');
  });

  it('goes at the end of the system prompt once, and stays the same for every request', () => {
    const h = harness(llmNode, { model: 'gemma', system: 'Be brief.', utterances: 'plain', openPrompt: '(Open.)' }, 'llm');
    expect(h.ios[0].channel).toBe('clock');
    h.ios[0].h.onRecv?.({ epochMs: Date.UTC(2026, 9, 21, 11, 12), timeZone: 'Europe/Berlin', locale: 'de-DE' }, undefined);
    h.ios[0].h.onClose?.(null);
    h.send(0, 'turn', { type: 'turn.open' });
    const system = (body(h.ios[1]).messages as JsonObject[])[0].content as string;
    expect(system).toBe(
      "Be brief.\n\nLive context:\n- The call started on Wednesday, 21 October 2026 at 13:12, the user's local time (time zone Europe/Berlin, GMT+02:00).\n- The user's browser language is de-DE.",
    );
    h.ios[1].h.onRecv?.(chunk('Evening!', 'stop'), undefined);
    h.ios[1].h.onRecv?.({ $done: true }, undefined);
    h.send(60_000 * 30, 'turn', { type: 'turn.user', text: 'Hi.', turnId: 't0' });
    expect((body(h.ios[2]).messages as JsonObject[])[0].content).toBe(system);
  });
});

describe('the LLM opens the call', () => {
  const llm = (params: Record<string, unknown> = {}) => harness(llmNode, { model: 'gemma', system: 'Be brief.', utterances: 'plain', liveContext: false, openPrompt: '(Open.)', ...params }, 'llm');

  it('answers the opening prompt, and the history keeps only its words', () => {
    const h = llm();
    h.send(0, 'turn', { type: 'turn.open' });
    expect(h.of('llm.start')).toEqual([{ type: 'llm.start', rid: 'llm-r0', turnId: null, opening: true }]);
    expect(contents(h.ios[0])).toEqual(['system: Be brief.', 'user: (Open.)']);
    h.ios[0].h.onRecv?.(chunk('Hey, good evening!', 'stop'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
    expect(h.of('llm.end')).toMatchObject([{ rid: 'llm-r0', status: 'done', text: 'Hey, good evening!' }]);
    h.send(1000, 'turn', { type: 'turn.user', text: 'Hi!', turnId: 't0' });
    expect(contents(h.ios[1])).toEqual(['system: Be brief.', 'assistant: Hey, good evening!', 'user: Hi!']);
  });

  it('opens only before anyone spoke', () => {
    const h = llm();
    h.send(0, 'turn', { type: 'turn.user', text: 'Hello?', turnId: 't0' });
    h.send(10, 'turn', { type: 'turn.open' });
    expect(h.ios).toHaveLength(1);
    expect(h.of('llm.start')).toHaveLength(1);
  });

  it('keeps what was heard of an opening the user cut off, found by its reply', () => {
    const h = llm();
    h.send(0, 'turn', { type: 'turn.open' });
    h.ios[0].h.onRecv?.(chunk('Hey there, how is your evening going?', 'stop'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
    h.send(500, 'turn', { type: 'turn.heard', turnId: null, rid: 'llm-r0', text: 'Hey there,' });
    h.send(600, 'turn', { type: 'turn.user', text: 'Sorry, quick question.', turnId: 't0' });
    expect(contents(h.ios[1])).toEqual(['system: Be brief.', 'assistant: Hey there,—', 'user: Sorry, quick question.']);
  });

  it('continues a paced, tagged opening with the opening prompt still there', () => {
    const h = llm({ utterances: 'tagged' });
    h.send(0, 'sched', { type: 'sched.config', pace: true, firstTokens: 2 });
    h.send(0, 'turn', { type: 'turn.open' });
    expect((body(h.ios[0]).messages as JsonObject[]).map((m) => m.role)).toEqual(['system', 'user', 'assistant']);
    h.ios[0].h.onRecv?.(chunk('1g1] Hi'), undefined);
    h.ios[0].h.onRecv?.(chunk(' there', 'length'), undefined);
    h.ios[0].h.onRecv?.({ $done: true }, undefined);
    h.send(100, 'sched', { type: 'sched.grant', rid: 'llm-r0', tokens: 4 });
    const msgs = body(h.ios[1]).messages as JsonObject[];
    expect(msgs.map((m) => m.role)).toEqual(['system', 'user', 'assistant']);
    expect(msgs[1].content).toBe('(Open.)');
    expect(msgs[2].content).toBe('[i1g1] Hi there');
  });
});

describe('turn-taking opens the call', () => {
  const turn = (params: Record<string, unknown> = {}) => harness(duplexTurnNode, { useModel: false, ...params });

  it('asks the LLM to open once the ASR is connected, once', () => {
    const h = turn();
    h.send(0, 'transcript', { type: 'asr.open' });
    h.send(10, 'transcript', { type: 'asr.open' });
    expect(h.of('turn.open')).toHaveLength(1);
  });

  it('waits for the user when off', () => {
    const h = turn({ opening: 'off' });
    h.send(0, 'transcript', { type: 'asr.open' });
    expect(h.of('turn.open')).toHaveLength(0);
  });

  it("tells the LLM what was heard of an opening cut off, by its reply", () => {
    const h = turn();
    h.send(0, 'transcript', { type: 'asr.open' });
    h.send(100, 'playback', { type: 'playback.start', rid: 'llm-r0', at: 100 });
    h.send(900, 'playback', { type: 'playback.interrupted', rid: 'llm-r0', heard: 'Hey there,' });
    expect(h.of('turn.heard')).toEqual([{ type: 'turn.heard', turnId: null, rid: 'llm-r0', text: 'Hey there,' }]);
  });
});
