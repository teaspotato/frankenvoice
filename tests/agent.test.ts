// The voice agent on its own clock (System 1) with mock System 2 workers, against a scripted fake
// of the LLM: the user's side (turn-taking's committed turns, the VAD, the ASR opening) is scripted
// on the session clock, and a stand-in speaker plays whatever the agent says. Recorded live, then
// replayed against the recording.

import { describe, expect, it } from 'vitest';
import { Runtime } from '../src/core/runtime';
import { LiveScheduler } from '../src/core/scheduler';
import { LiveWorld, type LiveDriver } from '../src/core/world';
import { prepareReplay } from '../src/core/replay';
import { decodeBundle, encodeBundle } from '../src/core/bundle';
import { GRAPH_FORMAT, GRAPH_FORMAT_VERSION, type GraphSpec } from '../src/core/graph';
import { resolveParams, type IOHandlers, type NodeContext, type NodeDef, type TimerHandle } from '../src/core/node';
import { agentLoopNode } from '../src/nodes/agent';
import { FLOOR } from '../src/nodes/turn';
import type { JsonObject, Msg } from '../src/core/types';
import { registry } from '../src/nodes';
import { cutDecision, DecisionParser, DECISION_GRAMMAR, parseLine, saidWords } from '../src/nodes/agent-format';

/** The user's side: [ms, port, message], on the session clock. */
const SCRIPT: [number, string, JsonObject][] = [
  [50, 'transcript', { type: 'asr.open' }],
  [400, 'vad', { type: 'vad.speech_start' }],
  [600, 'vad', { type: 'vad.speech_end' }],
  [600, 'turn', { type: 'turn.user', turnId: 'u0', text: 'how are you?' }],
  [1000, 'vad', { type: 'vad.speech_start' }],
  [1200, 'vad', { type: 'vad.speech_end' }],
  [1200, 'turn', { type: 'turn.user', turnId: 'u1', text: "what's the weather in Kyoto?" }],
  [2100, 'vad', { type: 'vad.speech_start' }],
  [2300, 'vad', { type: 'vad.speech_end' }],
  [2300, 'turn', { type: 'turn.user', turnId: 'u2', text: 'please be quiet for a bit' }],
  [3800, 'vad', { type: 'vad.speech_start' }],
  [4000, 'vad', { type: 'vad.speech_end' }],
  [4000, 'turn', { type: 'turn.user', turnId: 'u3', text: 'okay, you can talk again' }],
];

const scriptNode: NodeDef = {
  type: 'test.script',
  title: 'Scripted user',
  category: 'mock',
  inputs: [],
  outputs: [{ name: 'turn', type: 'turn' }, { name: 'vad', type: 'vad' }, { name: 'transcript', type: 'transcript' }],
  params: [],
  create(ctx) {
    return {
      start() {
        for (const [at, port, msg] of SCRIPT) ctx.setTimer(at, () => ctx.emit(port, msg as Msg), 'script');
      },
    };
  },
};

/** Plays a reply 50 ms after it starts, until 150 ms after its text ends; a cancel cuts it. */
const speakerNode: NodeDef = {
  type: 'test.speaker',
  title: 'Stand-in speaker',
  category: 'mock',
  inputs: [{ name: 'text', type: 'text' }, { name: 'control', type: 'control' }],
  outputs: [{ name: 'playback', type: 'playback' }],
  params: [],
  create(ctx) {
    const live = new Map<string, { started: boolean; timers: TimerHandle[] }>();
    return {
      onInput(port, msg) {
        if (port === 'control' && msg.type === 'cancel') {
          for (const [rid, r] of live) {
            r.timers.forEach((t) => t.clear());
            if (r.started) ctx.emit('playback', { type: 'playback.interrupted', rid, heard: '' });
          }
          live.clear();
          return;
        }
        const rid = msg.rid as string;
        if (msg.type === 'llm.start') {
          const r = { started: false, timers: [] as TimerHandle[] };
          live.set(rid, r);
          r.timers.push(ctx.setTimer(50, () => ((r.started = true), ctx.emit('playback', { type: 'playback.start', rid })), 'play'));
        } else if (msg.type === 'llm.end' && live.has(rid)) {
          live.get(rid)!.timers.push(
            ctx.setTimer(150, () => {
              live.delete(rid);
              ctx.emit('playback', { type: 'playback.end', rid });
            }, 'play'),
          );
        }
      },
    };
  },
};

/** Gemma, scripted: what it writes for the newest lines of the log. */
function decide(newest: string): string {
  if (/session start/.test(newest)) return "[i1g1] Hi, I'm here. What can I do for you?\n[silent]";
  if (/w1 result/.test(newest)) return "[i1g1] Kyoto is sunny today, about 25 degrees.\n[silent]";
  if (/user: how are you/.test(newest)) return "[i1g1] I'm doing well, thanks for asking!\n[silent]";
  if (/user: what's the weather/.test(newest)) return '[i1g1] Let me look that up.\n[think] Current weather in Kyoto, Japan\n[wait 30]';
  if (/user: please be quiet/.test(newest)) return '[i1g0] Sure.\n[quiet]';
  if (/user: okay, you can talk again/.test(newest)) return '[i1g1] Great, I am back.\n[silent]';
  if (/silence for/.test(newest)) return '[i1g1] Still there? Anything else I can look up?\n[silent]';
  return '[silent]';
}

const requests: JsonObject[] = [];
/** The chat endpoint: the decision in a few streamed chunks, 5 ms apart. */
const gemma: LiveDriver = (req, sink) => {
  const body = (req as { body: JsonObject }).body;
  requests.push(body);
  const msgs = body.messages as { role: string; content: string }[];
  const text = decide(msgs.at(-1)!.content);
  const chunks = text.match(/[\s\S]{1,12}/g)!;
  let closed = false;
  chunks.forEach((c, i) => setTimeout(() => !closed && sink.recv({ choices: [{ index: 0, delta: { content: c } }] }), 5 * (i + 1)));
  setTimeout(() => {
    if (closed) return;
    sink.recv({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
    sink.recv({ $done: true });
    sink.close(null);
  }, 5 * (chunks.length + 1));
  return { send() {}, close: () => void (closed = true) };
};

const e = (from: string, fp: string, to: string, tp: string) => ({ from: { node: from, port: fp }, to: { node: to, port: tp } });
const graph: GraphSpec = {
  format: GRAPH_FORMAT,
  formatVersion: GRAPH_FORMAT_VERSION,
  name: 'agent',
  nodes: [
    { id: 'script', type: 'test.script', params: {} },
    { id: 'llm', type: 'agent.loop', params: { deadAirMs: 800 } },
    { id: 'workers', type: 'workers.mock', params: { minMs: 400, maxMs: 400, result: 'Sunny, 25 degrees in Kyoto.' } },
    { id: 'speaker', type: 'test.speaker', params: {} },
  ],
  edges: [
    e('script', 'turn', 'llm', 'turn'),
    e('script', 'vad', 'llm', 'vad'),
    e('script', 'transcript', 'llm', 'transcript'),
    e('llm', 'text', 'speaker', 'text'),
    e('llm', 'control', 'speaker', 'control'),
    e('speaker', 'playback', 'llm', 'playback'),
    e('llm', 'workers', 'workers', 'workers'),
    e('workers', 'updates', 'llm', 'updates'),
  ],
};
const testRegistry = { ...registry, 'test.script': scriptNode, 'test.speaker': speakerNode };

describe('the voice agent on its own clock', () => {
  it('opens the call, answers, hands work to System 2 and reports it unprompted, keeps quiet when asked, and breaks a silence', async () => {
    const scheduler = new LiveScheduler();
    const rt = new Runtime({ graph, registry: testRegistry, scheduler, world: new LiveWorld({ openai: gemma }, scheduler), mode: 'live' });
    rt.start();
    await new Promise((r) => setTimeout(r, 5800));
    rt.stop();
    const run = rt.trajectory;
    expect(run.events.filter((x) => x.kind === 'log' && (x.data as { level: string }).level === 'error')).toEqual([]);
    const msgs = run.events.filter((x) => x.kind === 'msg').map((x) => ({ ...(x.data as JsonObject), t: x.t }) as JsonObject & { t: number; type: string });
    const said = msgs.filter((m) => m.type === 'agent.decide').map((m) => ({ t: m.t, text: String(m.text), status: m.status }));
    const turnAt = (id: string) => msgs.find((m) => m.type === 'turn.user' && m.turnId === id)!.t;
    const saidBetween = (a: number, b: number) => said.filter((s) => s.t >= a && s.t < b && saidWords(s.text)).map((s) => saidWords(s.text));

    // 1. It speaks first, at the session's start, before the user says anything.
    expect(saidBetween(0, turnAt('u0'))).toEqual(["Hi, I'm here. What can I do for you?"]);
    expect(msgs.find((m) => m.type === 'llm.start')!.t).toBeLessThan(turnAt('u0'));

    // 2. It answers a question.
    expect(saidBetween(turnAt('u0'), turnAt('u1'))).toEqual(["I'm doing well, thanks for asking!"]);

    // 3. It hands the lookup to System 2 with a holding line, then tells the result without being asked.
    const start = msgs.find((m) => m.type === 'worker.start')!;
    expect(start).toMatchObject({ workerId: 'w1', task: 'Current weather in Kyoto, Japan' });
    const done = msgs.find((m) => m.type === 'worker.done')!;
    expect(done.t).toBeGreaterThan(turnAt('u1'));
    expect(done.t).toBeLessThan(turnAt('u2'));
    expect(saidBetween(turnAt('u1'), turnAt('u2'))).toEqual(['Let me look that up.', 'Kyoto is sunny today, about 25 degrees.']);
    const report = msgs.find((m) => m.type === 'agent.think' && m.t >= done.t)!;
    expect(report.wake).toBe('news');
    expect(msgs.some((m) => m.type === 'turn.user' && m.t > done.t && m.t <= report.t)).toBe(false);

    // 4. Asked to be quiet, it says "Sure." and then nothing: no silence wakes it until the user speaks.
    expect(saidBetween(turnAt('u2'), turnAt('u3'))).toEqual(['Sure.']);
    expect(msgs.filter((m) => m.type === 'agent.think' && m.t > turnAt('u2') + 10 && m.t < turnAt('u3'))).toEqual([]);
    expect(msgs.filter((m) => m.type === 'agent.event' && m.kind === 'silence' && m.t > turnAt('u2') && m.t < turnAt('u3'))).toEqual([]);

    // 5. Once it may talk again, a silence of 0.8 s wakes it and it breaks it.
    const silence = msgs.find((m) => m.type === 'agent.event' && m.kind === 'silence' && m.t > turnAt('u3'))!;
    expect(silence).toBeTruthy();
    const lastEnd = msgs.filter((m) => m.type === 'playback.end' && m.t < silence.t).at(-1)!;
    // (Within a few ms: a live timer is stamped at its due time, the playback end when it ran.)
    expect(silence.t - lastEnd.t).toBeGreaterThan(790);
    expect(saidBetween(turnAt('u3'), Infinity)).toEqual(['Great, I am back.', 'Still there? Anything else I can look up?']);

    // The log is append-only: every request's messages start with the previous request's.
    const logs = requests.map((b) => (b.messages as { role: string; content: string }[]).map((m) => `${m.role}:${m.content}`));
    for (let i = 1; i < logs.length; i++) {
      const prev = logs[i - 1];
      const prefix = logs[i].slice(0, prev.length - 1);
      expect(prefix).toEqual(prev.slice(0, -1));
      expect(logs[i][prev.length - 1].startsWith(prev.at(-1)!)).toBe(true);
    }
    expect(requests.every((b) => b.grammar === DECISION_GRAMMAR)).toBe(true);

    // And it all replays exactly.
    const { runtime, sim } = prepareReplay(decodeBundle(encodeBundle(run)), { registry: testRegistry });
    sim.runFast();
    expect(runtime.trajectory.events.filter((x) => x.kind === 'io.diverge')).toEqual([]);
    expect(runtime.trajectory.events).toEqual(run.events);
  }, 15_000);
});

describe('the decision format', () => {
  it('reads every kind of line', () => {
    expect(parseLine('[i0g1] Hold on.')).toEqual({ kind: 'say', interruptible: false, gate: true, text: 'Hold on.' });
    expect(parseLine('[think] Find a ramen place near Kyoto station')).toEqual({ kind: 'think', text: 'Find a ramen place near Kyoto station' });
    expect(parseLine('[tell w2] Open on Sundays only')).toEqual({ kind: 'tell', worker: 'w2', text: 'Open on Sundays only' });
    expect(parseLine('[stop w1]')).toEqual({ kind: 'stop', worker: 'w1' });
    expect(parseLine('[wait 45]')).toEqual({ kind: 'wait', s: 45 });
    expect(parseLine('[silent]')).toEqual({ kind: 'silent' });
    expect(parseLine('[quiet]')).toEqual({ kind: 'quiet' });
    expect(parseLine('hello')).toBeNull();
  });

  it('streams a say line’s words as they come, and acts on other lines once they are whole', () => {
    const p = new DecisionParser();
    const out = [...p.feed('[i1g'), ...p.feed('0] Let me '), ...p.feed('see.\n[thi'), ...p.feed('nk] the time'), ...p.feed('\n[wait 5]')];
    expect(out).toEqual([
      { type: 'say', interruptible: true, gate: false },
      { type: 'words', text: 'Let me ' },
      { type: 'words', text: 'see.' },
      { type: 'line', line: { kind: 'say', interruptible: true, gate: false, text: 'Let me see.' }, raw: '[i1g0] Let me see.' },
      { type: 'line', line: { kind: 'think', text: 'the time' }, raw: '[think] the time' },
    ]);
    expect(p.end()).toEqual([{ type: 'line', line: { kind: 'wait', s: 5 }, raw: '[wait 5]' }]);
  });

  it('cuts a decision to what was heard, keeping its other lines', () => {
    const raw = '[i1g1] Let me look that up.\n[think] weather\n[i1g1] It may take a moment.\n[wait 20]';
    expect(cutDecision(raw, 7)).toBe('[i1g1] Let me look that up.\n[think] weather\n[i1g1] It may—\n[wait 20]');
    expect(cutDecision(raw, 2)).toBe('[i1g1] Let me—\n[think] weather\n[wait 20]');
    expect(cutDecision(raw, 0)).toBe('[think] weather\n[wait 20]');
    expect(saidWords(raw)).toBe('Let me look that up. It may take a moment.');
  });
});

/** Drives the agent node by hand: a manual clock, timers, fake IO, and what it emitted. */
function agentHarness(params: Record<string, unknown> = {}) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ios: { request: JsonObject; h: IOHandlers; closed: boolean }[] = [];
  const ctx: NodeContext = {
    id: 'llm',
    params: resolveParams(agentLoopNode, params as never),
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
        const x = { request: request as JsonObject, h, closed: false };
        ios.push(x);
        return { xid: String(ios.length), send() {}, close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: { append: () => { throw new Error('no media'); } },
  };
  const node = agentLoopNode.create(ctx as never);
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
  const write = (i: number, text: string) => ios[i].h.onRecv!({ choices: [{ index: 0, delta: { content: text } }] }, undefined);
  const end = (i: number) => (ios[i].h.onRecv!({ $done: true }, undefined), ios[i].h.onClose!(null));
  const messages = (i: number) => ((ios[i].request.body as JsonObject).messages as { role: string; content: string }[]).filter((m) => m.role !== 'system');
  return { send, advance, out, of, ios, write, end, messages };
}

describe('the voice agent, step by step', () => {
  it("drops a step the user's words overtake, with its unheard speech, and thinks again with both events", () => {
    const h = agentHarness();
    h.send(0, 'transcript', { type: 'asr.open' });
    h.advance(1);
    h.write(0, '[i1g1] Hello the');
    h.send(100, 'turn', { type: 'turn.user', turnId: 'u0', text: 'wait, a question' });
    expect(h.ios[0].closed).toBe(true);
    expect(h.out.find((o) => o.port === 'control')?.msg).toEqual({ type: 'cancel' });
    expect(h.of('llm.end')).toMatchObject([{ status: 'cancelled' }]);
    h.advance(101);
    // Nothing of it was heard: it leaves the log, and both events go to the next step.
    expect(h.messages(1).map((m) => m.role)).toEqual(['user']);
    expect(h.messages(1)[0].content).toMatch(/session start.*\n0\.1s user: wait, a question/s);
  });

  it('keeps what the user heard of a decision they cut in on, and says so on the log', () => {
    const h = agentHarness();
    h.send(0, 'transcript', { type: 'asr.open' });
    h.advance(1);
    h.write(0, '[i1g1] The weather today is sunny.\n[i1g1] Tomorrow it rains.\n[silent]');
    h.end(0);
    const rid = h.of('llm.start')[0].rid;
    h.send(200, 'playback', { type: 'playback.start', rid });
    h.send(900, 'control', { type: 'cancel' });
    h.send(950, 'turn', { type: 'turn.heard', turnId: null, rid, text: 'The weather today' });
    h.send(1500, 'turn', { type: 'turn.user', turnId: 'u0', text: 'and tomorrow?' });
    h.advance(1501);
    expect(h.messages(1).map((m) => m.content)).toEqual([
      '0.0s session start: the user is on the line and can hear you',
      '[i1g1] The weather today—\n[silent]',
      '0.9s user cut in; they heard you up to "The weather today"\n1.5s user: and tomorrow?',
    ]);
  });

  it('passes the floor back when a step says nothing, and holds news while quiet', () => {
    const h = agentHarness();
    h.send(0, 'turn', { type: 'turn.user', turnId: 'u0', text: 'please be quiet' });
    h.advance(1);
    h.write(0, '[quiet]');
    h.end(0);
    expect(h.of('agent.pass')).toHaveLength(1);
    h.send(500, 'updates', { type: 'worker.done', workerId: 'w9', text: 'done' });
    h.advance(60_000);
    expect(h.ios).toHaveLength(1);
    // The user's own words wake it, and the news is on the log by then.
    h.send(61_000, 'turn', { type: 'turn.user', turnId: 'u1', text: 'any news?' });
    h.advance(61_001);
    expect(h.messages(1).at(-1)!.content).toMatch(/w9 result: done\n61\.0s user: any news\?/);
  });

  it('turn-taking frees the floor when the agent passes', () => {
    expect(FLOOR.thinking.pass).toBe('listening');
    expect(FLOOR['thinking+user'].pass).toBe('listening');
    expect(FLOOR.speaking.pass).toBeUndefined();
  });
});
