// The pipeline in the server: the Node driver registry against fake upstreams (a gpubox ASR/TTS
// WebSocket server and an OpenAI-compatible SSE server, both from the fake drivers), and a fake
// page on /api/live feeding mic PCM and taking the speaker's commands.

import { createServer, type Server } from 'node:http';
import { mkdtempSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { WebSocket, WebSocketServer } from 'ws';
import { LiveHub } from '../server/live';
import { decodeBundle } from '../src/core/bundle';
import { prepareReplay } from '../src/core/replay';
import { compareTrajectories } from '../src/core/compare';
import { registry } from '../src/nodes';
import { buildGraph, defaultConfig } from '../src/pipeline';
import { frame, unframe } from '../src/live-protocol';
import type { Json } from '../src/core/types';
import type { LiveDriver } from '../src/core/world';
import type { Upstream } from '../server/drivers';
import { fakeDrivers } from './fake-drivers';

const listen = (s: Server) => new Promise<number>((r) => s.listen(0, '127.0.0.1', () => r((s.address() as { port: number }).port)));
const TOKEN = 'test-token';

/** Serve a fake driver as a gpubox WebSocket endpoint: its messages become frames, frames become sends. */
function serveDriver(ws: WebSocket, driver: LiveDriver, req: Json) {
  const conn = driver(req, {
    recv(m, bin) {
      if ((m as { type?: string } | null)?.type === '$open') return;
      if (ws.readyState !== ws.OPEN) return;
      if (bin) ws.send(bin);
      else ws.send(JSON.stringify(m));
    },
    close: () => ws.readyState === ws.OPEN && ws.close(1000),
  }, { now: () => 0 });
  ws.on('message', (data: Buffer, binary) => (binary ? conn.send(null, new Uint8Array(data)) : conn.send(JSON.parse(data.toString()) as Json)));
  ws.on('close', () => conn.close());
}

let servers: Server[] = [];
let gpuboxPort = 0;
let llmPort = 0;
/** The live session limit the hub uses (ms; 0: none), set per test. */
let limitMs = 0;
let livePort = 0;
const seen = { asr: 0, tts: 0, llm: 0, completions: 0, auth: [] as string[], ttsQueries: [] as string[], llmBodies: [] as { messages?: { role: string; content: unknown }[] }[] };
const sessions = mkdtempSync(join(tmpdir(), 'fv-sessions-'));

beforeAll(async () => {
  const gpubox = createServer((req, res) => {
    if (req.url !== '/v1/tts/config') return res.writeHead(404).end();
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ model: 'fake', n_q: 32, temp: 0.6, voice: 'v', cfg: 2, sample_rate: 24000 }));
  });
  const gwss = new WebSocketServer({ server: gpubox });
  gwss.on('connection', (ws, req) => {
    seen.auth.push(req.headers.authorization ?? '');
    const path = new URL(req.url!, 'http://x').pathname;
    if (path === '/v1/asr') (seen.asr++, serveDriver(ws, fakeDrivers['gpubox-asr'], {}));
    else if (path === '/v1/tts') {
      seen.tts++;
      seen.ttsQueries.push(new URL(req.url!, 'http://x').search);
      serveDriver(ws, fakeDrivers['gpubox-tts'], {});
    }
    else ws.close(4004);
  });
  const llm = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    if (req.url === '/completion') {
      // llama.cpp's own endpoint, on the server root (the tape).
      seen.completions++;
      let answer: Json = null;
      return fakeDrivers['llama-completion']({ body: JSON.parse(Buffer.concat(chunks).toString()) as Json }, {
        recv: (e) => void (answer = e),
        close: () => res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ...(answer as object), prompt: 'echoed', generation_settings: {} })),
      }, { now: () => 0 });
    }
    if (req.url !== '/v1/chat/completions') return res.writeHead(404).end();
    seen.llm++;
    seen.llmBodies.push(JSON.parse(Buffer.concat(chunks).toString()) as { messages?: { role: string; content: unknown }[] });
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    fakeDrivers.openai({ body: JSON.parse(Buffer.concat(chunks).toString()) as Json }, {
      recv: (e) => res.write(`data: ${(e as { $done?: boolean }).$done ? '[DONE]' : JSON.stringify(e)}\n\n`),
      close: () => res.end(),
    }, { now: () => 0 });
  });
  gpuboxPort = await listen(gpubox);
  llmPort = await listen(llm);
  const hub = new LiveHub({
    upstreams: (name): Upstream =>
      name === 'gpubox'
        ? { base: `ws://127.0.0.1:${gpuboxPort}`, headers: { authorization: `Bearer ${TOKEN}` } }
        : name === 'llm'
          ? { base: `http://127.0.0.1:${llmPort}/v1`, headers: {} }
          : { headers: {}, missing: 'TYPESAFE_API_KEY is not set' },
    sessionsDir: () => sessions,
    modelsDir: join(sessions, 'models'),
    maxSessionMs: () => limitMs,
    demoGraph: () => {
      const c = defaultConfig();
      c.vad.variant = 'energy';
      c.backchannel.variant = 'off';
      return buildGraph(c);
    },
    heartbeatMs: 300,
    log: () => {},
  });
  const server = createServer();
  const wss = new WebSocketServer({ noServer: true });
  server.on('upgrade', (req, socket, head) =>
    wss.handleUpgrade(req, socket, head, (ws) => hub.handle(ws, { host: !req.url?.includes('visitor') })),
  );
  livePort = await listen(server);
  servers = [gpubox, llm, server];
});

afterAll(() => {
  for (const s of servers) s.closeAllConnections?.(), s.close();
});

describe('the pipeline in the server', () => {
  it('runs a live session from the page mic to the page speaker, and saves it', async () => {
    const page = new WebSocket(`ws://127.0.0.1:${livePort}/api/live`);
    await new Promise((r) => page.once('open', r));
    const got = { session: 0, events: 0, tracks: new Set<string>(), mic: 0, play: 0, lateSent: 0, saved: '' };
    let bundle: Uint8Array | undefined;
    let micTimer: ReturnType<typeof setInterval> | undefined;
    const ended = new Promise<void>((resolve) => {
      page.on('message', (data: Buffer, binary) => {
        if (binary) {
          const { header: h, payload } = unframe(new Uint8Array(data));
          if (h.k === 'track') got.tracks.add(h.name as string);
          if (h.k === 'bundle') bundle = payload.slice();
          if (h.k === 'speaker' && h.cmd === 'play') {
            got.play++;
            expect(payload.byteLength).toBeGreaterThan(0);
            expect(typeof h.now).toBe('number');
            // As the page does when a run of playback starts late.
            if (got.play === 1) page.send(JSON.stringify({ type: 'speaker.late', id: h.id, at: h.at, ms: 12 })), got.lateSent++;
          }
          return;
        }
        const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
        if (m.type === 'session') got.session++;
        if (m.type === 'events') got.events += (m.events as Json[]).length;
        if (m.type === 'saved') (got.saved = m.path as string), resolve();
        if (m.type === 'mic.open') {
          got.mic++;
          const { frameSamples } = m.request as { frameSamples: number };
          page.send(JSON.stringify({ type: 'mic.ready', label: 'fake page mic', contextSampleRate: 16000, settings: {} }));
          // Noise bursts as speech, as the fake mic does, a frame every few ms.
          let n = 0;
          micTimer = setInterval(() => {
            const s = new Float32Array(frameSamples);
            const loud = n % 40 >= 8 && n % 40 < 24;
            for (let i = 0; i < s.length; i++) s[i] = (Math.random() - 0.5) * (loud ? 0.6 : 0.001);
            page.send(frame({ k: 'mic', frame: n++, ctxTime: n * 0.032 }, new Uint8Array(s.buffer)));
          }, 4);
        }
        if (m.type === 'mic.close') clearInterval(micTimer);
      });
    });
    const c = defaultConfig();
    c.vad.variant = 'energy';
    c.backchannel.variant = 'off';
    page.send(JSON.stringify({ type: 'start', graph: buildGraph(c) as unknown as Json, env: { userAgent: 'test' }, autosave: true, title: '' }));
    await new Promise((r) => setTimeout(r, 2500));
    page.send(JSON.stringify({ type: 'stop' }));
    await ended;
    page.close();

    expect(got.session).toBe(1);
    expect(got.mic).toBe(1);
    expect(got.events).toBeGreaterThan(100);
    expect(got.tracks.has('mic.audio')).toBe(true);
    expect(got.play).toBeGreaterThan(0);
    expect(seen.asr).toBeGreaterThan(0);
    expect(seen.tts).toBeGreaterThan(0);
    expect(seen.llm).toBeGreaterThan(0);
    expect(new Set(seen.auth)).toEqual(new Set([`Bearer ${TOKEN}`]));
    expect(readdirSync(sessions).filter((f) => f.endsWith('.fvs'))).toHaveLength(1);

    const traj = decodeBundle(bundle!);
    const types = new Set(traj.events.filter((e) => e.kind === 'msg').map((e) => (e.data as { type: string }).type));
    for (const t of ['asr.final', 'turn.user', 'llm.delta', 'tts.words', 'playback.start']) expect(types).toContain(t);
    // The agent opens once the ASR is connected.
    const msgs = traj.events.filter((e) => e.kind === 'msg').map((e) => e.data as { type: string; why?: string });
    const open = msgs.findIndex((m) => m.type === 'asr.open');
    expect(open).toBeGreaterThanOrEqual(0);
    expect(msgs.findIndex((m) => m.type === 'bc.say' && m.why === 'greeting')).toBeGreaterThan(open);
    // And its line opens the LLM transcript.
    const reply = seen.llmBodies.find((b) => b.messages?.some((m) => m.role === 'user'))!;
    expect(reply.messages!.filter((m) => m.role !== 'system')[0]).toMatchObject({ role: 'assistant', content: expect.stringContaining("What's on your mind?") });
    expect(traj.events.some((e) => e.kind === 'io.recv' && (e.data as { payload?: { type?: string } }).payload?.type === 'late')).toBe(got.lateSent > 0);
    expect(traj.events.filter((e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toEqual([]);
    // The server's recording replays exactly, like one made in the page.
    const { runtime, sim } = prepareReplay(traj, { registry });
    sim.runFast();
    expect(compareTrajectories(traj, runtime.trajectory).identical).toBe(true);
  }, 20_000);

  it('runs the tape: one /completion per cell on the LLM server root, its words spoken, the recording replayable', async () => {
    const page = new WebSocket(`ws://127.0.0.1:${livePort}/api/live`);
    await new Promise((r) => page.once('open', r));
    let bundle: Uint8Array | undefined;
    let plays = 0;
    let micTimer: ReturnType<typeof setInterval> | undefined;
    const llmBefore = seen.llm;
    const ended = new Promise<void>((resolve) => {
      page.on('message', (data: Buffer, binary) => {
        if (binary) {
          const { header: h, payload } = unframe(new Uint8Array(data));
          if (h.k === 'bundle') bundle = payload.slice();
          if (h.k === 'speaker' && h.cmd === 'play') plays++;
          return;
        }
        const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
        if (m.type === 'saved') resolve();
        if (m.type === 'mic.open') {
          const { frameSamples } = m.request as { frameSamples: number };
          page.send(JSON.stringify({ type: 'mic.ready', label: 'fake page mic', contextSampleRate: 16000, settings: {} }));
          let n = 0;
          micTimer = setInterval(() => {
            const s = new Float32Array(frameSamples);
            const loud = n % 40 >= 8 && n % 40 < 24;
            for (let i = 0; i < s.length; i++) s[i] = (Math.random() - 0.5) * (loud ? 0.6 : 0.001);
            page.send(frame({ k: 'mic', frame: n++, ctxTime: n * 0.032 }, new Uint8Array(s.buffer)));
          }, 4);
        }
        if (m.type === 'mic.close') clearInterval(micTimer);
      });
    });
    const c = defaultConfig();
    c.vad.variant = 'energy';
    c.turn.variant = 'tape';
    c.turn.params.tape = { cellMs: 120 };
    page.send(JSON.stringify({ type: 'start', graph: buildGraph(c) as unknown as Json, env: {}, autosave: true, title: '' }));
    await new Promise((r) => setTimeout(r, 2500));
    page.send(JSON.stringify({ type: 'stop' }));
    await ended;
    page.close();

    expect(seen.completions).toBeGreaterThan(3);
    // The chat endpoint is only asked for the session's title, never for a reply.
    expect(seen.llmBodies.slice(llmBefore).filter((b) => b.messages?.some((m) => m.role === 'user' && !String(m.content).includes('<transcript>')))).toEqual([]);
    expect(plays).toBeGreaterThan(0);
    const traj = decodeBundle(bundle!);
    const cells = traj.events.filter((e) => e.kind === 'msg' && (e.data as { type: string }).type === 'tape.cell').map((e) => e.data as { action: string; ms?: number; promptN?: number });
    expect(cells.some((x) => typeof x.ms === 'number' && x.promptN === 12)).toBe(true);
    expect(cells.some((x) => x.action === 'respond')).toBe(true);
    // The echoed prompt is not recorded.
    const recv = traj.events.find((e) => e.kind === 'io.recv' && e.node === 'turn');
    expect(JSON.stringify(recv?.data)).not.toContain('echoed');
    expect(traj.events.filter((e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toEqual([]);
    const { runtime, sim } = prepareReplay(traj, { registry });
    sim.runFast();
    expect(compareTrajectories(traj, runtime.trajectory).identical).toBe(true);
  }, 20_000);

  it('runs one session at a time: other pages queue, and the next starts when the one ahead ends', async () => {
    const c = defaultConfig();
    c.vad.variant = 'energy';
    c.backchannel.variant = 'off';
    const start = JSON.stringify({ type: 'start', graph: buildGraph(c) as unknown as Json, env: {}, autosave: true, title: '' });
    const open = async () => {
      const ws = new WebSocket(`ws://127.0.0.1:${livePort}/api/live`);
      const got = { users: [] as Json[], queue: [] as Json[], session: 0, saved: '', plays: 0 };
      let micTimer: ReturnType<typeof setInterval> | undefined;
      const saved = new Promise<void>((resolve) =>
        ws.on('message', (data: Buffer, binary) => {
          if (binary) {
            const { header: h } = unframe(new Uint8Array(data));
            if (h.k === 'speaker' && h.cmd === 'play') got.plays++;
            return;
          }
          const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
          if (m.type === 'users') got.users.push(m);
          if (m.type === 'queue') got.queue.push(m.position);
          if (m.type === 'session') got.session++;
          if (m.type === 'saved') (got.saved = m.path as string), resolve();
          if (m.type === 'mic.open') {
            const { frameSamples } = m.request as { frameSamples: number };
            ws.send(JSON.stringify({ type: 'mic.ready', label: 'fake', contextSampleRate: 16000, settings: {} }));
            let n = 0;
            micTimer = setInterval(() => {
              if (ws.readyState !== ws.OPEN) return clearInterval(micTimer);
              const s = new Float32Array(frameSamples);
              const loud = n % 40 >= 8 && n % 40 < 24;
              for (let i = 0; i < s.length; i++) s[i] = (Math.random() - 0.5) * (loud ? 0.6 : 0.001);
              ws.send(frame({ k: 'mic', frame: n++, ctxTime: n * 0.032 }, new Uint8Array(s.buffer)));
            }, 4);
          }
          if (m.type === 'mic.close') clearInterval(micTimer);
        }),
      );
      await new Promise((r) => ws.once('open', r));
      return { ws, got, saved };
    };
    const [a, b, c3] = [await open(), await open(), await open()];
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    a.ws.send(start);
    await wait(200);
    b.ws.send(start);
    c3.ws.send(start);
    await wait(300);
    expect([a.got.session, b.got.session, c3.got.session]).toEqual([1, 0, 0]);
    expect(b.got.queue).toEqual([1]);
    expect(c3.got.queue).toEqual([2]);
    for (const p of [a, b, c3]) expect(p.got.users.at(-1)).toEqual({ type: 'users', pages: 3, sessions: 1, max: 1, queued: 2 });
    // The third leaves the queue.
    c3.ws.send(JSON.stringify({ type: 'stop' }));
    await wait(100);
    expect(c3.got.queue.at(-1)).toBe(0);
    expect(a.got.users.at(-1)).toMatchObject({ queued: 1 });
    await wait(1200);
    expect(a.got.plays).toBeGreaterThan(0);
    // The first page goes away: its session ends and the next in line starts by itself.
    a.ws.close();
    await wait(1500);
    expect(b.got.queue.at(-1)).toBe(0);
    expect(b.got.session).toBe(1);
    expect(c3.got.session).toBe(0);
    expect(c3.got.users.at(-1)).toEqual({ type: 'users', pages: 2, sessions: 1, max: 1, queued: 0 });
    await wait(1000);
    expect(b.got.plays).toBeGreaterThan(0);
    b.ws.send(JSON.stringify({ type: 'stop' }));
    await b.saved;
    expect(b.got.saved).toMatch(/\.fvs$/);
    for (const p of [b, c3]) p.ws.close();
  }, 20_000);

  it('frees the box when a start throws: the next in line starts, and the run before it is still saved', async () => {
    const c = defaultConfig();
    c.vad.variant = 'energy';
    c.backchannel.variant = 'off';
    const good = buildGraph(c);
    // Passes checkGraph; Runtime.start throws on the unknown node type.
    const bad = { ...good, nodes: [...good.nodes, { id: 'bogus', type: 'no-such-node', params: {} }] };
    const start = (g: unknown) => JSON.stringify({ type: 'start', graph: g as Json, env: {}, autosave: true, title: '' });
    const open = async () => {
      const ws = new WebSocket(`ws://127.0.0.1:${livePort}/api/live`);
      const msgs: { type: string; [k: string]: Json }[] = [];
      const bins: string[] = [];
      ws.on('message', (data: Buffer, binary) => {
        if (binary) return void bins.push(unframe(new Uint8Array(data)).header.k);
        const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
        msgs.push(m);
        if (m.type === 'mic.open') ws.send(JSON.stringify({ type: 'mic.ready', label: 'fake', contextSampleRate: 16000, settings: {} }));
      });
      await new Promise((r) => ws.once('open', r));
      const of = (t: string) => msgs.filter((m) => m.type === t);
      return { ws, of, bins };
    };
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const savedBefore = readdirSync(sessions).filter((f) => f.endsWith('.fvs')).length;
    const [a, b, d] = [await open(), await open(), await open()];
    a.ws.send(start(good));
    await wait(200);
    b.ws.send(start(bad));
    d.ws.send(start(good));
    await wait(200);
    expect(b.of('queue').map((m) => m.position)).toEqual([1]);
    expect(d.of('queue').map((m) => m.position)).toEqual([2]);
    // The first ends: the bad start is next in line, inside the first run's finish.
    a.ws.send(JSON.stringify({ type: 'stop' }));
    await wait(800);
    expect(a.of('saved')).toHaveLength(1);
    expect(b.of('notice').map((m) => m.text)).toContainEqual(expect.stringMatching(/could not start the session: .*unknown node type "no-such-node"/));
    expect(b.of('ended')).toHaveLength(1);
    expect(b.bins).toContain('bundle');
    expect(b.of('saved')).toHaveLength(0);
    // The slot went to the next in line.
    expect(d.of('session')).toHaveLength(1);
    expect(d.of('queue').at(-1)).toMatchObject({ position: 0, started: true });
    expect(d.of('users').at(-1)).toMatchObject({ sessions: 1, queued: 0 });
    d.ws.send(JSON.stringify({ type: 'stop' }));
    await wait(500);
    expect(d.of('saved')).toHaveLength(1);
    // A bad start with the box free frees it too: the same page starts again.
    b.ws.send(start(bad));
    await wait(200);
    expect(b.of('notice')).toHaveLength(2);
    expect(b.of('users').at(-1)).toMatchObject({ sessions: 0 });
    b.ws.send(start(good));
    await wait(300);
    expect(b.of('session')).toHaveLength(3);
    b.ws.send(JSON.stringify({ type: 'stop' }));
    await wait(500);
    expect(b.of('saved')).toHaveLength(1);
    expect(readdirSync(sessions).filter((f) => f.endsWith('.fvs'))).toHaveLength(savedBefore + 3);
    for (const p of [a, b, d]) p.ws.close();
  }, 20_000);

  it("drops a visitor's oversized messages and mic frames, and mic audio faster than twice real time", async () => {
    const ws = new WebSocket(`ws://127.0.0.1:${livePort}/api/live?visitor`);
    let micBytes = 0;
    let pongs = 0;
    let req: { frameSamples: number; sampleRate: number } | undefined;
    ws.on('message', (data: Buffer, binary) => {
      if (binary) {
        const { header: h, payload } = unframe(new Uint8Array(data));
        if (h.k === 'track' && h.name === 'mic.audio') micBytes += payload.byteLength;
        return;
      }
      const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
      if (m.type === 'pong') pongs++;
      if (m.type === 'mic.open') {
        req = m.request as typeof req;
        ws.send(JSON.stringify({ type: 'mic.ready', label: 'fake', contextSampleRate: 16000, settings: {} }));
      }
    });
    await new Promise((r) => ws.once('open', r));
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    ws.send(JSON.stringify({ type: 'start', graph: {}, env: {}, autosave: false, title: '' }));
    await wait(300);
    const { frameSamples, sampleRate } = req!;
    const mic = (n: number, samples: number) => ws.send(frame({ k: 'mic', frame: n, ctxTime: 0 }, new Uint8Array(new Float32Array(samples).buffer)));
    // Bigger than the frame the mic asked for, then bigger than any visitor message, then one that fits.
    mic(0, frameSamples * 4);
    mic(1, 40_000);
    mic(2, frameSamples);
    ws.send(JSON.stringify({ type: 'ping', id: 1, t: 0 }));
    await wait(300);
    expect(micBytes).toBe(frameSamples * 4);
    expect(pongs).toBe(1);
    // A burst of 10 s of audio at once: about 2 s of it gets through.
    micBytes = 0;
    const n = Math.ceil((10 * sampleRate) / frameSamples);
    for (let i = 0; i < n; i++) mic(3 + i, frameSamples);
    await wait(300);
    const seconds = micBytes / 4 / sampleRate;
    expect(seconds).toBeGreaterThan(1.5);
    expect(seconds).toBeLessThan(3.5);
    const saved = new Promise((r) => ws.on('message', (d: Buffer, bin) => !bin && JSON.parse(d.toString()).type === 'saved' && r(null)));
    ws.send(JSON.stringify({ type: 'stop' }));
    await saved;
    ws.close();
  }, 20_000);

  it('drops a page that stopped answering, and stops a session at the time limit', async () => {
    const c = defaultConfig();
    c.vad.variant = 'energy';
    c.backchannel.variant = 'off';
    const start = JSON.stringify({ type: 'start', graph: buildGraph(c) as unknown as Json, env: {}, autosave: true, title: '' });
    const open = async (autoPong: boolean) => {
      const ws = new WebSocket(`ws://127.0.0.1:${livePort}/api/live`, { autoPong });
      const got = { msgs: [] as { type: string; [k: string]: Json }[], closed: false };
      let micTimer: ReturnType<typeof setInterval> | undefined;
      ws.on('close', () => ((got.closed = true), clearInterval(micTimer)));
      ws.on('message', (data: Buffer, binary) => {
        if (binary) return;
        const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
        got.msgs.push(m);
        if (m.type === 'mic.open') {
          const { frameSamples } = m.request as { frameSamples: number };
          ws.send(JSON.stringify({ type: 'mic.ready', label: 'fake', contextSampleRate: 16000, settings: {} }));
          let n = 0;
          micTimer = setInterval(() => {
            if (ws.readyState !== ws.OPEN) return clearInterval(micTimer);
            ws.send(frame({ k: 'mic', frame: n++, ctxTime: n * 0.032 }, new Uint8Array(new Float32Array(frameSamples).buffer)));
          }, 30);
        }
      });
      await new Promise((r) => ws.once('open', r));
      return { ws, got, of: (t: string) => got.msgs.filter((m) => m.type === t) };
    };
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    limitMs = 2500;
    try {
      // A page whose tab went away without a close: it never answers pings.
      const dead = await open(false);
      dead.ws.send(start);
      await wait(150);
      expect(dead.of('limit')).toEqual([{ type: 'limit', leftMs: 2500 }]);
      const b = await open(true);
      b.ws.send(start);
      await wait(100);
      const q = b.of('queue')[0];
      expect(q).toMatchObject({ position: 1 });
      expect(q.waitMs as number).toBeGreaterThan(2000);
      expect(q.waitMs as number).toBeLessThanOrEqual(2500);
      // Two missed pings drop the dead page, and the next in line starts.
      await wait(900);
      expect(dead.got.closed).toBe(true);
      expect(b.of('session')).toHaveLength(1);
      expect(b.of('queue').at(-1)).toMatchObject({ position: 0, started: true });
      // The session stops by itself at the limit, and is saved.
      await wait(2800);
      expect(b.of('notice').map((m) => m.text)).toContainEqual(expect.stringMatching(/limit/));
      expect(b.of('saved')).toHaveLength(1);
      expect(b.got.closed).toBe(false);
      b.ws.close();
    } finally {
      limitMs = 0;
    }
  }, 20_000);

  it('runs the demo pipeline for a visitor, whatever graph it sends, and keeps replays on the box for the host', async () => {
    const ws = new WebSocket(`ws://127.0.0.1:${livePort}/api/live?visitor`);
    const msgs: { type: string; [k: string]: Json }[] = [];
    ws.on('message', (data: Buffer, binary) => {
      if (binary) return;
      const m = JSON.parse(data.toString()) as { type: string; [k: string]: Json };
      msgs.push(m);
      if (m.type === 'mic.open') ws.send(JSON.stringify({ type: 'mic.ready', label: 'fake', contextSampleRate: 16000, settings: {} }));
    });
    await new Promise((r) => ws.once('open', r));
    ws.send(frame({ k: 'replay', liveChannels: ['gpubox-tts'] }, new Uint8Array(4)));
    ws.send(JSON.stringify({ type: 'start', graph: { format: 'not a graph' }, env: {}, autosave: false, title: '' }));
    await new Promise((r) => setTimeout(r, 300));
    expect(msgs.filter((m) => m.type === 'notice').map((m) => m.text)).toEqual(['replays on the GPU box are for the host only']);
    expect(msgs.filter((m) => m.type === 'session')).toHaveLength(1);
    const saved = new Promise((r) => ws.on('message', (d: Buffer, bin) => !bin && JSON.parse(d.toString()).type === 'saved' && r(null)));
    ws.send(JSON.stringify({ type: 'stop' }));
    // A visitor's session is always kept.
    await saved;
    ws.close();
  }, 20_000);

  it('renders a canned line once on the box, in the background, then serves it from the cache', async () => {
    const { serverDrivers } = await import('../server/drivers');
    const cannedDir = mkdtempSync(join(tmpdir(), 'fv-canned-'));
    const d = serverDrivers(
      (name): Upstream => (name === 'gpubox' ? { base: `ws://127.0.0.1:${gpuboxPort}`, headers: { authorization: `Bearer ${TOKEN}` } } : { headers: {}, missing: 'off' }),
      { modelsDir: sessions, cannedDir },
    );
    const get = () =>
      new Promise<{ head: Json; pcm: Uint8Array }>((resolve, reject) => {
        let head: Json = null;
        let pcm: Uint8Array = new Uint8Array();
        d['tts-canned']({ text: 'Mm-hmm.', voice: '', sampleRate: 24000 }, {
          recv: (m, bin) => (bin ? (pcm = bin) : (head = m)),
          close: (e) => (e ? reject(new Error(JSON.stringify(e))) : resolve({ head, pcm })),
        }, { now: () => 0 });
      });
    const before = seen.tts;
    const a = await get();
    expect(a.head).toMatchObject({ type: 'Canned', cached: false, sampleRate: 24000 });
    expect(seen.tts).toBe(before + 1);
    expect(seen.ttsQueries.at(-1)).toContain('priority=background');
    const b = await get();
    expect(b.head).toMatchObject({ type: 'Canned', cached: true, key: (a.head as { key: string }).key });
    expect(seen.tts).toBe(before + 1);
    expect(b.pcm).toEqual(a.pcm);
    expect(readdirSync(cannedDir)).toHaveLength(1);
  });

  it('runs onnxruntime-node VADs: Silero and FireRedVAD', async () => {
    const { serverDrivers } = await import('../server/drivers');
    const d = serverDrivers(() => ({ headers: {}, missing: 'off' }), { modelsDir: sessions, cannedDir: sessions });
    const tone = new Float32Array(512).map((_, i) => 0.3 * Math.sin(i / 3));
    for (const [ch, req] of [
      ['silero', { model: 'v5', sampleRate: 16000, frameSamples: 512 }],
      ['firered', { model: 'stream-vad', sampleRate: 16000, chunkSamples: 512 }],
    ] as const) {
      const out = await new Promise<Json>((resolve, reject) => {
        const conn = d[ch](req, { recv: (m) => resolve(m), close: (e) => reject(new Error(JSON.stringify(e))) }, { now: () => 0 });
        conn.send(null, new Uint8Array(tone.buffer.slice(0)));
      });
      const p = (out as { p: number | number[] }).p;
      expect([p].flat().every((x) => x >= 0 && x <= 1)).toBe(true);
    }
  });
});
