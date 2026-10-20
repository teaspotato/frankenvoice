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
let livePort = 0;
const seen = { asr: 0, tts: 0, llm: 0, auth: [] as string[], ttsQueries: [] as string[] };
const sessions = mkdtempSync(join(tmpdir(), 'fv-sessions-'));
let cap = 4;

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
    if (req.url !== '/v1/chat/completions') return res.writeHead(404).end();
    seen.llm++;
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
    maxSessions: () => cap,
    log: () => {},
  });
  const server = createServer();
  const wss = new WebSocketServer({ noServer: true });
  server.on('upgrade', (req, socket, head) => wss.handleUpgrade(req, socket, head, (ws) => hub.handle(ws)));
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
    expect(traj.events.some((e) => e.kind === 'io.recv' && (e.data as { payload?: { type?: string } }).payload?.type === 'late')).toBe(got.lateSent > 0);
    expect(traj.events.filter((e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toEqual([]);
    // The server's recording replays exactly, like one made in the page.
    const { runtime, sim } = prepareReplay(traj, { registry });
    sim.runFast();
    expect(compareTrajectories(traj, runtime.trajectory).identical).toBe(true);
  }, 20_000);

  it('runs a session per page, up to the cap, and tells every page who is on', async () => {
    cap = 2;
    const c = defaultConfig();
    c.vad.variant = 'energy';
    c.backchannel.variant = 'off';
    const start = JSON.stringify({ type: 'start', graph: buildGraph(c) as unknown as Json, env: {}, autosave: true, title: '' });
    const open = async () => {
      const ws = new WebSocket(`ws://127.0.0.1:${livePort}/api/live`);
      const got = { users: [] as Json[], session: 0, full: '', saved: '', plays: 0 };
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
          if (m.type === 'session') got.session++;
          if (m.type === 'full') got.full = m.text as string;
          if (m.type === 'saved') (got.saved = m.path as string), resolve();
          if (m.type === 'mic.open') {
            const { frameSamples } = m.request as { frameSamples: number };
            ws.send(JSON.stringify({ type: 'mic.ready', label: 'fake', contextSampleRate: 16000, settings: {} }));
            let n = 0;
            micTimer = setInterval(() => {
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
    const pages = [await open(), await open(), await open()];
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    pages[0].ws.send(start);
    pages[1].ws.send(start);
    await wait(300);
    pages[2].ws.send(start);
    await wait(2200);
    expect(pages.map((p) => p.got.session)).toEqual([1, 1, 0]);
    expect(pages[2].got.full).toMatch(/full: 2 of 2/);
    expect(pages[0].got.plays).toBeGreaterThan(0);
    expect(pages[1].got.plays).toBeGreaterThan(0);
    for (const p of pages) expect(p.got.users.at(-1)).toEqual({ type: 'users', pages: 3, sessions: 2, max: 2 });
    pages[0].ws.send(JSON.stringify({ type: 'stop' }));
    pages[1].ws.send(JSON.stringify({ type: 'stop' }));
    await Promise.all([pages[0].saved, pages[1].saved]);
    expect(pages[0].got.saved).not.toBe(pages[1].got.saved);
    await wait(50);
    expect(pages[2].got.users.at(-1)).toEqual({ type: 'users', pages: 3, sessions: 0, max: 2 });
    pages[0].ws.close();
    await wait(100);
    expect(pages[2].got.users.at(-1)).toEqual({ type: 'users', pages: 2, sessions: 0, max: 2 });
    for (const p of pages) p.ws.close();
    cap = 4;
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
