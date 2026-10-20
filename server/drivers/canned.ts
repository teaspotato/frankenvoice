// Canned lines (backchannels, openers, "you go first") rendered once per box: a disk cache keyed by
// everything that decides the audio (text, voice, cfg, model settings, sample rate). A miss renders
// on the GPU box as a background job, which any reply preempts. The node sees one exchange either
// way, its audio recorded like any other IO, so a replay serves it from the recording and never
// reads the cache.

import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { WebSocket } from 'ws';
import type { LiveDriver } from '../../src/core/world';
import type { Json } from '../../src/core/types';
import type { Upstreams } from './index';

const errorJson = (e: unknown): Json => ({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });

type TtsConfig = { model: string; n_q: number; temp: number; voice: string; cfg: number; sample_rate: number };

export function cannedDriver(up: Upstreams, cacheDir: string): LiveDriver {
  // The box's TTS settings, asked once (again after a failure).
  let config: Promise<TtsConfig> | undefined;
  const boxConfig = () => {
    const u = up('gpubox');
    if (u.missing) return Promise.reject(new Error(u.missing));
    config ??= fetch(`${u.base!.replace(/^ws(s?):/, 'http$1:')}/v1/tts/config`, { headers: u.headers, signal: AbortSignal.timeout(5000) })
      .then((r) => (r.ok ? (r.json() as Promise<TtsConfig>) : Promise.reject(new Error(`tts config: HTTP ${r.status}`))))
      .catch((e) => {
        config = undefined;
        throw e;
      });
    return config;
  };

  /** Render on the box in the background: Speak, Flush, Close; the audio once Done. */
  const render = (text: string, voice: string, signal: { closed: boolean }) =>
    new Promise<{ pcm: Uint8Array; stats: Json }>((resolve, reject) => {
      const u = up('gpubox');
      if (u.missing) return reject(new Error(u.missing));
      const url = new URL(`${u.base!}/v1/tts`);
      url.searchParams.set('priority', 'background');
      if (voice) url.searchParams.set('voice', voice);
      const ws = new WebSocket(url, { headers: u.headers });
      const chunks: Buffer[] = [];
      let stats: Json = null;
      ws.on('open', () => {
        for (const m of [{ type: 'Speak', text }, { type: 'Flush' }, { type: 'Close' }]) ws.send(JSON.stringify(m));
      });
      ws.on('message', (data: Buffer, binary) => {
        if (signal.closed) return ws.close();
        if (binary) return void chunks.push(Buffer.from(data));
        const m = JSON.parse(data.toString('utf8')) as { type?: string; stats?: Json; message?: string };
        if (m.type === 'Done') stats = m.stats ?? null;
        else if (m.type === 'Error') reject(new Error(m.message ?? 'tts error'));
      });
      ws.on('unexpected-response', (_req, res) => (reject(new Error(`tts HTTP ${res.statusCode}`)), ws.terminate()));
      ws.on('error', reject);
      ws.on('close', () => {
        const pcm = Buffer.concat(chunks);
        if (stats !== null && pcm.length) resolve({ pcm: new Uint8Array(pcm), stats });
        else reject(new Error('tts closed before the line was rendered'));
      });
    });

  return (req, sink) => {
    const r = req as { text: string; voice?: string };
    const state = { closed: false };
    (async () => {
      let key: string | undefined;
      let cfg: TtsConfig | undefined;
      try {
        cfg = await boxConfig();
        key = createHash('sha256')
          .update(JSON.stringify({ text: r.text, voice: r.voice || cfg.voice, cfg: cfg.cfg, model: cfg.model, n_q: cfg.n_q, temp: cfg.temp, sr: cfg.sample_rate }))
          .digest('hex')
          .slice(0, 32);
      } catch {
        // Without the box's settings there is no key: render uncached.
      }
      const file = key && join(cacheDir, `${key}.pcm`);
      const hit = file ? await readFile(file).catch(() => undefined) : undefined;
      if (state.closed) return;
      if (hit) {
        sink.recv({ type: 'Canned', cached: true, key: key!, sampleRate: cfg!.sample_rate });
        sink.recv(null, new Uint8Array(hit));
        return sink.close(null);
      }
      const { pcm, stats } = await render(r.text, r.voice ?? '', state);
      if (file) {
        await mkdir(cacheDir, { recursive: true });
        const tmp = `${file}.${process.pid}.tmp`;
        await writeFile(tmp, pcm);
        await rename(tmp, file);
      }
      if (state.closed) return;
      sink.recv({ type: 'Canned', cached: false, key: key ?? null, sampleRate: cfg?.sample_rate ?? null, stats });
      sink.recv(null, pcm);
      sink.close(null);
    })().catch((e) => !state.closed && sink.close(errorJson(e)));
    return {
      send() {},
      close() {
        state.closed = true;
      },
    };
  };
}
