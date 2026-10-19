import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';

interface StreamParams {
  model: string;
  sampleRate: number;
  firstClauseChars: number;
  maxFlushesPerMin: number;
}

/** Where to cut `buf` for the next segment, or -1 to wait for more text. */
export function cutAt(buf: string, firstClauseChars: number): number {
  const sentence = /[.!?。！？…]+["')\]]*(\s+|$)/g;
  let m: RegExpExecArray | null;
  while ((m = sentence.exec(buf))) {
    // A boundary at the very end might still be "3." of "3.5"; wait for more text.
    if (m.index + m[0].length === buf.length && !m[1]) break;
    return m.index + m[0].length;
  }
  if (firstClauseChars > 0) {
    // The reply's first words: a clause is enough to start speaking.
    const clause = /[,;:—–]\s+/g;
    while ((m = clause.exec(buf))) if (m.index >= firstClauseChars) return m.index + m[0].length;
  }
  return -1;
}

/**
 * Deepgram Aura over one WebSocket per reply, opened when the LLM starts (so
 * the connection is up before the first token). Text goes out as it streams
 * in: each sentence (or, at the start of a reply, the first clause) is sent
 * with `Speak` and spoken with `Flush`, and audio streams back on the same
 * socket in order, each flush ending with `Flushed`. No request per sentence.
 * Deepgram only synthesizes on Flush and allows 20 flushes a minute, so past
 * that, sentences are held and go out together with the next allowed flush.
 * Raw socket messages and audio bytes are recorded, so replay is exact.
 */
export const streamingTtsNode: NodeDef<StreamParams> = {
  type: 'tts.stream',
  title: 'TTS (Deepgram streaming)',
  category: 'tts',
  description: 'Deepgram Aura over a WebSocket: LLM text streams in, audio streams back, one connection per reply.',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    { key: 'model', label: 'Voice model', type: 'string', default: 'aura-2-thalia-en' },
    { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000 },
    {
      key: 'firstClauseChars',
      label: 'Speak the first clause after (chars)',
      type: 'number',
      default: 20,
      description: 'Start speaking at the first comma, colon or dash this far into a reply instead of waiting for the end of its first sentence (0 = whole sentences only).',
    },
    { key: 'maxFlushesPerMin', label: 'Max flushes per minute', type: 'number', default: 20, description: 'Deepgram’s limit.' },
  ],
  channels: [{ name: 'deepgram-tts-ws', description: 'Deepgram /v1/speak WebSocket: Speak/Flush out, raw 16-bit PCM and Flushed in' }],
  requires: () => ['deepgram'],
  create(ctx) {
    const p = ctx.params;
    type Seg = { seg: number; text: string; started: boolean };
    type Reply = {
      rid: string;
      io: ReturnType<typeof ctx.io.open>;
      /** LLM text not sent yet. */
      buf: string;
      /** Flushed segments whose audio has not all arrived, oldest first. */
      segs: Seg[];
      segNo: number;
      remainder: Uint8Array;
      llmDone: boolean;
    };
    let reply: Reply | undefined;
    /** When recent flushes went out, for Deepgram's per-minute limit. */
    let flushes: number[] = [];

    const emitAudio = (r: Reply, s: Seg, samples: Float32Array) => {
      if (!s.started) {
        s.started = true;
        ctx.emit('audio', { type: 'tts.segment_start', rid: r.rid, seg: s.seg, text: s.text });
      }
      if (!samples.length) return;
      const ref = ctx.media.append('audio', samples, p.sampleRate);
      ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples, rid: r.rid, seg: s.seg });
    };

    const finish = (r: Reply) => {
      if (reply !== r) return;
      reply = undefined;
      if (!r.io.closed) {
        r.io.send({ type: 'Close' });
        r.io.close();
      }
      ctx.emit('audio', { type: 'tts.response_end', rid: r.rid });
      ctx.setState('status', 'idle');
    };

    /** Send whatever text is ready, flushing within Deepgram's rate limit. */
    const send = (r: Reply, all: boolean) => {
      let text = '';
      for (;;) {
        const i = all ? r.buf.length : cutAt(r.buf, r.segNo === 0 && !text ? p.firstClauseChars : 0);
        if (i <= 0) break;
        text += r.buf.slice(0, i);
        r.buf = r.buf.slice(i);
        if (all) break;
      }
      text = text.trim();
      if (!text) return;
      flushes = flushes.filter((t) => ctx.now() - t < 60_000);
      if (!all && flushes.length >= p.maxFlushesPerMin) {
        // Over the limit: hold it, it goes with the next flush that is allowed.
        r.buf = `${text} ${r.buf}`;
        ctx.log('debug', 'flush held back: Deepgram allows only so many per minute', { held: text });
        return;
      }
      flushes.push(ctx.now());
      r.segs.push({ seg: r.segNo++, text, started: false });
      r.io.send({ type: 'Speak', text });
      r.io.send({ type: 'Flush' });
      ctx.setState('status', 'synthesizing');
    };

    const start = (rid: string) => {
      // Set right after the socket opens; its handlers only run later.
      let r!: Reply;
      const io = ctx.io.open(
        'deepgram-tts-ws',
        { path: '/v1/speak', query: { model: p.model, encoding: 'linear16', sample_rate: p.sampleRate } },
        {
          onRecv(data: Json, bin) {
            if (reply !== r) return;
            if (bin) {
              const s = r.segs[0];
              if (!s) return;
              const all = new Uint8Array(r.remainder.length + bin.length);
              all.set(r.remainder);
              all.set(bin, r.remainder.length);
              const usable = all.length - (all.length % 2);
              r.remainder = all.slice(usable);
              const view = new DataView(all.buffer, all.byteOffset, usable);
              const out = new Float32Array(usable / 2);
              for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
              emitAudio(r, s, out);
              return;
            }
            const m = (data ?? {}) as JsonObject;
            if (m.type === 'Flushed') {
              const s = r.segs.shift();
              if (s) {
                if (!s.started) emitAudio(r, s, new Float32Array());
                ctx.emit('audio', { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
              }
              if (r.llmDone && !r.segs.length && !r.buf) finish(r);
            } else if (m.type === 'Warning' || m.type === 'Error') ctx.log('warn', `Deepgram TTS ${String(m.type)}`, m);
          },
          onClose(data) {
            if (reply !== r) return;
            if (data && (data as { error?: unknown }).error) ctx.log('error', 'TTS connection failed', data);
            for (const s of r.segs) ctx.emit('audio', { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
            r.segs = [];
            finish(r);
          },
        },
      );
      r = { rid, io, buf: '', segs: [], segNo: 0, remainder: new Uint8Array(), llmDone: false };
      reply = r;
    };

    const cancel = () => {
      const r = reply;
      if (!r) return;
      reply = undefined;
      r.io.close({ status: 'cancelled' });
      ctx.emit('audio', { type: 'tts.cancelled', rid: r.rid });
      ctx.setState('status', 'idle');
    };

    return {
      onInput(port, msg: Msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') cancel();
          return;
        }
        const r = reply;
        switch (msg.type) {
          case 'llm.start':
            cancel();
            start(msg.rid as string);
            break;
          case 'llm.delta':
            if (!r || r.rid !== msg.rid) return;
            r.buf += msg.text as string;
            send(r, false);
            break;
          case 'llm.end':
            if (!r || r.rid !== msg.rid) return;
            r.llmDone = true;
            send(r, true);
            if (!r.segs.length) finish(r);
            break;
        }
      },
      stop() {
        cancel();
      },
    };
  },
};
