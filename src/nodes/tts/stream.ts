import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { int16BytesFromF32 } from '../audio';
import { addTail } from './tail';

interface StreamParams {
  model: string;
  sampleRate: number;
  firstClauseChars: number;
  maxFlushesPerMin: number;
  alignWords: boolean;
}

/** A word of a reply's audio, in samples of its media track. */
export interface SpokenWord {
  word: string;
  start: number;
  end: number;
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
    {
      key: 'alignWords',
      label: 'Find word timings',
      type: 'boolean',
      default: true,
      description: 'Transcribe the reply’s own audio (Deepgram nova-3) for word timings, so a cut-off reply resumes from the start of the word it stopped in.',
    },
  ],
  channels: [
    { name: 'deepgram-tts-ws', description: 'Deepgram /v1/speak WebSocket: Speak/Flush out, raw 16-bit PCM and Flushed in' },
    { name: 'deepgram', description: 'Deepgram /v1/listen WebSocket on the reply’s own audio, for word timings (tts.words)' },
  ],
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
      align?: Aligner;
    };
    /**
     * The reply's audio streamed to Deepgram ASR. Its times count from the reply's first sample, and
     * `pieces` maps them back to the media track (backchannels share the track, so it has gaps).
     */
    type Aligner = { io: ReturnType<typeof ctx.io.open>; sent: number; pieces: { a: number; m: number; n: number }[] };
    let reply: Reply | undefined;
    /** When recent flushes went out, for Deepgram's per-minute limit. */
    let flushes: number[] = [];
    /** Opener lines (fillers), synthesized ahead so they play at once. */
    const canned = new Map<string, Float32Array[]>();

    const decoder = () => {
      let rest = new Uint8Array();
      return (bin: Uint8Array) => {
        const all = new Uint8Array(rest.length + bin.length);
        all.set(rest);
        all.set(bin, rest.length);
        const usable = all.length - (all.length % 2);
        rest = all.slice(usable);
        const view = new DataView(all.buffer, all.byteOffset, usable);
        const out = new Float32Array(usable / 2);
        for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
        return out;
      };
    };

    /**
     * Synthesize each line once, ahead; the audio is kept for when the line is said. Deepgram limits
     * flushes per connection, so a long list is split over several sockets.
     */
    const prerender = (lines: string[]) => {
      const todo = [...new Set(lines)].filter((l) => !canned.has(l));
      const per = Math.max(1, p.maxFlushesPerMin);
      for (let i = 0; i < todo.length; i += per) renderBatch(todo.slice(i, i + per));
    };

    /** Lines already re-sent once after their socket closed early. */
    const retried = new Set<string>();
    const renderBatch = (todo: string[]) => {
      const queue = [...todo];
      let chunks: Float32Array[] = [];
      const decode = decoder();
      const io = ctx.io.open(
        'deepgram-tts-ws',
        { path: '/v1/speak', query: { model: p.model, encoding: 'linear16', sample_rate: p.sampleRate } },
        {
          onRecv(data: Json, bin) {
            if (bin) return void chunks.push(decode(bin));
            if ((data as JsonObject | null)?.type !== 'Flushed') return;
            const line = queue.shift();
            if (line) canned.set(line, chunks);
            chunks = [];
            ctx.setState('canned', canned.size);
            if (!queue.length && !io.closed) {
              io.send({ type: 'Close' });
              io.close();
            }
          },
          onClose(err) {
            // Closed before every line came back: try those once more.
            const left = queue.filter((l) => !retried.has(l));
            queue.length = 0;
            if (!left.length) return;
            ctx.log('warn', 'canned lines not rendered; retrying', { lines: left, error: (err ?? null) as Json });
            for (const l of left) retried.add(l);
            renderBatch(left);
          },
        },
      );
      for (const line of todo) {
        io.send({ type: 'Speak', text: line });
        io.send({ type: 'Flush' });
      }
    };

    const emitAudio = (r: Reply, s: Seg, samples: Float32Array) => {
      if (!s.started) {
        s.started = true;
        ctx.emit('audio', { type: 'tts.segment_start', rid: r.rid, seg: s.seg, text: s.text });
      }
      if (!samples.length) return;
      const ref = ctx.media.append('audio', samples, p.sampleRate);
      ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples, rid: r.rid, seg: s.seg });
      // Opened with the reply's first audio: Deepgram closes a socket that gets none for 10 s.
      if (p.alignWords && !r.align) r.align = align(r.rid);
      const a = r.align;
      if (a && !a.io.closed) {
        a.pieces.push({ a: a.sent, m: ref.start, n: samples.length });
        a.sent += samples.length;
        a.io.send(null, { bin: int16BytesFromF32(samples), media: { track: ref.track, start: ref.start, end: ref.end }, logBin: false });
      }
    };

    const align = (rid: string): Aligner => {
      const a: Aligner = { sent: 0, pieces: [], io: undefined as never };
      /** Reply-audio seconds to a media track sample. */
      const toMedia = (sec: number) => {
        const x = Math.round(sec * p.sampleRate);
        const piece = a.pieces.findLast((q) => q.a <= x) ?? a.pieces[0];
        return piece ? piece.m + Math.min(x - piece.a, piece.n) : x;
      };
      a.io = ctx.io.open(
        'deepgram',
        { path: '/v1/listen', query: { model: 'nova-3', language: 'en', encoding: 'linear16', sample_rate: p.sampleRate, channels: 1, smart_format: false } },
        {
          onRecv(data: Json) {
            const m = (data ?? {}) as JsonObject;
            if (m.type !== 'Results' || !m.is_final) return;
            const alt = ((m.channel as JsonObject | undefined)?.alternatives as JsonObject[] | undefined)?.[0];
            const words = ((alt?.words ?? []) as { word: string; start: number; end: number }[]).map((w) => ({ word: w.word, start: toMedia(w.start), end: toMedia(w.end) }));
            if (words.length) ctx.emit('audio', { type: 'tts.words', rid, words: words as unknown as Json });
          },
          onClose(data) {
            if (data && (data as { error?: unknown }).error) ctx.log('warn', 'word timings unavailable: Deepgram ASR on the reply failed', data);
          },
        },
      );
      return a;
    };

    const finish = (r: Reply) => {
      if (reply !== r) return;
      reply = undefined;
      if (!r.io.closed) {
        r.io.send({ type: 'Close' });
        r.io.close();
      }
      // Deepgram sends the last words and then closes; they're still wanted while the reply plays.
      if (r.align && !r.align.io.closed) r.align.io.send({ type: 'CloseStream' });
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

    /** Play a canned line's audio as the reply's next segment, without a request. */
    const playCanned = (r: Reply, text: string) => {
      const audio = canned.get(text);
      if (!audio || r.segs.length) return false;
      const s: Seg = { seg: r.segNo++, text, started: false };
      emitAudio(r, s, new Float32Array());
      for (const a of audio) emitAudio(r, s, a);
      ctx.emit('audio', { type: 'tts.segment_end', rid: r.rid, seg: s.seg, canned: true });
      return true;
    };

    /** A backchannel: a rendered line played as a cue, outside any reply. */
    const playCue = (id: string, text: string, tailMs: number) => {
      const audio = canned.get(text);
      if (!audio) return void ctx.log('warn', `backchannel "${text}" isn't rendered yet`);
      // One chunk, so it plays as one piece.
      let samples = new Float32Array(audio.reduce((n, a) => n + a.length, 0));
      let o = 0;
      for (const a of audio) {
        samples.set(a, o);
        o += a.length;
      }
      if (tailMs > 0 && samples.length) samples = addTail(samples, p.sampleRate, tailMs);
      if (samples.length) {
        const ref = ctx.media.append('audio', samples, p.sampleRate);
        ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples, rid: id, cue: true });
      }
      ctx.emit('audio', { type: 'tts.cue_end', rid: id, text });
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
              if (r.align && !r.align.io.closed) r.align.io.send({ type: 'Finalize' });
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
      // The reply may still be finishing its last sound; its words up to here have come back.
      if (r.align && !r.align.io.closed) r.align.io.close({ status: 'cancelled' });
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
          case 'opener.lines':
          case 'bc.lines':
            prerender(msg.lines as string[]);
            break;
          case 'bc.say':
            // The take, or else the line itself if that take never rendered.
            playCue(msg.id as string, (canned.has(msg.take as string) ? msg.take : msg.line) as string, Number(msg.tailMs ?? 0));
            break;
          case 'opener.say':
          case 'llm.delta': {
            if (!r || r.rid !== msg.rid) return;
            const opener = msg.type === 'opener.say';
            if (opener && !r.buf && playCanned(r, (msg.text as string).trim())) return;
            r.buf += msg.text as string;
            // An opener is said as soon as it arrives, whatever its punctuation.
            send(r, opener);
            break;
          }
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
