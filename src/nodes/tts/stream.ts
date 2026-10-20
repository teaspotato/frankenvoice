import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { int16BytesFromF32 } from '../audio';
import { addTail } from './tail';

interface StreamParams {
  model: string;
  voice: string;
  sampleRate: number;
  firstClauseChars: number;
  maxFlushesPerMin: number;
  alignWords: boolean;
}

/** Where the audio comes from. Both speak Speak/Flush/Flushed/Close over one WebSocket per reply. */
interface Backend {
  type: string;
  title: string;
  description: string;
  /** Channel of the TTS socket. */
  channel: string;
  request(p: StreamParams): Json;
  /**
   * Text streams to the model word by word (Kyutai), so a Flush only marks where a segment ends;
   * the model keeps speaking across it. Otherwise (Deepgram) text goes out a segment at a time and
   * each Flush synthesizes it.
   */
  wordStreaming: boolean;
  /** The server sends `Words` with the reply's word timings itself (else Deepgram ASR finds them). */
  ownWords: boolean;
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

const clauseParam = {
  key: 'firstClauseChars',
  label: 'Speak the first clause after (chars)',
  type: 'number',
  default: 20,
  description: 'Start speaking at the first comma, colon or dash this far into a reply instead of waiting for the end of its first sentence (0 = whole sentences only).',
} as const;

const deepgramBackend: Backend = {
  type: 'tts.stream',
  title: 'TTS (Deepgram streaming)',
  description: 'Deepgram Aura over a WebSocket: LLM text streams in, audio streams back, one connection per reply.',
  channel: 'deepgram-tts-ws',
  request: (p) => ({ path: '/v1/speak', query: { model: p.model, encoding: 'linear16', sample_rate: p.sampleRate } }),
  wordStreaming: false,
  ownWords: false,
};

const kyutaiBackend: Backend = {
  type: 'tts.kyutai',
  title: 'TTS (Kyutai on the GPU box)',
  description: 'Kyutai TTS on the GPU box over a WebSocket: words stream in as the LLM writes them, audio and word timings stream back.',
  channel: 'gpubox-tts',
  request: (p) => ({ path: '/v1/tts', query: (p.voice ? { voice: p.voice } : {}) as JsonObject }),
  wordStreaming: true,
  ownWords: true,
};

/**
 * Streaming TTS over one WebSocket per reply, opened when the LLM starts (so
 * the connection is up before the first token).
 *
 * Deepgram Aura: each sentence (or, at the start of a reply, the first clause)
 * is sent with `Speak` and spoken with `Flush`, and audio streams back on the
 * same socket in order, each flush ending with `Flushed`. Deepgram only
 * synthesizes on Flush and allows 20 flushes a minute, so past that, sentences
 * are held and go out together with the next allowed flush.
 *
 * Kyutai (GPU box): whole words go out as soon as the LLM writes them, and the
 * model speaks across sentences; `Flush` only marks a segment's end, and
 * `Close` ends the text. Word timings come back on the same socket.
 *
 * Raw socket messages and audio bytes are recorded, so replay is exact.
 */
function streamingTts(b: Backend): NodeDef<StreamParams> {
  return {
    type: b.type,
    title: b.title,
    category: 'tts',
    description: b.description,
    inputs: [
      { name: 'text', type: 'text' },
      { name: 'control', type: 'control' },
    ],
    outputs: [{ name: 'audio', type: 'audio' }],
    params: b.wordStreaming
      ? [
          { key: 'voice', label: 'Voice', type: 'string', default: '', description: 'Kyutai voice (a path in kyutai/tts-voices); empty uses the server’s default.' },
          { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000, description: 'What the server sends (Kyutai: 24 kHz).' },
          { ...clauseParam, default: 0, description: 'Kyutai gets words as they come, so this only sets where the first segment ends on the dashboard (0 = at the first sentence end).' },
          {
            key: 'alignWords',
            label: 'Use word timings',
            type: 'boolean',
            default: true,
            description: 'Pass on the word timings the server sends, so a cut-off reply resumes from the start of the word it stopped in.',
          },
        ]
      : [
          { key: 'model', label: 'Voice model', type: 'string', default: 'aura-2-thalia-en' },
          { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000 },
          clauseParam,
          { key: 'maxFlushesPerMin', label: 'Max flushes per minute', type: 'number', default: 20, description: 'Deepgram’s limit.' },
          {
            key: 'alignWords',
            label: 'Find word timings',
            type: 'boolean',
            default: true,
            description: 'Transcribe the reply’s own audio (Deepgram nova-3) for word timings, so a cut-off reply resumes from the start of the word it stopped in.',
          },
        ],
    channels: b.wordStreaming
      ? [{ name: b.channel, description: 'GPU box /v1/tts WebSocket: Speak/Flush/Close out, raw 16-bit PCM, Words, Flushed and Done in' }]
      : [
          { name: b.channel, description: 'Deepgram /v1/speak WebSocket: Speak/Flush out, raw 16-bit PCM and Flushed in' },
          { name: 'deepgram', description: 'Deepgram /v1/listen WebSocket on the reply’s own audio, for word timings (tts.words)' },
        ],
    requires: () => [b.wordStreaming ? 'gpubox' : 'deepgram'],
    create(ctx) {
      const p = ctx.params;
      /** `held`: audio of a segment still open (word streaming), kept until its text is known. */
      type Seg = { seg: number; text: string; started: boolean; held?: Float32Array[] };
      type Reply = {
        rid: string;
        io: ReturnType<typeof ctx.io.open>;
        /** LLM text not in a flushed segment yet. */
        buf: string;
        /** Word streaming: how much of `buf` already went out with Speak. */
        spoken: number;
        /** Flushed segments whose audio has not all arrived, oldest first; with word streaming, the last may still be open. */
        segs: Seg[];
        /** Word streaming: the segment taking text now, not flushed yet. */
        open?: Seg;
        segNo: number;
        remainder: Uint8Array;
        llmDone: boolean;
        align?: Aligner;
        /** Word streaming: the socket's audio (samples from its first) placed on the media track, for the server's word times. */
        pieces: { a: number; m: number; n: number }[];
        received: number;
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
        if (b.wordStreaming) {
          for (const l of todo) if (!waiting.includes(l)) waiting.push(l);
          return pump();
        }
        const per = Math.max(1, p.maxFlushesPerMin);
        for (let i = 0; i < todo.length; i += per) renderBatch(todo.slice(i, i + per));
      };

      /**
       * Kyutai speaks one utterance per socket and the GPU box renders one at a time, so lines go one
       * by one and wait while a reply is speaking. Dozens of sockets at once also held up the ASR
       * socket: the browser opens one WebSocket to a host at a time.
       */
      const waiting: string[] = [];
      let rendering = false;
      const pump = () => {
        if (!b.wordStreaming || rendering || reply || !waiting.length) return;
        rendering = true;
        let done = false;
        renderBatch([waiting.shift()!], () => {
          if (done) return;
          done = true;
          rendering = false;
          pump();
        });
      };

      /** Lines already re-sent once after their socket closed early. */
      const retried = new Set<string>();
      const renderBatch = (todo: string[], done?: () => void) => {
        const queue = [...todo];
        let chunks: Float32Array[] = [];
        const decode = decoder();
        const io = ctx.io.open(
          b.channel,
          b.request(p),
          {
            onRecv(data: Json, bin) {
              if (bin) return void chunks.push(decode(bin));
              if ((data as JsonObject | null)?.type !== 'Flushed') return;
              const line = queue.shift();
              if (line) canned.set(line, chunks);
              chunks = [];
              ctx.setState('canned', canned.size);
              if (!queue.length && !io.closed) {
                if (!b.wordStreaming) io.send({ type: 'Close' });
                io.close();
                done?.();
              }
            },
            onClose(err) {
              // Closed before every line came back: try those once more.
              const left = queue.filter((l) => !retried.has(l));
              queue.length = 0;
              if (!left.length) return done?.();
              ctx.log('warn', 'canned lines not rendered; retrying', { lines: left, error: (err ?? null) as Json });
              for (const l of left) retried.add(l);
              renderBatch(left, done);
            },
          },
        );
        for (const line of todo) {
          io.send({ type: 'Speak', text: line });
          io.send({ type: 'Flush' });
        }
        if (b.wordStreaming) io.send({ type: 'Close' });
      };

      const emitAudio = (r: Reply, s: Seg, samples: Float32Array) => {
        if (!s.started) {
          s.started = true;
          ctx.emit('audio', { type: 'tts.segment_start', rid: r.rid, seg: s.seg, text: s.text });
        }
        if (!samples.length) return;
        const ref = ctx.media.append('audio', samples, p.sampleRate);
        ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples, rid: r.rid, seg: s.seg });
        if (b.ownWords) return ref;
        // Opened with the reply's first audio: Deepgram closes a socket that gets none for 10 s.
        if (p.alignWords && !r.align) r.align = align(r.rid);
        const a = r.align;
        if (a && !a.io.closed) {
          a.pieces.push({ a: a.sent, m: ref.start, n: samples.length });
          a.sent += samples.length;
          a.io.send(null, { bin: int16BytesFromF32(samples), media: { track: ref.track, start: ref.start, end: ref.end }, logBin: false });
        }
        return ref;
      };

      /** Socket audio onto the track, noting where it went (the server's word times count socket samples). */
      const deliver = (r: Reply, s: Seg, out: Float32Array) => {
        const ref = emitAudio(r, s, out);
        if (ref) r.pieces.push({ a: r.received, m: ref.start, n: out.length });
        r.received += out.length;
      };

      /** Seconds of reply audio to a media track sample, through the pieces it was placed in. */
      const placeOn = (pieces: { a: number; m: number; n: number }[], sec: number) => {
        const x = Math.round(sec * p.sampleRate);
        const piece = pieces.findLast((q) => q.a <= x) ?? pieces[0];
        return piece ? piece.m + Math.min(x - piece.a, piece.n) : x;
      };

      const align = (rid: string): Aligner => {
        const a: Aligner = { sent: 0, pieces: [], io: undefined as never };
        const toMedia = (sec: number) => placeOn(a.pieces, sec);
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
          // With word streaming, Close (end of text) went out when the LLM finished.
          if (!b.wordStreaming) r.io.send({ type: 'Close' });
          r.io.close();
        }
        // Deepgram sends the last words and then closes; they're still wanted while the reply plays.
        if (r.align && !r.align.io.closed) r.align.io.send({ type: 'CloseStream' });
        ctx.emit('audio', { type: 'tts.response_end', rid: r.rid });
        ctx.setState('status', 'idle');
        pump();
      };

      /** Send whatever text is ready, flushing within Deepgram's rate limit. */
      const send = (r: Reply, all: boolean, end = all) => {
        if (b.wordStreaming) return sendWords(r, all, end);
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

      /**
       * Word streaming: whole words go out as soon as they are written (all of the text when `all`),
       * and a Flush marks each segment's end. `end` closes the text, so the model finishes speaking.
       */
      const sendWords = (r: Reply, all: boolean, end: boolean) => {
        const ready = all ? r.buf.length : Math.max(r.spoken, r.buf.search(/\s\S*$/) + 1);
        if (ready > r.spoken) {
          const text = r.buf.slice(r.spoken, ready);
          if (!r.open && text.trim()) {
            r.open = { seg: r.segNo++, text: '', started: false };
            r.segs.push(r.open);
          }
          r.io.send({ type: 'Speak', text });
          r.spoken = ready;
          ctx.setState('status', 'synthesizing');
        }
        for (;;) {
          const i = all ? r.spoken : cutAt(r.buf, r.open?.seg === 0 ? p.firstClauseChars : 0);
          if (i <= 0 || i > r.spoken) break;
          const text = r.buf.slice(0, i).trim();
          r.buf = r.buf.slice(i);
          r.spoken -= i;
          if (r.open) {
            const s = r.open;
            s.text = text;
            r.open = undefined;
            r.io.send({ type: 'Flush' });
            // Its audio waited for the text, which tts.segment_start carries.
            for (const out of s.held?.splice(0) ?? []) deliver(r, s, out);
          }
          if (r.spoken > 0 && r.buf.slice(0, r.spoken).trim()) {
            r.open = { seg: r.segNo++, text: '', started: false };
            r.segs.push(r.open);
          }
          if (all) break;
        }
        if (end) r.io.send({ type: 'Close' });
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
          b.channel,
          b.request(p),
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
                if (s === r.open) (s.held ??= []).push(out);
                else deliver(r, s, out);
                return;
              }
              const m = (data ?? {}) as JsonObject;
              if (m.type === 'Words' && b.ownWords) {
                if (!p.alignWords) return;
                const words = ((m.words ?? []) as { word: string; start: number; end: number }[]).map((w) => ({
                  word: w.word,
                  start: placeOn(r.pieces, w.start),
                  end: placeOn(r.pieces, w.end),
                }));
                if (words.length) ctx.emit('audio', { type: 'tts.words', rid: r.rid, words: words as unknown as Json });
                return;
              }
              if (m.type === 'Flushed') {
                const s = r.segs.shift();
                if (s) {
                  if (!s.started) emitAudio(r, s, new Float32Array());
                  ctx.emit('audio', { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
                }
                if (r.align && !r.align.io.closed) r.align.io.send({ type: 'Finalize' });
                if (r.llmDone && !r.segs.length && !r.buf) finish(r);
              } else if (m.type === 'Warning' || m.type === 'Error') ctx.log('warn', `TTS ${String(m.type)}`, m);
            },
            onClose(data) {
              if (reply !== r) return;
              if (data && (data as { error?: unknown }).error) ctx.log('error', 'TTS connection failed', data);
              for (const s of r.segs) {
                for (const out of s.held?.splice(0) ?? []) deliver(r, s, out);
                ctx.emit('audio', { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
              }
              r.segs = [];
              finish(r);
            },
          },
        );
        r = { rid, io, buf: '', spoken: 0, segs: [], segNo: 0, remainder: new Uint8Array(), llmDone: false, pieces: [], received: 0 };
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
        pump();
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
              // An opener is said as soon as it arrives, whatever its punctuation; the reply follows it.
              send(r, opener, false);
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
}

export const streamingTtsNode = streamingTts(deepgramBackend);
export const kyutaiTtsNode = streamingTts(kyutaiBackend);
