import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { addTail, tailAfter } from './tail';

interface StreamParams {
  voice: string;
  sampleRate: number;
  firstClauseChars: number;
  alignWords: boolean;
}

const CHANNEL = 'gpubox-tts';
/** Canned lines: rendered once per box (a server-side cache, else a background job on the box). */
const CANNED = 'tts-canned';

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
 * Kyutai TTS on the GPU box, one WebSocket per reply, opened when the LLM
 * starts (so the connection is up before the first token). Whole words go out
 * as soon as the LLM writes them, and the model speaks across sentences;
 * `Flush` only marks a segment's end, and `Close` ends the text. Audio and
 * word timings come back on the same socket.
 *
 * Raw socket messages and audio bytes are recorded, so replay is exact.
 */
export const kyutaiTtsNode: NodeDef<StreamParams> = {
  type: 'tts.kyutai',
  title: 'TTS (Kyutai on the GPU box)',
  category: 'tts',
  description: 'Kyutai TTS on the GPU box over a WebSocket: words stream in as the LLM writes them, audio and word timings stream back.',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    { key: 'voice', label: 'Voice', type: 'string', default: '', description: 'Kyutai voice (a path in kyutai/tts-voices); empty uses the server’s default.' },
    { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000, description: 'What the server sends (Kyutai: 24 kHz).' },
    {
      key: 'firstClauseChars',
      label: 'Speak the first clause after (chars)',
      type: 'number',
      default: 0,
      description: 'Kyutai gets words as they come, so this only sets where the first segment ends on the dashboard (0 = at the first sentence end).',
    },
    {
      key: 'alignWords',
      label: 'Use word timings',
      type: 'boolean',
      default: true,
      description: 'Pass on the word timings the server sends, so a cut-off reply resumes from the start of the word it stopped in.',
    },
  ],
  channels: [
    { name: CHANNEL, description: 'GPU box /v1/tts WebSocket: Speak/Flush/Close out, raw 16-bit PCM, Words, Flushed and Done in' },
    { name: CANNED, description: "A canned line's audio (16-bit PCM), from the server's cache or rendered on the box in the background" },
  ],
  requires: () => ['gpubox'],
  create(ctx) {
    const p = ctx.params;
    const request = (): Json => ({ path: '/v1/tts', query: (p.voice ? { voice: p.voice } : {}) as JsonObject });
    /** The open segment's `text` grows as words go out; its audio plays at once. */
    /** An utterance's flags (tagged replies), carried by its segments. */
    type Flags = { interruptible: boolean; gate: boolean };
    type Seg = { seg: number; text: string; started: boolean; announced?: string; flags?: Flags };
    type Reply = {
      rid: string;
      io: ReturnType<typeof ctx.io.open>;
      /** LLM text not in a flushed segment yet. */
      buf: string;
      /** How much of `buf` already went out with Speak. */
      spoken: number;
      /** Segments whose audio has not all arrived, oldest first; the last may still be open. */
      segs: Seg[];
      /** The segment taking text now, not flushed yet. */
      open?: Seg;
      segNo: number;
      remainder: Uint8Array;
      llmDone: boolean;
      /** The socket's audio (samples from its first) placed on the media track, for the server's word times. */
      pieces: { a: number; m: number; n: number }[];
      received: number;
      /** The flags of the utterance being written (tagged replies). */
      flags?: Flags;
      /** Held (a speculative reply): what it would emit, kept until it is released or dropped. */
      hold?: Msg[];
    };
    /** Held replies by rid, until released or dropped (they may finish rendering before). */
    const held = new Map<string, Reply>();
    /** A reply's output: emitted, or kept while the reply is held. */
    const out = (r: Reply, m: Msg) => (r.hold ? void r.hold.push(m) : void ctx.emit('audio', m));
    let reply: Reply | undefined;
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

    /** Synthesize each line once, ahead; the audio is kept for when the line is said. */
    const prerender = (lines: string[]) => {
      for (const l of new Set(lines)) if (!canned.has(l) && !waiting.includes(l)) waiting.push(l);
      pump();
    };

    /**
     * One line at a time. The server answers from its cache, or renders on the box as a background
     * job that never holds up a reply (the box preempts it), so lines don't wait for replies.
     */
    const waiting: string[] = [];
    let rendering = false;
    const pump = () => {
      if (rendering || !waiting.length) return;
      rendering = true;
      let done = false;
      render(waiting.shift()!, () => {
        if (done) return;
        done = true;
        rendering = false;
        pump();
      });
    };

    /** Lines already re-sent once after their socket closed early. */
    const retried = new Set<string>();
    const render = (line: string, done: () => void) => {
      const chunks: Float32Array[] = [];
      const decode = decoder();
      ctx.io.open(CANNED, { text: line, voice: p.voice, sampleRate: p.sampleRate }, {
        onRecv(_data: Json, bin) {
          if (bin) chunks.push(decode(bin));
        },
        onClose(err) {
          if (!err && chunks.length) {
            canned.set(line, chunks);
            ctx.setState('canned', canned.size);
            return done();
          }
          // Not rendered: try it once more.
          if (retried.has(line)) return done();
          ctx.log('warn', 'canned line not rendered; retrying', { line, error: (err ?? null) as Json });
          retried.add(line);
          render(line, done);
        },
      });
    };

    const emitAudio = (r: Reply, s: Seg, samples: Float32Array) => {
      if (!s.started) {
        s.started = true;
        s.announced = s.text;
        out(r, { type: 'tts.segment_start', rid: r.rid, seg: s.seg, text: s.text, ...(s.flags ?? {}) });
      }
      if (!samples.length) return;
      const ref = ctx.media.append('audio', samples, p.sampleRate);
      out(r, { type: 'audio', ...ref, sampleRate: p.sampleRate, samples, rid: r.rid, seg: s.seg });
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

    const finish = (r: Reply) => {
      if (reply !== r) return;
      reply = undefined;
      // Close (end of text) went out when the LLM finished.
      if (!r.io.closed) r.io.close();
      out(r, { type: 'tts.response_end', rid: r.rid });
      ctx.setState('status', 'idle');
    };

    /**
     * Whole words go out as soon as they are written (all of the text when `all`), and a Flush
     * marks each segment's end. `end` closes the text, so the model finishes speaking.
     */
    const send = (r: Reply, all: boolean, end = all) => {
      const ready = all ? r.buf.length : Math.max(r.spoken, r.buf.search(/\s\S*$/) + 1);
      if (ready > r.spoken) {
        const text = r.buf.slice(r.spoken, ready);
        if (!r.open && text.trim()) {
          r.open = { seg: r.segNo++, text: '', started: false, flags: r.flags };
          r.segs.push(r.open);
        }
        if (r.open) r.open.text = r.buf.slice(0, ready).trim();
        r.io.send({ type: 'Speak', text });
        // For the scheduler: how many words TTS has been given.
        const words = text.split(/\s+/).filter(Boolean).length;
        if (words) out(r, { type: 'tts.sent', rid: r.rid, words });
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
          r.open = undefined;
          r.io.send({ type: 'Flush' });
          // Its audio may have started with part of the text: tell the rest.
          if (s.started && s.announced !== text) out(r, { type: 'tts.segment_text', rid: r.rid, seg: s.seg, text });
          s.text = text;
        }
        if (r.spoken > 0 && r.buf.slice(0, r.spoken).trim()) {
          r.open = { seg: r.segNo++, text: r.buf.slice(0, r.spoken).trim(), started: false, flags: r.flags };
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
      out(r, { type: 'tts.segment_end', rid: r.rid, seg: s.seg, canned: true });
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

    /**
     * Generated backchannels, by id: a socket of their own (not the reply's), opened while the LLM
     * writes the line. Their audio plays as a cue as it arrives, and the line's last sound is
     * sustained and faded over `tailMs` at its end. Nothing stops one once its audio has started.
     */
    type GenCue = { id: string; io: ReturnType<typeof ctx.io.open>; decode: (b: Uint8Array) => Float32Array; last: Float32Array; tailMs: number; started: boolean; text?: string };
    const genCues = new Map<string, GenCue>();
    const prepCue = (id: string) => {
      if (genCues.has(id)) return genCues.get(id)!;
      let c!: GenCue;
      const io = ctx.io.open(CHANNEL, request(), {
        onRecv(_data: Json, bin) {
          if (!bin || genCues.get(id) !== c) return;
          const out = c.decode(bin);
          if (!out.length) return;
          c.started = true;
          // The last 80 ms seen, for the tail.
          const keep = Math.round(p.sampleRate * 0.08);
          const all = new Float32Array(c.last.length + out.length);
          all.set(c.last);
          all.set(out, c.last.length);
          c.last = all.slice(Math.max(0, all.length - keep));
          const ref = ctx.media.append('audio', out, p.sampleRate);
          ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples: out, rid: id, cue: true });
        },
        onClose(data) {
          if (genCues.get(id) !== c) return;
          genCues.delete(id);
          if (data && (data as { error?: unknown }).error) ctx.log('warn', 'backchannel TTS failed', data);
          if (c.started && c.tailMs > 0) {
            // Only the sustained part: the line's own audio already went out.
            const ext = tailAfter(c.last, p.sampleRate, c.tailMs);
            if (ext.length) {
              const ref = ctx.media.append('audio', ext, p.sampleRate);
              ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples: ext, rid: id, cue: true });
            }
          }
          ctx.emit('audio', { type: 'tts.cue_end', rid: id, text: c.text ?? '', ...(c.started ? {} : { silent: true }) });
        },
      });
      c = { id, io, decode: decoder(), last: new Float32Array(), tailMs: 0, started: false };
      genCues.set(id, c);
      return c;
    };
    /** Close a generated backchannel's socket; with `unheard`, only if none of its audio has played. */
    const dropCue = (id: string, unheard: boolean, why: string) => {
      const c = genCues.get(id);
      if (!c || (unheard && c.started)) return;
      genCues.delete(id);
      if (!c.io.closed) c.io.close({ status: 'cancelled' });
      ctx.emit('audio', { type: 'tts.cue_dropped', rid: id, why });
    };

    const start = (rid: string) => {
      // Set right after the socket opens; its handlers only run later.
      let r!: Reply;
      const io = ctx.io.open(
        CHANNEL,
        request(),
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
              deliver(r, s, out);
              return;
            }
            const m = (data ?? {}) as JsonObject;
            if (m.type === 'Words') {
              if (!p.alignWords) return;
              const words = ((m.words ?? []) as { word: string; start: number; end: number }[]).map((w) => ({
                word: w.word,
                start: placeOn(r.pieces, w.start),
                end: placeOn(r.pieces, w.end),
              }));
              if (words.length) out(r, { type: 'tts.words', rid: r.rid, words: words as unknown as Json });
              return;
            }
            if (m.type === 'Consumed') {
              // For the scheduler: the words the model still has to read.
              out(r, { type: 'tts.consumed', rid: r.rid, words: m.words ?? 0, queued: m.queued ?? 0 });
              return;
            }
            if (m.type === 'Flushed') {
              const s = r.segs.shift();
              if (s) {
                if (!s.started) emitAudio(r, s, new Float32Array());
                out(r, { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
              }
              if (r.llmDone && !r.segs.length && !r.buf) finish(r);
            } else if (m.type === 'Warning' || m.type === 'Error') ctx.log('warn', `TTS ${String(m.type)}`, m);
          },
          onClose(data) {
            if (reply !== r) return;
            if (data && (data as { error?: unknown }).error) ctx.log('error', 'TTS connection failed', data);
            for (const s of r.segs) out(r, { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
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
      out(r, { type: 'tts.cancelled', rid: r.rid });
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
            if (msg.held) {
              // Rendered now, heard only once released.
              reply!.hold = [];
              held.set(msg.rid as string, reply!);
            }
            break;
          case 'llm.release': {
            const h = held.get(msg.rid as string);
            if (!h) break;
            held.delete(h.rid);
            const msgs = h.hold ?? [];
            h.hold = undefined;
            for (const m of msgs) ctx.emit('audio', m);
            break;
          }
          case 'llm.drop': {
            const h = held.get(msg.rid as string);
            if (!h) break;
            held.delete(h.rid);
            if (reply === h) cancel();
            h.hold = undefined;
            break;
          }
          case 'llm.utterance':
            // A new utterance: the one before ends its segment here, and the next ones carry its flags.
            if (!r || r.rid !== msg.rid) return;
            if (r.buf.trim()) send(r, true, false);
            r.flags = { interruptible: msg.interruptible !== false, gate: msg.gate !== false };
            break;
          case 'opener.lines':
          case 'bc.lines':
            prerender(msg.lines as string[]);
            break;
          case 'bc.prep':
            prepCue(msg.id as string);
            break;
          case 'bc.drop':
            dropCue(msg.id as string, !!msg.unheard, String(msg.why ?? 'dropped'));
            break;
          case 'bc.say': {
            if (msg.gen) {
              const c = prepCue(msg.id as string);
              c.text = String(msg.line);
              c.tailMs = Number(msg.tailMs ?? 0);
              c.io.send({ type: 'Speak', text: c.text });
              c.io.send({ type: 'Close' });
              break;
            }
            dropCue(msg.id as string, false, 'canned line said instead');
            // The take, or else the line itself if that take never rendered, or else rendered now on a cue socket.
            const text = (canned.has(msg.take as string) ? msg.take : msg.line) as string;
            if (!canned.has(text)) {
              const c = prepCue(msg.id as string);
              c.text = text;
              c.tailMs = Number(msg.tailMs ?? 0);
              c.io.send({ type: 'Speak', text });
              c.io.send({ type: 'Close' });
              break;
            }
            playCue(msg.id as string, text, Number(msg.tailMs ?? 0));
            break;
          }
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
        for (const id of [...genCues.keys()]) dropCue(id, false, 'stopped');
      },
    };
  },
};
