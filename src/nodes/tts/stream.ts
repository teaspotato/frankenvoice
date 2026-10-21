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
  title: 'TTS on the GPU box (Kyutai or Qwen3-TTS)',
  category: 'tts',
  description:
    'Streaming TTS on the GPU box over a WebSocket: words stream in as the LLM writes them, audio and word timings stream back. The box runs Kyutai or Qwen3-TTS (setup.sh tts kyutai|qwen); the stage shows which as “engine”. Voice directions (llm.style) reach Qwen3-TTS 1.7B only.',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    {
      key: 'engine',
      label: 'Engine on the box',
      type: 'select',
      options: ['qwen', 'kyutai'],
      default: 'qwen',
      description: 'Qwen3-TTS (1.7B, takes voice directions) or Kyutai. They don’t fit on the GPU together: picking one swaps the box’s TTS worker (about 30 to 90 s, shown on this stage) before the next session.',
    },
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
      /** Text went out on the socket (so the server sends Done and closes it when it has said it all). */
      spoke: boolean;
      /** The last segment whose Flushed came: the audio after it (the model's last frames) is still its. */
      tail?: Seg;
      /** Output waiting for the replies before it to finish playing out, so it queues after theirs. */
      queue: Msg[];
      /** In case the server never closes the socket after the text's end. */
      guard?: { clear(): void };
      /** Played out or cut; it leaves `line` once the replies before it have. */
      ended?: boolean;
      /** An opener said on a socket of its own, ahead of this reply in `line`. */
      opener?: Reply;
      /** This is such an opener: part of a reply (its rid), which alone ends or cancels the reply. */
      part?: boolean;
    };
    /** Held replies by rid, until released or dropped (they may finish rendering before). */
    const held = new Map<string, Reply>();
    /** The reply taking text now. */
    let reply: Reply | undefined;
    /**
     * Replies in the order they play: ones whose text is complete and whose audio is still coming
     * (the model speaks on for a while after its last Flushed; a new reply doesn't cut them off),
     * then the current one. Only the first emits as it goes; the others' output waits behind it.
     */
    const line: Reply[] = [];
    const live = (r: Reply) => !r.ended && line.includes(r);
    /** Emits now if no older reply is still playing out, else after it. */
    const emitNow = (r: Reply, m: Msg) => (line.length && line[0] !== r && line.includes(r) ? void r.queue.push(m) : void ctx.emit('audio', m));
    /** A reply's output: emitted, or kept while the reply is held. */
    const out = (r: Reply, m: Msg) => (r.hold ? void r.hold.push(m) : emitNow(r, m));
    /** A reply played out or was cut: once it heads the line it leaves, and the next one's waiting output follows. */
    const retire = (r: Reply) => {
      r.ended = true;
      r.guard?.clear();
      r.guard = undefined;
      if (reply === r) reply = undefined;
      while (line[0]?.ended) {
        line.shift();
        const n = line[0];
        if (!n) break;
        const q = n.queue;
        n.queue = [];
        for (const m of q) ctx.emit('audio', m);
      }
    };
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

    /** The reply has played out: the server said all of it and closed (Done), or it said nothing. */
    const finish = (r: Reply) => {
      if (!live(r)) return;
      // Close (end of text) went out when the LLM finished.
      if (!r.io.closed) r.io.close();
      if (!r.part) out(r, { type: 'tts.response_end', rid: r.rid });
      retire(r);
      if (!reply) ctx.setState('status', 'idle');
    };
    /** How long to wait for the server's Done after the last Flushed before closing anyway. */
    const DONE_GRACE_MS = 3000;

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
        r.spoke = true;
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

    const start = (rid: string): Reply => {
      // Set right after the socket opens; its handlers only run later.
      let r!: Reply;
      const io = ctx.io.open(
        CHANNEL,
        request(),
        {
          onRecv(data: Json, bin) {
            if (!live(r)) return;
            if (bin) {
              // After the last Flushed the model still speaks for a few frames: that audio ends its segment.
              const s = r.segs[0] ?? r.tail;
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
            // The box says which engine answers the socket (Kyutai or Qwen3-TTS, swapped with setup.sh tts).
            if (m.type === 'Ready' && typeof m.engine === 'string') ctx.setState('engine', [m.engine, m.model].filter(Boolean).join(' '));
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
                r.tail = s;
              }
              // The reply ends on Done (the socket's close), once its last frames have arrived.
              if (r.llmDone && !r.segs.length && !r.buf) awaitDone(r);
            } else if (m.type === 'Warning' || m.type === 'Error') ctx.log('warn', `TTS ${String(m.type)}`, m);
          },
          onClose(data) {
            if (!live(r)) return;
            if (data && (data as { error?: unknown }).error) ctx.log('error', 'TTS connection failed', data);
            for (const s of r.segs) out(r, { type: 'tts.segment_end', rid: r.rid, seg: s.seg });
            r.segs = [];
            finish(r);
          },
        },
      );
      r = { rid, io, buf: '', spoken: 0, segs: [], segNo: 0, remainder: new Uint8Array(), llmDone: false, pieces: [], received: 0, spoke: false, queue: [] };
      reply = r;
      line.push(r);
      return r;
    };

    /**
     * An opener that isn't rendered ahead, said before the reply has any words: on a socket of its
     * own with its text closed, so the model says it at once instead of holding it for the words
     * after it. It plays as the reply's first segment, and the reply's audio queues behind it.
     */
    const sayOpener = (r: Reply, text: string) => {
      const o = start(r.rid);
      reply = r;
      line.splice(line.indexOf(o), 1);
      line.splice(line.indexOf(r), 0, o);
      o.part = true;
      o.segNo = r.segNo++;
      o.llmDone = true;
      o.buf = text;
      r.opener = o;
      send(o, true);
    };

    /** All text said: wait for the server's Done, but not forever. */
    const awaitDone = (r: Reply) => {
      if (r.guard) return;
      r.guard = ctx.setTimer(DONE_GRACE_MS, () => {
        r.guard = undefined;
        if (!live(r)) return;
        ctx.log('warn', 'TTS sent no Done after the reply; closing it');
        finish(r);
      }, 'tts-done');
    };

    const cancelReply = (r: Reply) => {
      if (!live(r)) return;
      r.io.close({ status: 'cancelled' });
      // What it was holding back behind an older reply is never heard.
      r.queue = [];
      if (!r.part) out(r, { type: 'tts.cancelled', rid: r.rid });
      retire(r);
      // Its opener goes with it.
      if (r.opener) cancelReply(r.opener);
      if (!reply) ctx.setState('status', 'idle');
    };
    /** Cuts the current reply; with `all`, the ones still playing out too (a turn cut). */
    const cancel = (all = false) => {
      if (reply) cancelReply(reply);
      // Newest first, so none of their waiting audio is let out on the way.
      if (all) for (const r of [...line].reverse()) cancelReply(r);
    };
    /**
     * A new reply: the current one plays out if its text is complete (and it isn't held), else it
     * is cut. One playing out queues the new reply's audio behind its own.
     */
    const supersede = () => {
      const r = reply;
      if (!r) return;
      // It stays in `line` until it has played out.
      if (r.llmDone && !r.hold) reply = undefined;
      else cancelReply(r);
    };

    return {
      onInput(port, msg: Msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') cancel(true);
          return;
        }
        const r = reply;
        switch (msg.type) {
          case 'llm.start':
            supersede();
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
            for (const m of msgs) emitNow(h, m);
            break;
          }
          case 'llm.drop': {
            const h = held.get(msg.rid as string);
            if (!h) break;
            held.delete(h.rid);
            if (live(h)) cancelReply(h);
            h.hold = undefined;
            break;
          }
          case 'llm.utterance':
            // A new utterance: the one before ends its segment here, and the next ones carry its flags.
            if (!r || r.rid !== msg.rid) return;
            if (r.buf.trim()) send(r, true, false);
            r.flags = { interruptible: msg.interruptible !== false, gate: msg.gate !== false };
            break;
          case 'llm.style':
            // The reply's voice direction, before its first word: the box's Qwen3-TTS takes it as instruct (Kyutai ignores it).
            if (!r || r.rid !== msg.rid || !msg.text) return;
            r.io.send({ type: 'Instruct', text: msg.text as string });
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
            if (opener && !r.buf && !r.spoke && !r.segs.length && !r.hold && !r.opener) return sayOpener(r, msg.text as string);
            r.buf += msg.text as string;
            // An opener is said as soon as it arrives, whatever its punctuation; the reply follows it.
            send(r, opener, false);
            break;
          }
          case 'llm.end':
            if (!r || r.rid !== msg.rid) return;
            r.llmDone = true;
            send(r, true);
            // Nothing said: nothing to wait for. Else it ends on Done.
            if (!r.spoke) finish(r);
            else if (!r.segs.length) awaitDone(r);
            break;
        }
      },
      stop() {
        cancel(true);
        for (const id of [...genCues.keys()]) dropCue(id, false, 'stopped');
      },
    };
  },
};
