import type { NodeDef, TimerHandle } from '../core/node';
import type { Json } from '../core/types';
import { isAudioMsg } from '../core/types';
import { bytesFromF32 } from './audio';
import { fadeIn, level, soundEnd, wordStart, type Span } from './cutpoint';
import type { SpokenWord } from './tts/stream';

interface PlayedSegment {
  seg: number;
  text: string;
  /** Session time its first audio plays, and the scheduled end of its audio so far. */
  start?: number;
  end?: number;
  /** When it was announced to start (playback.segment). */
  announced?: number;
  /** Pauses inside it: when, and for how long. */
  gaps?: { at: number; ms: number }[];
}

/** How long `s` was paused before `at`. */
const pausedBefore = (s: PlayedSegment, at: number) => (s.gaps ?? []).reduce((sum, g) => sum + Math.max(0, Math.min(at, g.at + g.ms) - g.at), 0);

/**
 * The words of `segs` played before `at`. A sentence cut off midway is
 * estimated from how far through its audio playback got, rounded to whole words.
 */
export function heardText(segs: PlayedSegment[], at: number): string {
  const out: string[] = [];
  for (const s of segs) {
    if (s.start === undefined || s.start >= at) break;
    if (s.end !== undefined && s.end <= at) {
      out.push(s.text);
      continue;
    }
    const len = s.end !== undefined ? s.end - s.start - pausedBefore(s, s.end) : 0;
    const frac = len > 0 ? (at - s.start - pausedBefore(s, at)) / len : 0;
    const ws = s.text.split(/\s+/).filter(Boolean);
    let chars = 0;
    const part: string[] = [];
    for (const w of ws) {
      // Keep a word once most of it was heard.
      if (chars + w.length / 2 > frac * s.text.length) break;
      part.push(w);
      chars += w.length + 1;
    }
    if (part.length) out.push(part.join(' '));
    break;
  }
  return out.join(' ');
}

/**
 * Plays agent audio. The playback schedule (when each chunk starts and ends,
 * in session time) is computed here, deterministically, and logged; the
 * 'speaker' driver only renders it. That schedule is what later steps use to
 * know what the user has actually heard.
 *
 * Turn-taking can pause the agent (control `pause`) while it judges the user's
 * speech over it: the audio stops and the rest of the reply is held, and
 * `resume` plays it again, with the schedule shifted by the pause
 * (playback.paused / playback.resumed; the held sentences are announced again).
 *
 * A pause or cancel doesn't stop mid-sound: playback runs on to the end of the
 * syllable playing (the next quiet point) and fades out over a short release.
 * A resume picks up from the start of the word it stopped in (word timings
 * from tts.words), fading in, so the word is heard whole.
 */
interface SpeakerParams {
  cutoff: 'syllable' | 'hard';
  maxFinishMs: number;
  releaseMs: number;
  resumeFrom: 'word' | 'cut';
  attackMs: number;
}

/** Played audio kept for going back to the start of a word. */
const HISTORY_MS = 1500;

export const speakerNode: NodeDef<SpeakerParams> = {
  type: 'speaker',
  title: 'Speaker',
  category: 'sink',
  description: 'Schedules and plays audio; reports playback start/end and segment timing.',
  inputs: [
    { name: 'audio', type: 'audio' },
    { name: 'control', type: 'control' },
  ],
  outputs: [
    { name: 'playback', type: 'playback' },
    { name: 'played', type: 'played', description: 'Each chunk of audio with the session time it plays at (echo reference)' },
  ],
  params: [
    {
      key: 'cutoff',
      label: 'Cut off at',
      type: 'select',
      options: ['syllable', 'hard'],
      default: 'syllable',
      description: 'syllable: when paused or cancelled, play on to the end of the sound playing, then fade out. hard: stop at once.',
    },
    { key: 'maxFinishMs', label: 'Finish the sound for at most (ms)', type: 'number', default: 300 },
    { key: 'releaseMs', label: 'Release (ms)', type: 'number', default: 40, description: 'Fade-out at the cut.' },
    {
      key: 'resumeFrom',
      label: 'Resume from',
      type: 'select',
      options: ['word', 'cut'],
      default: 'word',
      description: 'word: replay the word it stopped in from its start (needs tts.words). cut: carry on where it stopped.',
    },
    { key: 'attackMs', label: 'Attack (ms)', type: 'number', default: 10, description: 'Fade-in when resuming.' },
  ],
  channels: [{ name: 'speaker', description: 'Scheduled playback through Web Audio', output: true }],
  create(ctx) {
    let io: ReturnType<typeof ctx.io.open> | undefined;
    let endAt = 0;
    let playing = false;
    let rid: string | null = null;
    let endTimer: TimerHandle | undefined;
    /** Sentences per reply, for working out what was heard. */
    const segs = new Map<string | null, PlayedSegment[]>();
    /** Replies the TTS has sent everything for (or gave up on): playback ends when their audio does. */
    const complete = new Set<string | null>();
    /** Audio sent that may not have played yet, so a pause can hold it back. */
    type Chunk = { at: number; track: string; start: number; end: number; sampleRate: number; samples: Float32Array };
    let queued: Chunk[] = [];
    /** While paused, when it stopped. The schedule stands still meanwhile: new audio queues after `endAt`. */
    let paused: number | null = null;
    const clock = () => paused ?? ctx.now();
    const p = ctx.params;
    /** Word timings per reply, in media samples (tts.words). */
    const words = new Map<string | null, SpokenWord[]>();

    /** The audio of `chunks` from t0 to t1, as one piece (gaps are silence). */
    const spanOf = (chunks: Chunk[], t0: number, t1: number): Span | null => {
      const rate = chunks[0]?.sampleRate;
      if (!rate) return null;
      const samples = new Float32Array(Math.max(0, Math.round(((t1 - t0) / 1000) * rate)));
      for (const c of chunks) {
        if (c.sampleRate !== rate) continue;
        const o = Math.round(((c.at - t0) / 1000) * rate);
        for (let i = Math.max(0, -o); i < c.samples.length && o + i < samples.length; i++) samples[o + i] = c.samples[i];
      }
      return { at: t0, sampleRate: rate, samples };
    };
    /** Session time a media sample of the reply plays at, from the chunks still known. */
    const timeOf = (track: string, pos: number) => {
      const c = queued.find((q) => q.track === track && q.start <= pos && pos < q.end);
      return c ? c.at + ((pos - c.start) / c.sampleRate) * 1000 : undefined;
    };
    /** The media sample playing at session time `t`. */
    const posAt = (t: number) => {
      const c = queued.find((q) => q.at <= t && t < q.at + (q.samples.length / q.sampleRate) * 1000);
      return c ? { track: c.track, pos: c.start + Math.round(((t - c.at) / 1000) * c.sampleRate) } : undefined;
    };

    /**
     * Cutting the reply off at `at`: when the sound ends (`stop`, then a release of `release` ms),
     * and where to resume (`from`, a time already played or `stop` itself).
     */
    const cutPoint = (at: number) => {
      const a = spanOf(queued, at - HISTORY_MS, at + p.maxFinishMs + 50);
      // Nothing sounding at the cut (between words, or out of audio): stop right there.
      if (!a || level(a, at, at + 10) < 1e-3) return { stop: at, release: 0, from: at };
      const stop = p.cutoff === 'syllable' ? soundEnd(a, at, p.maxFinishMs) : at;
      let from = stop;
      if (p.resumeFrom === 'word') {
        // Back to the start of the word playing at the cut.
        const here = posAt(at);
        const w = here && words.get(rid)?.findLast((x) => x.start <= here.pos);
        const est = w && here.pos < w.end + 0.08 * a.sampleRate ? timeOf(here.track, w.start) : undefined;
        if (est !== undefined) from = wordStart(a, at, HISTORY_MS, est);
      }
      return { stop, release: Math.max(0, p.releaseMs), from: Math.min(from, stop) };
    };

    const send = (c: Chunk) => {
      io?.send({ cmd: 'play', at: c.at, sampleRate: c.sampleRate }, {
        bin: bytesFromF32(c.samples),
        media: { track: c.track, start: c.start, end: c.end },
        logBin: false,
      });
      ctx.emit('played', { type: 'audio', track: c.track, start: c.start, end: c.end, sampleRate: c.sampleRate, samples: c.samples, at: c.at });
    };
    const announce = (r: string | null, s: PlayedSegment) =>
      ctx.emit('playback', { type: 'playback.segment', rid: r as string, seg: s.seg, text: s.text, at: s.announced! });

    /** Backchannels playing or recently played; they play on their own output, which a flush doesn't stop. */
    let cues: Chunk[] = [];
    /** The reply's output was flushed at `at`: cut the echo reference there, but put back the cues still sounding. */
    const flushPlayed = (at: number) => {
      ctx.emit('played', { type: 'played.flush', at });
      cues = cues.filter((c) => c.at + (c.samples.length / c.sampleRate) * 1000 > at);
      for (const c of cues) {
        const cut = Math.max(0, Math.round(((at - c.at) / 1000) * c.sampleRate));
        const samples = c.samples.subarray(cut);
        ctx.emit('played', { type: 'audio', track: c.track, start: c.start + cut, end: c.end, sampleRate: c.sampleRate, samples, at: Math.max(at, c.at), cue: true });
      }
    };

    /** Stop the reply's output: at once, or at `stop` with a fade over `release` ms. Returns when the sound ends. */
    const flushAt = (stop: number, release: number) => {
      const now = ctx.now();
      if (stop <= now && release <= 0) io?.send({ cmd: 'flush' });
      else io?.send({ cmd: 'flush', at: stop, releaseMs: release });
      return stop + release;
    };

    const pause = () => {
      const at = ctx.now();
      const cut = cutPoint(at);
      const silent = flushAt(cut.stop, cut.release);
      silentAt = silent;
      endTimer?.clear();
      endTimer = undefined;
      // Keep what hasn't played, from where it resumes; the chunk playing there is split.
      const from = cut.from;
      queued = queued.flatMap((c) => {
        const dur = (c.samples.length / c.sampleRate) * 1000;
        if (c.at + dur <= from) return [];
        if (c.at >= from) return [c];
        const n = Math.round(((from - c.at) / 1000) * c.sampleRate);
        return [{ ...c, at: from, start: c.start + n, samples: c.samples.subarray(n) }];
      });
      paused = from;
      endAt = Math.max(endAt, from);
      ctx.setState('paused', true);
      flushPlayed(silent);
      ctx.emit('playback', { type: 'playback.paused', rid, at: silent, ...(from !== silent ? { from } : {}) });
    };
    /** When the last cut's release has faded out; nothing new plays before. */
    let silentAt = -Infinity;
    const resume = () => {
      const from = paused!;
      const at = Math.max(ctx.now(), silentAt);
      const shift = at - from;
      paused = null;
      endAt += shift;
      ctx.setState('paused', false);
      ctx.emit('playback', { type: 'playback.resumed', rid, at, from, shift });
      for (const s of segs.get(rid) ?? []) {
        if (s.end !== undefined && s.end > from) {
          if (s.start !== undefined && s.start < from) (s.gaps ??= []).push({ at: from, ms: shift });
          else if (s.start !== undefined) s.start += shift;
          s.end += shift;
        }
        if (s.announced !== undefined && s.announced >= from) {
          s.announced += shift;
          announce(rid, s);
        }
      }
      // A replay starts mid-reply: fade it in.
      if (queued[0] && p.attackMs > 0) queued[0] = { ...queued[0], samples: fadeIn(queued[0].samples, Math.round((p.attackMs / 1000) * queued[0].sampleRate)) };
      for (const c of queued) {
        c.at += shift;
        send(c);
      }
      armEndTimer();
    };

    /** Playback volume: below 1 while turn-taking ducks the agent under the user's speech. */
    let gain = 1;
    const setGain = (g: number) => {
      if (g === gain) return;
      gain = g;
      io?.send({ cmd: 'gain', gain: g });
      ctx.setState('ducked', g < 1);
      ctx.emit('playback', { type: 'playback.gain', rid, gain: g, at: ctx.now() });
    };

    const end = () => {
      setGain(1);
      queued = [];
      if (paused !== null) {
        paused = null;
        ctx.setState('paused', false);
      }
      playing = false;
      ctx.setState('playing', false);
      ctx.setState('buffering', false);
      ctx.emit('playback', { type: 'playback.end', rid });
    };

    // One pending timer at a time; if more audio was queued meanwhile it re-arms.
    // Running out of audio before the reply is complete is a gap (the TTS is
    // behind), not the end of playback.
    const armEndTimer = () => {
      if (endTimer) return;
      endTimer = ctx.setTimer(Math.max(0, endAt - ctx.now()), () => {
        endTimer = undefined;
        if (!playing) return;
        if (ctx.now() < endAt) return armEndTimer();
        if (complete.has(rid)) end();
        else ctx.setState('buffering', true);
      }, 'playback-end');
    };

    return {
      start() {
        io = ctx.io.open('speaker', {}, {
          onRecv(data) {
            ctx.setState('device', data);
          },
        });
      },
      onInput(port, msg) {
        if (port === 'control') {
          if (msg.type === 'duck' && playing) setGain(10 ** (-(msg.db as number) / 20));
          else if (msg.type === 'unduck') setGain(1);
          if (msg.type === 'pause' && playing && paused === null) pause();
          else if (msg.type === 'resume' && paused !== null) resume();
          if (msg.type === 'cancel') setGain(1);
          if (msg.type === 'cancel' && playing) {
            const wasPaused = paused !== null;
            // Paused, the sound already stopped and the user heard up to the pause.
            const cut = wasPaused ? null : cutPoint(ctx.now());
            const at = cut ? flushAt(cut.stop, cut.release) : ctx.now();
            if (cut) silentAt = at;
            const heard = heardText(segs.get(rid) ?? [], cut ? cut.stop : clock());
            endTimer?.clear();
            endTimer = undefined;
            playing = false;
            endAt = at;
            queued = [];
            if (wasPaused) {
              paused = null;
              ctx.setState('paused', false);
            }
            ctx.setState('playing', false);
            ctx.emit('playback', { type: 'playback.interrupted', rid, at, heard });
            // Paused, the device was flushed (and the echo reference cut) at the pause.
            if (!wasPaused) flushPlayed(at);
          }
          return;
        }
        if (msg.type === 'tts.segment_start') {
          const r = (msg.rid as string) ?? null;
          if (!segs.has(r)) segs.set(r, []);
          const s: PlayedSegment = { seg: msg.seg as number, text: msg.text as string, announced: Math.max(clock(), endAt) };
          segs.get(r)!.push(s);
          // While paused it is announced on resume, when its time is known.
          if (paused === null || r !== rid) announce(r, s);
          return;
        }
        if (msg.type === 'tts.words') {
          const r = (msg.rid as string) ?? null;
          words.set(r, [...(words.get(r) ?? []), ...(msg.words as unknown as SpokenWord[])]);
          return;
        }
        if (msg.type === 'tts.response_end' || msg.type === 'tts.cancelled') {
          const r = (msg.rid as string) ?? null;
          complete.add(r);
          if (playing && paused === null && r === rid && ctx.now() >= endAt) end();
          return;
        }
        if (!isAudioMsg(msg)) return;
        if (msg.cue) {
          // A backchannel: plays now, over whatever is playing, and starts no reply.
          const at = ctx.now();
          io?.send({ cmd: 'cue', at, sampleRate: msg.sampleRate }, {
            bin: bytesFromF32(msg.samples),
            media: { track: msg.track, start: msg.start, end: msg.end },
            logBin: false,
          });
          ctx.emit('played', { type: 'audio', track: msg.track, start: msg.start, end: msg.end, sampleRate: msg.sampleRate, samples: msg.samples, at, cue: true });
          cues = [...cues.filter((c) => c.at + (c.samples.length / c.sampleRate) * 1000 > at), { at, track: msg.track, start: msg.start, end: msg.end, sampleRate: msg.sampleRate, samples: msg.samples }];
          ctx.emit('playback', { type: 'playback.cue', rid: msg.rid as Json, at, end: at + (msg.samples.length / msg.sampleRate) * 1000 });
          return;
        }
        const r = (msg.rid as string) ?? null;
        if (playing && r !== rid && paused !== null) {
          // A new reply while paused: the held one is dropped, and the new one starts now.
          end();
          endAt = ctx.now();
        }
        const at = Math.max(clock(), endAt);
        const dur = (msg.samples.length / msg.sampleRate) * 1000;
        endAt = at + dur;
        const c: Chunk = { at, track: msg.track, start: msg.start, end: msg.end, sampleRate: msg.sampleRate, samples: msg.samples };
        // Forget what has played, but for the last moments, which a cut may go back into (while paused, nothing after the pause has played).
        queued = queued.filter((q) => q.at + (q.samples.length / q.sampleRate) * 1000 > clock() - (paused === null ? HISTORY_MS : 0));
        queued.push(c);
        if (paused === null) send(c);
        if (playing && r !== rid) {
          end();
          queued = [c];
        }
        if (!playing) {
          // Forget older replies.
          for (const k of [...segs.keys()]) if (k !== r) segs.delete(k);
          for (const k of [...complete]) if (k !== r) complete.delete(k);
          for (const k of [...words.keys()]) if (k !== r) words.delete(k);
          setGain(1);
          playing = true;
          rid = r;
          ctx.setState('playing', true);
          ctx.emit('playback', { type: 'playback.start', rid, at });
        }
        ctx.setState('buffering', false);
        const s = segs.get(r)?.find((x) => x.seg === (msg.seg as Json));
        if (s) {
          s.start ??= at;
          s.end = endAt;
        }
        if (paused === null) armEndTimer();
      },
      stop() {
        endTimer?.clear();
        io?.close();
      },
    };
  },
};
