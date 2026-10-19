import type { NodeDef, TimerHandle } from '../core/node';
import type { Json } from '../core/types';
import { isAudioMsg } from '../core/types';
import { bytesFromF32 } from './audio';

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
 * speech over it: the audio stops at once and the rest of the reply is held,
 * and `resume` plays it from where it stopped, with the schedule shifted by the
 * pause (playback.paused / playback.resumed; the held sentences are announced again).
 */
export const speakerNode: NodeDef<Record<string, never>> = {
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
  params: [],
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

    const pause = () => {
      const at = ctx.now();
      io?.send({ cmd: 'flush' });
      endTimer?.clear();
      endTimer = undefined;
      // Keep what hasn't played; the chunk playing now is split where it stopped.
      queued = queued.flatMap((c) => {
        const dur = (c.samples.length / c.sampleRate) * 1000;
        if (c.at + dur <= at) return [];
        if (c.at >= at) return [c];
        const cut = Math.round(((at - c.at) / 1000) * c.sampleRate);
        return [{ ...c, at, start: c.start + cut, samples: c.samples.subarray(cut) }];
      });
      paused = at;
      endAt = Math.max(endAt, at);
      ctx.setState('paused', true);
      ctx.emit('played', { type: 'played.flush', at });
      ctx.emit('playback', { type: 'playback.paused', rid, at });
    };
    const resume = () => {
      const from = paused!;
      const at = ctx.now();
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
            const at = ctx.now();
            // Paused, the user heard up to the pause.
            const heard = heardText(segs.get(rid) ?? [], clock());
            if (paused === null) io?.send({ cmd: 'flush' });
            endTimer?.clear();
            endTimer = undefined;
            playing = false;
            endAt = at;
            queued = [];
            const wasPaused = paused !== null;
            if (wasPaused) {
              paused = null;
              ctx.setState('paused', false);
            }
            ctx.setState('playing', false);
            ctx.emit('playback', { type: 'playback.interrupted', rid, at, heard });
            // Paused, the device was flushed (and the echo reference cut) at the pause.
            if (!wasPaused) ctx.emit('played', { type: 'played.flush', at });
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
        // Forget what has played (while paused, nothing after the pause has).
        queued = queued.filter((q) => q.at + (q.samples.length / q.sampleRate) * 1000 > clock());
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
