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
}

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
    const frac = s.end !== undefined && s.end > s.start ? (at - s.start) / (s.end - s.start) : 0;
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

    const end = () => {
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
          if (msg.type === 'cancel' && playing) {
            io?.send({ cmd: 'flush' });
            const at = ctx.now();
            endTimer?.clear();
            endTimer = undefined;
            playing = false;
            endAt = at;
            ctx.setState('playing', false);
            ctx.emit('playback', { type: 'playback.interrupted', rid, at, heard: heardText(segs.get(rid) ?? [], at) });
            ctx.emit('played', { type: 'played.flush', at });
          }
          return;
        }
        if (msg.type === 'tts.segment_start') {
          const r = (msg.rid as string) ?? null;
          if (!segs.has(r)) segs.set(r, []);
          segs.get(r)!.push({ seg: msg.seg as number, text: msg.text as string });
          ctx.emit('playback', { type: 'playback.segment', rid: msg.rid as string, seg: msg.seg as number, text: msg.text as string, at: Math.max(ctx.now(), endAt) });
          return;
        }
        if (msg.type === 'tts.response_end' || msg.type === 'tts.cancelled') {
          const r = (msg.rid as string) ?? null;
          complete.add(r);
          if (playing && r === rid && ctx.now() >= endAt) end();
          return;
        }
        if (!isAudioMsg(msg)) return;
        const now = ctx.now();
        const at = Math.max(now, endAt);
        const dur = (msg.samples.length / msg.sampleRate) * 1000;
        endAt = at + dur;
        io?.send({ cmd: 'play', at, sampleRate: msg.sampleRate }, {
          bin: bytesFromF32(msg.samples),
          media: { track: msg.track, start: msg.start, end: msg.end },
          logBin: false,
        });
        ctx.emit('played', { type: 'audio', track: msg.track, start: msg.start, end: msg.end, sampleRate: msg.sampleRate, samples: msg.samples, at });
        const r = (msg.rid as string) ?? null;
        if (playing && r !== rid) end();
        if (!playing) {
          // Forget older replies.
          for (const k of [...segs.keys()]) if (k !== r) segs.delete(k);
          for (const k of [...complete]) if (k !== r) complete.delete(k);
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
        armEndTimer();
      },
      stop() {
        endTimer?.clear();
        io?.close();
      },
    };
  },
};
