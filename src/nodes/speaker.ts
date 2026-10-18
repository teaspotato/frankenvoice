import type { NodeDef, TimerHandle } from '../core/node';
import { isAudioMsg } from '../core/types';
import { bytesFromF32 } from './audio';

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

    // One pending timer at a time; if more audio was queued meanwhile it re-arms.
    const armEndTimer = () => {
      if (endTimer) return;
      endTimer = ctx.setTimer(Math.max(0, endAt - ctx.now()), () => {
        endTimer = undefined;
        if (!playing) return;
        if (ctx.now() < endAt) return armEndTimer();
        playing = false;
        ctx.setState('playing', false);
        ctx.emit('playback', { type: 'playback.end', rid });
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
            ctx.emit('playback', { type: 'playback.interrupted', rid, at });
            ctx.emit('played', { type: 'played.flush', at });
          }
          return;
        }
        if (msg.type === 'tts.segment_start') {
          ctx.emit('playback', { type: 'playback.segment', rid: msg.rid as string, seg: msg.seg as number, text: msg.text as string, at: Math.max(ctx.now(), endAt) });
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
        if (!playing || r !== rid) {
          playing = true;
          rid = r;
          ctx.setState('playing', true);
          ctx.emit('playback', { type: 'playback.start', rid, at });
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
