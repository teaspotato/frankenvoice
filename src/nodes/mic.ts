import type { NodeDef } from '../core/node';
import { f32FromBytes } from './audio';

interface MicParams {
  sampleRate: number;
  frameSamples: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

/**
 * Microphone source. The audio itself arrives over the 'mic' IO channel, so a
 * replay re-delivers exactly the frames that were captured, when they were captured.
 */
export const micNode: NodeDef<MicParams> = {
  type: 'mic',
  title: 'Microphone',
  category: 'source',
  description: 'Captures mono audio from the browser microphone, resampled to the target rate.',
  inputs: [],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 16000 },
    { key: 'frameSamples', label: 'Frame size (samples)', type: 'number', default: 512 },
    { key: 'echoCancellation', label: 'Echo cancellation', type: 'boolean', default: true },
    { key: 'noiseSuppression', label: 'Noise suppression', type: 'boolean', default: true },
    { key: 'autoGainControl', label: 'Auto gain', type: 'boolean', default: true },
  ],
  channels: [{ name: 'mic', description: 'Captured PCM frames (float32) plus capture timestamps' }],
  create(ctx) {
    const p = ctx.params;
    let io: ReturnType<typeof ctx.io.open> | undefined;
    return {
      start() {
        ctx.setState('status', 'opening');
        io = ctx.io.open(
          'mic',
          {
            sampleRate: p.sampleRate,
            frameSamples: p.frameSamples,
            echoCancellation: p.echoCancellation,
            noiseSuppression: p.noiseSuppression,
            autoGainControl: p.autoGainControl,
          },
          {
            onRecv(data, bin) {
              if (!bin) {
                ctx.setState('status', 'capturing');
                ctx.setState('device', data);
                return;
              }
              const samples = f32FromBytes(bin);
              const ref = ctx.media.append('audio', samples, p.sampleRate);
              ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples });
            },
            onClose(data) {
              ctx.setState('status', 'closed');
              if (data) ctx.log('warn', 'microphone closed', data);
            },
          },
        );
      },
      stop() {
        io?.close();
      },
    };
  },
};
