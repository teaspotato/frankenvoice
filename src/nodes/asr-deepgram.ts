import type { NodeDef } from '../core/node';
import type { Json } from '../core/types';
import { isAudioMsg } from '../core/types';
import { int16BytesFromF32 } from './audio';

interface DeepgramParams {
  model: string;
  language: string;
  endpointingMs: number;
  utteranceEndMs: number;
  smartFormat: boolean;
}

interface DgResults {
  type: 'Results';
  is_final?: boolean;
  speech_final?: boolean;
  start: number;
  duration: number;
  channel: { alternatives: { transcript: string; confidence?: number; words?: { word: string; start: number; end: number }[] }[] };
}

/**
 * Streaming ASR over Deepgram's live WebSocket API (through the local proxy).
 * Every server message is an io.recv event, so interim results replay at the
 * exact moments they first arrived.
 */
export const deepgramAsrNode: NodeDef<DeepgramParams> = {
  type: 'asr.deepgram',
  title: 'Deepgram ASR',
  category: 'asr',
  description: 'Streaming speech-to-text with interim results and endpointing.',
  inputs: [{ name: 'audio', type: 'audio' }],
  outputs: [{ name: 'transcript', type: 'transcript' }],
  params: [
    { key: 'model', label: 'Model', type: 'string', default: 'nova-3' },
    { key: 'language', label: 'Language', type: 'string', default: 'en' },
    { key: 'endpointingMs', label: 'Endpointing (ms)', type: 'number', default: 300 },
    { key: 'utteranceEndMs', label: 'Utterance end (ms)', type: 'number', default: 1000 },
    { key: 'smartFormat', label: 'Smart format', type: 'boolean', default: true },
  ],
  channels: [{ name: 'deepgram', description: 'Deepgram /v1/listen WebSocket' }],
  create(ctx) {
    const p = ctx.params;
    let io: ReturnType<typeof ctx.io.open> | undefined;
    let base = 0; // first sample sent; Deepgram times are seconds from here
    let sr = 16000;
    let track = '';
    const toSample = (sec: number) => base + Math.round(sec * sr);

    return {
      onInput(_port, msg) {
        if (!isAudioMsg(msg)) return;
        if (!io) {
          base = msg.start;
          sr = msg.sampleRate;
          track = msg.track;
          const query: Record<string, Json> = {
            model: p.model,
            language: p.language,
            encoding: 'linear16',
            sample_rate: sr,
            channels: 1,
            interim_results: true,
            vad_events: true,
            endpointing: p.endpointingMs,
            utterance_end_ms: p.utteranceEndMs,
            smart_format: p.smartFormat,
          };
          ctx.setState('status', 'connecting');
          io = ctx.io.open('deepgram', { path: '/v1/listen', query }, {
            onRecv(data) {
              const m = data as { type?: string } & Record<string, unknown>;
              switch (m.type) {
                case '$open':
                  ctx.setState('status', 'streaming');
                  break;
                case 'Metadata':
                  ctx.setState('requestId', (m.request_id as string) ?? null);
                  break;
                case 'SpeechStarted':
                  ctx.emit('transcript', { type: 'asr.speech_started', track, at: toSample(m.timestamp as number) });
                  break;
                case 'UtteranceEnd':
                  ctx.emit('transcript', { type: 'asr.utterance_end', track, at: toSample(m.last_word_end as number) });
                  break;
                case 'Results': {
                  const r = m as unknown as DgResults;
                  const alt = r.channel.alternatives[0];
                  ctx.emit('transcript', {
                    type: r.is_final ? 'asr.final' : 'asr.partial',
                    text: alt?.transcript ?? '',
                    confidence: alt?.confidence ?? null,
                    speechFinal: !!r.speech_final,
                    track,
                    start: toSample(r.start),
                    end: toSample(r.start + r.duration),
                    words: (alt?.words ?? []).map((w) => ({ word: w.word, start: toSample(w.start), end: toSample(w.end) })),
                  });
                  break;
                }
                default:
                  ctx.log('debug', `deepgram message ${String(m.type)}`, data);
              }
            },
            onClose(data) {
              ctx.setState('status', 'closed');
              if (data) ctx.log('warn', 'deepgram closed', data);
            },
          });
        }
        if (msg.track !== track) return;
        io.send(null, { bin: int16BytesFromF32(msg.samples), media: { track, start: msg.start, end: msg.end }, logBin: false });
      },
      stop() {
        if (io && !io.closed) {
          io.send({ type: 'CloseStream' });
          io.close();
        }
      },
    };
  },
};
