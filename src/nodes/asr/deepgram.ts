import type { JsonObject } from '../../core/types';
import { int16BytesFromF32 } from '../audio';
import type { AsrAdapter, AsrEvent } from './types';

interface DgResults {
  is_final?: boolean;
  speech_final?: boolean;
  start: number;
  duration: number;
  channel: { alternatives: { transcript: string; confidence?: number; words?: { word: string; start: number; end: number }[] }[] };
}

/** Deepgram live transcription (/v1/listen WebSocket). `$open` is synthesized by the driver on connect. */
export const deepgramAdapter: AsrAdapter = {
  provider: 'deepgram',
  channel: 'deepgram',
  params: [
    { key: 'model', label: 'Model', type: 'string', default: 'nova-3' },
    { key: 'language', label: 'Language', type: 'string', default: 'en' },
    { key: 'endpointingMs', label: 'Endpointing (ms)', type: 'number', default: 300 },
    { key: 'utteranceEndMs', label: 'Utterance end (ms)', type: 'number', default: 1000 },
    { key: 'smartFormat', label: 'Smart format', type: 'boolean', default: true },
  ],
  buildRequest(p, sampleRate) {
    return {
      path: '/v1/listen',
      query: {
        model: p.model,
        language: p.language,
        encoding: 'linear16',
        sample_rate: sampleRate,
        channels: 1,
        interim_results: true,
        vad_events: true,
        endpointing: p.endpointingMs,
        utterance_end_ms: p.utteranceEndMs,
        smart_format: p.smartFormat,
      },
    };
  },
  encode: int16BytesFromF32,
  closeMessage: () => ({ type: 'CloseStream' }),
  parse(data): AsrEvent[] {
    const m = data as JsonObject;
    switch (m.type) {
      case '$open':
        return [{ type: 'open' }];
      case 'Metadata':
        return [{ type: 'meta', key: 'requestId', value: (m.request_id as string) ?? null }];
      case 'SpeechStarted':
        return [{ type: 'speech_started', at: m.timestamp as number }];
      case 'UtteranceEnd':
        return [{ type: 'utterance_end', at: m.last_word_end as number }];
      case 'Results': {
        const r = m as unknown as DgResults;
        const alt = r.channel.alternatives[0];
        return [
          {
            type: r.is_final ? 'final' : 'partial',
            text: alt?.transcript ?? '',
            confidence: alt?.confidence ?? null,
            speechFinal: !!r.speech_final,
            start: r.start,
            end: r.start + r.duration,
            words: alt?.words ?? [],
          },
        ];
      }
      default:
        return [{ type: 'unknown' }];
    }
  },
};
