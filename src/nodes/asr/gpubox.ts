import type { JsonObject } from '../../core/types';
import { int16BytesFromF32 } from '../audio';
import type { AsrAdapter, AsrEvent } from './types';

type Word = { word: string; start: number; end: number };

/**
 * Qwen3-ASR on the GPU box (gpubox/server.py, protocol in gpubox/PROTOCOL.md). Partials are
 * re-decodes of the segment since the last final; finals come on Finalize, with word times from
 * a forced aligner. `$open` is synthesized by the driver on connect.
 */
export const gpuboxAsrAdapter: AsrAdapter = {
  channel: 'gpubox-asr',
  params: [
    { key: 'language', label: 'Language', type: 'string', default: 'English', description: 'Qwen3-ASR language name (English, Chinese, Japanese, …).' },
    {
      key: 'context',
      label: 'Context',
      type: 'text',
      default: '',
      description: 'Hotwords or the topic, given to the model as context to bias recognition.',
    },
    { key: 'partialMs', label: 'Partial every (ms)', type: 'number', default: 250, description: 'Re-decode the segment after this much new audio.' },
  ],
  buildRequest(p, sampleRate) {
    return {
      path: '/v1/asr',
      query: { language: p.language, sample_rate: sampleRate, partial_ms: p.partialMs, ...(p.context ? { context: p.context } : {}) },
    };
  },
  encode: int16BytesFromF32,
  closeMessage: () => ({ type: 'CloseStream' }),
  finalizeMessage: () => ({ type: 'Finalize' }),
  parse(data): AsrEvent[] {
    const m = data as JsonObject;
    switch (m.type) {
      case '$open':
        return [{ type: 'open' }];
      case 'Ready':
        return [{ type: 'meta', key: 'model', value: (m.model as string) ?? null }];
      case 'SpeechStarted':
        return [{ type: 'speech_started', at: m.at as number }];
      case 'UtteranceEnd':
        return [{ type: 'utterance_end', at: m.at as number }];
      case 'Results':
        return [
          {
            type: m.is_final ? 'final' : 'partial',
            text: (m.text as string) ?? '',
            confidence: null,
            speechFinal: !!m.speech_final,
            ...(m.from_finalize ? { fromFinalize: true } : {}),
            start: m.start as number,
            end: m.end as number,
            words: (m.words as Word[] | undefined) ?? [],
          },
        ];
      case 'Error':
        return [{ type: 'meta', key: 'error', value: (m.message as string) ?? null }];
      default:
        return [{ type: 'unknown' }];
    }
  },
};
