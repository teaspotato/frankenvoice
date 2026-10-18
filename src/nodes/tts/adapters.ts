import type { TtsAdapter } from './types';

/** Cartesia POST /tts/bytes, raw float32 PCM. */
export const cartesiaAdapter: TtsAdapter = {
  provider: 'cartesia',
  channel: 'cartesia',
  params: [
    { key: 'model', label: 'Model', type: 'string', default: 'sonic-2' },
    { key: 'voiceId', label: 'Voice ID', type: 'string', default: 'a0e99841-438c-4a64-b679-ae501e7d6091' },
    { key: 'language', label: 'Language', type: 'string', default: 'en' },
    { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000 },
  ],
  buildRequest(text, p) {
    return {
      path: '/tts/bytes',
      body: {
        model_id: p.model,
        transcript: text,
        voice: { mode: 'id', id: p.voiceId },
        output_format: { container: 'raw', encoding: 'pcm_f32le', sample_rate: p.sampleRate },
        language: p.language,
      },
    };
  },
  format: (p) => ({ encoding: 'f32le', sampleRate: p.sampleRate as number }),
};

/** OpenAI-compatible POST /audio/speech with response_format "pcm" (16-bit, 24 kHz, mono). */
export const openaiTtsAdapter: TtsAdapter = {
  provider: 'openai-compatible',
  channel: 'openai-tts',
  params: [
    {
      key: 'endpoint',
      label: 'Endpoint',
      type: 'select',
      options: ['openai', 'custom'],
      default: 'openai',
      description: 'Named endpoint configured in the proxy (.env).',
    },
    { key: 'model', label: 'Model', type: 'string', default: 'gpt-4o-mini-tts' },
    { key: 'voice', label: 'Voice', type: 'string', default: 'alloy' },
    { key: 'instructions', label: 'Instructions', type: 'text', default: '' },
  ],
  buildRequest(text, p) {
    return {
      endpoint: p.endpoint,
      path: '/audio/speech',
      body: {
        model: p.model,
        input: text,
        voice: p.voice,
        response_format: 'pcm',
        ...(p.instructions ? { instructions: p.instructions } : {}),
      },
    };
  },
  format: () => ({ encoding: 's16le', sampleRate: 24000 }),
};
