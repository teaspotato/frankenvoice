import type { NodeDef } from '../../core/node';
import type { Json, JsonObject } from '../../core/types';
import { isAudioMsg } from '../../core/types';
import { deepgramAdapter } from './deepgram';
import type { AsrAdapter } from './types';
import { paramsFor, providerParams } from '../providers';

export const asrAdapters: Record<string, AsrAdapter> = {
  [deepgramAdapter.provider]: deepgramAdapter,
};

/**
 * Streaming speech-to-text. The provider is a parameter; every provider's
 * messages are normalized into asr.partial / asr.final / asr.speech_started /
 * asr.utterance_end with sample positions on the input audio track. Raw
 * provider messages are what gets recorded, so replay reproduces the stream.
 */
export const asrNode: NodeDef = {
  type: 'asr',
  title: 'ASR',
  category: 'asr',
  description: 'Streaming speech-to-text with interim results and endpointing. Providers: Deepgram.',
  inputs: [{ name: 'audio', type: 'audio' }],
  outputs: [{ name: 'transcript', type: 'transcript' }],
  params: [
    { key: 'provider', label: 'Provider', type: 'select', options: Object.keys(asrAdapters), default: 'deepgram' },
    ...providerParams(Object.values(asrAdapters)),
  ],
  channels: Object.values(asrAdapters).map((a) => ({ name: a.channel, description: `${a.provider} streaming connection` })),
  requires: (p) => [String(p.provider)],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const adapter = asrAdapters[p.provider as string];
    let io: ReturnType<typeof ctx.io.open> | undefined;
    let base = 0; // first sample sent; providers report seconds from here
    let sr = 16000;
    let track = '';
    const toSample = (sec: number) => base + Math.round(sec * sr);

    if (!adapter) ctx.log('error', `unknown ASR provider "${String(p.provider)}"`);
    ctx.setState('provider', p.provider);

    return {
      onInput(_port, msg) {
        if (!isAudioMsg(msg) || !adapter) return;
        if (!io) {
          base = msg.start;
          sr = msg.sampleRate;
          track = msg.track;
          ctx.setState('status', 'connecting');
          io = ctx.io.open(adapter.channel, adapter.buildRequest(paramsFor(adapter.provider, p), sr), {
            onRecv(data) {
              for (const ev of adapter.parse(data)) {
                switch (ev.type) {
                  case 'open':
                    ctx.setState('status', 'streaming');
                    break;
                  case 'meta':
                    ctx.setState(ev.key, ev.value);
                    break;
                  case 'speech_started':
                  case 'utterance_end':
                    ctx.emit('transcript', { type: `asr.${ev.type}`, track, at: toSample(ev.at) });
                    break;
                  case 'partial':
                  case 'final':
                    ctx.emit('transcript', {
                      type: `asr.${ev.type}`,
                      text: ev.text,
                      confidence: ev.confidence,
                      speechFinal: ev.speechFinal,
                      track,
                      start: toSample(ev.start),
                      end: toSample(ev.end),
                      words: ev.words.map((w) => ({ word: w.word, start: toSample(w.start), end: toSample(w.end) })),
                    });
                    break;
                  case 'unknown':
                    ctx.log('debug', 'unhandled ASR message', data);
                }
              }
            },
            onClose(data: Json) {
              ctx.setState('status', 'closed');
              if (data) ctx.log('warn', 'ASR connection closed', data);
            },
          });
        }
        if (msg.track !== track) return;
        io.send(null, { bin: adapter.encode(msg.samples), media: { track, start: msg.start, end: msg.end }, logBin: false });
      },
      stop() {
        if (io && !io.closed) {
          const bye = adapter?.closeMessage?.();
          if (bye !== undefined) io.send(bye);
          io.close();
        }
      },
    };
  },
};
