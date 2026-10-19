import type { NodeDef } from '../../core/node';
import type { AudioMsg, Json, JsonObject } from '../../core/types';
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
    let base = 0; // first sample sent on this connection; providers report seconds from here
    let sr = 16000;
    let track = '';
    const toSample = (sec: number) => base + Math.round(sec * sr);
    // Reconnecting after the connection drops mid-session: audio since the last final result is
    // kept so the utterance in progress is sent again, and repeated failures back off, then give up.
    const KEEP_MS = 5000;
    let recent: AudioMsg[] = [];
    let finalEnd = 0;
    let failures = 0;
    let retryAt = 0;
    let stopping = false;
    let heard = false;
    let opened = false;

    if (!adapter) ctx.log('error', `unknown ASR provider "${String(p.provider)}"`);
    ctx.setState('provider', p.provider);

    const send = (m: AudioMsg) => io!.send(null, { bin: adapter!.encode(m.samples), media: { track, start: m.start, end: m.end }, logBin: false });

    return {
      onInput(_port, msg) {
        if (!isAudioMsg(msg) || !adapter || stopping) return;
        if (!track) track = msg.track;
        if (msg.track !== track) return;
        recent.push(msg);
        while (recent.length && (recent[0].end <= finalEnd || msg.end - recent[0].start > (KEEP_MS * msg.sampleRate) / 1000)) recent.shift();
        if (!io) {
          if (failures >= 4 || ctx.now() < retryAt) return;
          const resend = opened ? recent : [msg];
          opened = true;
          base = resend[0].start;
          sr = msg.sampleRate;
          heard = false;
          ctx.setState('status', 'connecting');
          const conn = ctx.io.open(adapter.channel, adapter.buildRequest(paramsFor(adapter.provider, p), sr), {
            onRecv(data) {
              heard = true;
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
                    if (ev.type === 'final') finalEnd = toSample(ev.end);
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
              if (io === conn) io = undefined;
              ctx.setState('status', 'closed');
              if (stopping) return;
              // Dropped mid-session: reconnect on the next audio frame.
              failures = heard ? 1 : failures + 1;
              retryAt = ctx.now() + 500 * 2 ** (failures - 1);
              if (failures >= 4) ctx.log('error', 'ASR connection lost; gave up after repeated failures', data);
              else ctx.log('error', 'ASR connection lost; reconnecting', data);
            },
          });
          io = conn;
          for (const m of resend.slice(0, -1)) send(m);
        }
        send(msg);
      },
      stop() {
        stopping = true;
        if (io && !io.closed) {
          const bye = adapter?.closeMessage?.();
          if (bye !== undefined) io.send(bye);
          io.close();
        }
      },
    };
  },
};
