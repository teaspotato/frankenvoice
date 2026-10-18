// Stand-ins for the browser drivers: same message shapes, random timing and
// random chunking, so a live run is genuinely nondeterministic.

import type { DriverRegistry, LiveDriver } from '../src/core/world';
import type { Json } from '../src/core/types';

const jitter = (base: number, spread: number) => base + Math.random() * spread;
const later = (ms: number, fn: () => void) => setTimeout(fn, ms);

/** 16 kHz mic: silence, then a loud noise burst ("speech"), then silence; repeated. */
const mic: LiveDriver = (req, sink) => {
  const { frameSamples } = req as { frameSamples: number };
  let frame = 0;
  let stopped = false;
  later(1, () => sink.recv({ type: 'ready', label: 'fake mic' }));
  const tick = () => {
    if (stopped) return;
    const s = new Float32Array(frameSamples);
    const inSpeech = frame % 40 >= 8 && frame % 40 < 24;
    for (let i = 0; i < s.length; i++) s[i] = inSpeech ? (Math.random() - 0.5) * 0.6 : (Math.random() - 0.5) * 0.001;
    sink.recv({ frame: frame++ }, new Uint8Array(s.buffer));
    later(jitter(3, 3), tick);
  };
  later(2, tick);
  return { send() {}, close: () => void (stopped = true) };
};

const silero: LiveDriver = (_req, sink) => {
  let chain = Promise.resolve();
  return {
    send(_d, bin) {
      const f = new Float32Array(bin!.slice().buffer);
      let sum = 0;
      for (const x of f) sum += x * x;
      const p = Math.min(1, Math.sqrt(sum / f.length) * 5);
      chain = chain.then(() => new Promise((r) => later(jitter(0, 2), () => (sink.recv({ p: Math.round(p * 1000) / 1000 }), r()))));
    },
    close() {},
  };
};

const utterances = ['hello there', 'what time is it', 'tell me a joke'];
const deepgram: LiveDriver = (_req, sink) => {
  let sent = 0;
  let utt = 0;
  let closed = false;
  later(jitter(2, 3), () => sink.recv({ type: '$open' }));
  later(jitter(3, 3), () => sink.recv({ type: 'Metadata', request_id: 'fake' }));
  return {
    send(data, bin) {
      if (!bin || closed) return;
      sent += bin.byteLength / 2;
      // Every ~40 frames, transcribe one "utterance": two partials, then a final.
      if (sent >= (utt + 1) * 40 * 512 && utt < utterances.length) {
        const text = utterances[utt];
        const start = (utt * 40 * 512) / 16000;
        utt++;
        const words = text.split(' ');
        words.forEach((_, i) => {
          later(jitter(1 + i * 2, 2), () => {
            if (closed) return;
            sink.recv(results(words.slice(0, i + 1).join(' '), start, false, false));
          });
        });
        later(jitter(10, 4), () => !closed && sink.recv(results(text, start, true, true)));
      }
      void data;
    },
    close: () => void (closed = true),
  };
};

function results(text: string, start: number, isFinal: boolean, speechFinal: boolean): Json {
  return {
    type: 'Results',
    is_final: isFinal,
    speech_final: speechFinal,
    start,
    duration: 0.8,
    channel: { alternatives: [{ transcript: text, confidence: 0.9, words: text.split(' ').map((w, i) => ({ word: w, start: start + i * 0.2, end: start + i * 0.2 + 0.15 })) }] },
  };
}

const anthropic: LiveDriver = (req, sink) => {
  const r = req as { messages: { role: string }[] };
  const reply = `Reply number ${r.messages.length}. This is a second sentence! And a third?`;
  const tokens = reply.match(/\S+\s*/g)!;
  let closed = false;
  let t = jitter(2, 3);
  const ev = (e: Json) => {
    t += jitter(0.5, 2);
    later(t, () => !closed && sink.recv(e));
  };
  ev({ type: 'message_start', message: { id: 'msg_fake', role: 'assistant', content: [] } });
  ev({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const tok of tokens) ev({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: tok } });
  ev({ type: 'content_block_stop', index: 0 });
  ev({ type: 'message_delta', delta: { stop_reason: 'end_turn' } });
  ev({ type: 'message_stop' });
  later(t + 1, () => !closed && sink.close());
  return { send() {}, close: () => void (closed = true) };
};

const cartesia: LiveDriver = (req, sink) => {
  const { body } = req as { body: { transcript: string; output_format: { sample_rate: number } } };
  const n = Math.round(body.transcript.length * 0.004 * body.output_format.sample_rate);
  const bytes = new Uint8Array(new Float32Array(n).map((_, i) => Math.sin(i / 10) * 0.1).buffer);
  let closed = false;
  let off = 0;
  let t = jitter(2, 4);
  while (off < bytes.length) {
    const len = Math.min(bytes.length - off, 1 + Math.floor(Math.random() * 3000)); // deliberately unaligned
    const chunk = bytes.slice(off, off + len);
    off += len;
    t += jitter(0.2, 1);
    later(t, () => !closed && sink.recv(null, chunk));
  }
  later(t + 0.5, () => !closed && sink.close());
  return { send() {}, close: () => void (closed = true) };
};

const speaker: LiveDriver = (_req, sink) => {
  later(1, () => sink.recv({ type: 'ready', outputLatency: 0.02 }));
  return { send() {}, close() {} };
};

export const fakeDrivers: DriverRegistry = { mic, silero, deepgram, anthropic, cartesia, speaker };
