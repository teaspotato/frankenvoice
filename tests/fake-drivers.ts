// Stand-ins for the browser drivers: same message shapes, random timing and
// random chunking, so a live run is genuinely nondeterministic.

import type { DriverRegistry, LiveDriver } from '../src/core/world';
import type { Json } from '../src/core/types';
import { defaultConfig, type PipelineConfig } from '../src/pipeline';

const jitter = (base: number, spread: number) => base + Math.random() * spread;
const later = (ms: number, fn: () => void) => setTimeout(fn, ms);

/**
 * Delivers messages one at a time, in order, each a gap after the last, like a
 * stream. (Separate timers can fire out of order when their times are close.)
 */
function sequence(deliver: (m: Json) => void, isClosed: () => boolean) {
  const queue: { m: Json | (() => void); gap: number }[] = [];
  let pumping = false;
  const pump = () => {
    const next = queue.shift();
    if (!next) return void (pumping = false);
    pumping = true;
    later(next.gap, () => {
      if (!isClosed()) typeof next.m === 'function' ? next.m() : deliver(next.m);
      pump();
    });
  };
  return (m: Json | (() => void), gap = jitter(0.5, 2)) => {
    queue.push({ m, gap });
    if (!pumping) pump();
  };
}

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

/** FireRedVAD stand-in: one probability per completed 10 ms frame (400-sample windows, 160 hop), from RMS. */
const firered: LiveDriver = (_req, sink) => {
  let chain = Promise.resolve();
  let buf: number[] = [];
  return {
    send(_d, bin) {
      buf.push(...new Float32Array(bin!.slice().buffer));
      const p: number[] = [];
      while (buf.length >= 400) {
        let sum = 0;
        for (let i = 0; i < 400; i++) sum += buf[i] * buf[i];
        p.push(Math.round(Math.min(1, Math.sqrt(sum / 400) * 5) * 1000) / 1000);
        buf = buf.slice(160);
      }
      chain = chain.then(() => new Promise((r) => later(jitter(0, 2), () => (sink.recv({ p }), r()))));
    },
    close() {},
  };
};

const utterances = ['hello there', 'what time is it', 'tell me a joke'];
const words = (text: string, start: number) => text.split(' ').filter(Boolean).map((w, i) => ({ word: w, start: start + i * 0.2, end: start + i * 0.2 + 0.15 }));
const results = (text: string, start: number, isFinal: boolean, fromFinalize = false): Json => ({
  type: 'Results',
  is_final: isFinal,
  speech_final: isFinal,
  from_finalize: fromFinalize,
  start,
  end: start + 0.8,
  text,
  words: words(text, start),
  decode_ms: 20,
});

/** Qwen3-ASR on the GPU box: every ~40 frames an "utterance", as partials then a final (sooner on Finalize). */
const gpuboxAsr: LiveDriver = (_req, sink) => {
  let sent = 0;
  let utt = 0;
  let closed = false;
  /** The utterance in progress, until its final is sent. */
  let pending: { text: string; start: number } | undefined;
  const out = sequence((m) => sink.recv(m), () => closed);
  out({ type: '$open' }, jitter(2, 3));
  out({ type: 'Ready', model: 'fake-qwen3-asr' });
  const final = (fromFinalize: boolean) => {
    const u = pending;
    pending = undefined;
    if (u) out(results(u.text, u.start, true, fromFinalize));
    else if (fromFinalize) out(results('', sent / 16000, true, true));
  };
  return {
    send(data, bin) {
      if (closed) return;
      const m = data as { type?: string } | null;
      if (m?.type === 'Finalize') return final(true);
      if (m?.type === 'CloseStream') return final(false);
      if (!bin) return;
      sent += bin.byteLength / 2;
      if (sent >= (utt + 1) * 40 * 512 && utt < utterances.length) {
        const text = utterances[utt];
        const start = (utt * 40 * 512) / 16000;
        utt++;
        pending = { text, start };
        out({ type: 'SpeechStarted', at: start });
        const ws = text.split(' ');
        ws.forEach((_, i) => out(() => pending?.text === text && sink.recv(results(ws.slice(0, i + 1).join(' '), start, false)), jitter(1, 2)));
        out(() => pending?.text === text && final(false), jitter(6, 4));
      }
    },
    close: () => void (closed = true),
  };
};

/** TypeSafe Choice: picks the first option that isn't "-", about a millisecond later. */
const typesafe: LiveDriver = (req, sink) => {
  const criteria = (req as { body: { questions: { pick: { criteria: Record<string, string> } } } }).body.questions.pick.criteria;
  const choice = Object.keys(criteria).find((k) => k !== '-') ?? '-';
  let closed = false;
  later(jitter(0.5, 1), () => {
    if (closed) return;
    sink.recv({ answers: { pick: { type: 'choice', choice, confidence: 0.95, probabilities: { [choice]: 0.95 } } } });
    sink.close();
  });
  return { send() {}, close: () => void (closed = true) };
};

/** A chat model: a fixed reply, cut at max_tokens; a last assistant message is continued (llama.cpp's prefill). */
const openai: LiveDriver = (req, sink) => {
  const r = req as { body: { messages: { role: string; content: string }[]; max_tokens?: number } };
  const msgs = r.body.messages;
  const prefill = msgs.at(-1)?.role === 'assistant' ? msgs.at(-1)!.content : '';
  const users = msgs.filter((m) => m.role !== 'assistant').length;
  // Backchannels run on slot 1; a reply with a grammar is tagged, one utterance per line.
  const b = r.body as { grammar?: string; id_slot?: number };
  const bc = b.id_slot === 1;
  let tokens: string[] = bc
    ? ['Oh,', ' okay.']
    : b.grammar
      ? ['[i', '1g0] ', 'Reply ', 'number ', `${users}.`, '\n[i', '0g1] ', 'This ', 'is ', 'a ', 'second ', 'sentence!', '\n[i', '1g1] ', 'And ', 'a ', 'third?']
      : `Reply number ${users}. This is a second sentence! And a third?`.match(/\S+\s*/g)!;
  let skip = 0;
  while (prefill && !bc && skip < tokens.length && tokens.slice(0, skip).join('').trimEnd() !== prefill) skip++;
  tokens = tokens.slice(skip);
  const cut = r.body.max_tokens !== undefined && tokens.length > r.body.max_tokens;
  if (cut) tokens = tokens.slice(0, r.body.max_tokens);
  let closed = false;
  let t = jitter(2, 3);
  const ev = (e: Json) => {
    t += jitter(0.5, 2);
    later(t, () => !closed && sink.recv(e));
  };
  for (const tok of tokens) ev({ id: 'c1', choices: [{ index: 0, delta: { content: tok }, finish_reason: null }] });
  ev({ id: 'c1', choices: [{ index: 0, delta: {}, finish_reason: cut ? 'length' : 'stop' }] });
  ev({ $done: true });
  later(t + 1, () => !closed && sink.close());
  return { send() {}, close: () => void (closed = true) };
};

/**
 * llama.cpp /completion for the tape: reads the open line and the grammar, and answers like a
 * plain agent: listens while words come in, responds in the first cell without new words,
 * continues twice, then finishes. Capped at the grammar's word limit.
 */
const llamaCompletion: LiveDriver = (req, sink) => {
  const { prompt, grammar } = (req as { body: { prompt: string; grammar: string } }).body;
  const lines = prompt.slice(prompt.lastIndexOf('<channel|>') + '<channel|>'.length).split('\n');
  const open = lines[lines.length - 1];
  const max = Number(/\{0,(\d+)\}/.exec(grammar)?.[1] ?? 0) + 1;
  const cap = (s: string) => `"${s.split(' ').slice(0, max).join(' ')}"`;
  const lastAnswer = lines.findLastIndex((l) => /: (respond|resume) /.test(l));
  const heard = lines.slice(lastAnswer + 1, -1).some((l) => /user: "/.test(l));
  // No new words (the fake mic's noise keeps the VAD busy, so voice doesn't count).
  const quiet = !/user: "/.test(open);
  let content = ' listen';
  if (grammar.includes('" continue "')) {
    const since = lines.slice(lastAnswer + 1, -1).filter((l) => / continue "/.test(l)).length;
    content = since >= 2 ? ` finish ${cap('That is all.')}` : ` continue ${cap('and some more words')}`;
  } else if (heard && quiet) content = ` respond ${cap('Sure, here is an answer')}`;
  let closed = false;
  later(jitter(3, 4), () => {
    if (closed) return;
    sink.recv({ content, stop_type: 'word', timings: { prompt_n: 12, prompt_ms: 4.2, predicted_n: 6, predicted_ms: 30.1 } });
    sink.close(null);
  });
  return { send() {}, close: () => void (closed = true) };
};

/** Kyutai on the GPU box: audio, word times and Flushed for each flushed segment; Done and close after Close. */
const gpuboxTts: LiveDriver = (_req, sink) => {
  let closed = false;
  let text = '';
  /** Seconds of audio sent on this socket. */
  let at = 0;
  /** Words the model has read. */
  let read = 0;
  const out = sequence((m) => sink.recv(m), () => closed);
  out({ type: '$open' }, jitter(2, 3));
  out({ type: 'Ready', sample_rate: 24000 });
  const speak = () => {
    const ws = text.trim().split(/\s+/).filter(Boolean);
    text = '';
    if (!ws.length) return;
    const n = Math.round(ws.join(' ').length * 0.004 * 24000);
    const pcm = new Int16Array(n).map((_, i) => Math.round(Math.sin(i / 10) * 3000));
    const bytes = new Uint8Array(pcm.buffer);
    const per = n / 24000 / ws.length;
    read += ws.length;
    out({ type: 'Consumed', words: read, queued: 0 }, jitter(0.5, 1));
    out({ type: 'Words', words: ws.map((word, i) => ({ word, start: at + i * per, end: at + (i + 1) * per })) }, jitter(1, 2));
    at += n / 24000;
    // Deliberately split mid-sample.
    for (let off = 0; off < bytes.length; ) {
      const len = Math.min(bytes.length - off, 1 + Math.floor(Math.random() * 3000));
      const chunk = bytes.slice(off, (off += len));
      out(() => sink.recv(null, chunk), jitter(0.2, 1));
    }
  };
  return {
    send(data) {
      const m = data as { type: string; text?: string };
      if (closed) return;
      if (m.type === 'Speak') text += m.text ?? '';
      else if (m.type === 'Flush') {
        speak();
        out({ type: 'Flushed' });
      } else if (m.type === 'Close') {
        speak();
        out({ type: 'Done', stats: {} });
        out(() => sink.close({ code: 1000, reason: '' }), 0.5);
      }
    },
    close: () => void (closed = true),
  };
};

/** The server's canned-line cache: a tone for the line, from the cache or a background render. */
const ttsCanned: LiveDriver = (req, sink) => {
  const { text } = req as { text: string };
  let closed = false;
  const n = Math.round(text.length * 0.004 * 24000);
  const pcm = new Int16Array(n).map((_, i) => Math.round(Math.sin(i / 7) * 3000));
  const cached = Math.random() < 0.5;
  later(jitter(cached ? 0.5 : 5, 3), () => {
    if (closed) return;
    sink.recv({ type: 'Canned', cached, key: 'k', sampleRate: 24000 });
    sink.recv(null, new Uint8Array(pcm.buffer));
    sink.close(null);
  });
  return { send() {}, close: () => void (closed = true) };
};

/** RNNoise stand-in: halves the signal, one reply per frame, after a random delay. */
const rnnoise: LiveDriver = (_req, sink) => {
  let closed = false;
  let t = 0;
  return {
    send(_data, bin) {
      if (!bin) return;
      const f = new Float32Array(bin.slice().buffer).map((v) => v * 0.5);
      t = Math.max(t, performance.now()) + jitter(0.2, 1);
      later(t - performance.now(), () => !closed && sink.recv({ vad: Math.random() }, new Uint8Array(f.buffer)));
    },
    close: () => void (closed = true),
  };
};

/**
 * Turn model stand-in: ready after a moment, then K = 2 frames per two blocks
 * (plus frame 0 first), scores from each channel's energy, after a random delay.
 * End of turn fires when the user goes quiet; take-the-floor stays below threshold.
 */
const turnModel: LiveDriver = (req, sink) => {
  const { frames } = req as { frames: number };
  let closed = false;
  let blocks = 0;
  let t = 1;
  let pending: { u: number; a: number }[] = [];
  later(jitter(2, 3), () => !closed && sink.recv({ type: 'ready', ep: 'fake', frames }));
  const rms = (x: Float32Array) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);
  return {
    send(_data, bin) {
      if (!bin || closed) return;
      const x = new Float32Array(bin.slice().buffer);
      pending.push({ u: rms(x.subarray(0, x.length / 2)), a: rms(x.subarray(x.length / 2)) });
      const outs: Json[] = blocks++ === 0 ? [{ t: 0, eot: [0, 0], int: [0, 0], silent: [1, 1], post: [] }] : [];
      if (pending.length < frames) {
        if (outs.length) later(jitter(0, 2), () => !closed && sink.recv({ type: 'frames', outs, ms: 1, backlog: 0 }));
        return;
      }
      for (const { u, a } of pending) {
        const talk = Math.min(1, u * 5);
        outs.push({ t: t++, eot: [Math.round((1 - talk) * 1000) / 1000, 0.1], int: [Math.round(talk * 150) / 1000, Math.min(1, a * 5)], silent: [1 - talk, 1], post: [] });
      }
      pending = [];
      later(jitter(0, 2), () => !closed && sink.recv({ type: 'frames', outs, ms: 1, backlog: 0 }));
    },
    close: () => void (closed = true),
  };
};

const speaker: LiveDriver = (_req, sink) => {
  later(1, () => sink.recv({ type: 'ready', outputLatency: 0.02 }));
  return { send() {}, close() {} };
};

export const fakeDrivers: DriverRegistry = { mic, silero, firered, 'gpubox-asr': gpuboxAsr, 'gpubox-tts': gpuboxTts, 'tts-canned': ttsCanned, openai, 'llama-completion': llamaCompletion, rnnoise, speaker, 'turn-model': turnModel, typesafe };

/** The default pipeline with the turn model on (it is off by default, a big download). */
export function testConfig(): PipelineConfig {
  const c = defaultConfig();
  c.turnmodel.variant = 'ssl_turn';
  // The fake user talks soon after replies start; these tests are about the rest of the floor rules.
  c.turn.params['full-duplex'] = { yieldWithinMs: 0 };
  // Small chunks, so replies are paced by the scheduler.
  c.sched.params.tick = { firstTokens: 3, chunkTokens: 2 };
  return c;
}
