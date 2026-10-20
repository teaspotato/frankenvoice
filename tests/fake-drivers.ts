// Stand-ins for the browser drivers: same message shapes, random timing and
// random chunking, so a live run is genuinely nondeterministic.

import type { DriverRegistry, LiveDriver } from '../src/core/world';
import type { Json } from '../src/core/types';

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
  const r = req as { messages: { role: string; content: { type: string }[] }[]; tools?: { name: string }[] };
  const tools = new Set((r.tools ?? []).map((x) => x.name));
  const last = r.messages[r.messages.length - 1];
  // The conversation starts a worker on the first user turn.
  const delegate = tools.has('spawn_worker') && r.messages.length === 1 && last.content[0]?.type === 'text';
  const reply = delegate ? 'Let me look that up.' : `Reply number ${r.messages.length}. This is a second sentence! And a third?`;
  const tokens = reply.match(/\S+\s*/g)!;
  let closed = false;
  const ev = sequence((e) => sink.recv(e), () => closed);
  ev({ type: 'message_start', message: { id: 'msg_fake', role: 'assistant', content: [] } }, jitter(2.5, 4));
  let i = 0;
  ev({ type: 'content_block_start', index: i, content_block: { type: 'text', text: '' } });
  for (const tok of tokens) ev({ type: 'content_block_delta', index: i, delta: { type: 'text_delta', text: tok } });
  ev({ type: 'content_block_stop', index: i++ });
  if (delegate) {
    ev({ type: 'content_block_start', index: i, content_block: { type: 'tool_use', id: 'toolu_fake', name: 'spawn_worker', input: {} } });
    ev({ type: 'content_block_delta', index: i, delta: { type: 'input_json_delta', partial_json: '{"task":"the question"}' } });
    ev({ type: 'content_block_stop', index: i++ });
  }
  ev({ type: 'message_delta', delta: { stop_reason: delegate ? 'tool_use' : 'end_turn' } });
  ev({ type: 'message_stop' });
  ev(() => sink.close(), 1);
  return { send() {}, close: () => void (closed = true) };
};

/** An ACP agent: sets up a session, then answers each prompt after thinking and one web fetch (asking permission first). */
const acp: LiveDriver = (_req, sink) => {
  let closed = false;
  const out = sequence((m) => sink.recv(m), () => closed);
  const update = (sessionId: string, u: Json) => out({ jsonrpc: '2.0', method: 'session/update', params: { sessionId, update: u } });
  out({ $open: { cwd: '/tmp/fake-workers', command: 'fake-acp', pid: 1 } }, 1);
  let calls = 0;
  /** The rest of a prompt's turn, once its permission request is answered. */
  let pendingPrompt: (() => void) | undefined;
  return {
    send(data) {
      const m = data as { id?: number; method?: string; params?: { sessionId?: string }; result?: unknown };
      const ok = (result: Json) => out({ jsonrpc: '2.0', id: m.id!, result });
      if (m.method === 'initialize') ok({ protocolVersion: 1, agentCapabilities: {} });
      else if (m.method === 'session/new') ok({ sessionId: 'sess-fake', configOptions: ['model', 'effort'].map((id) => ({ id, name: id, type: 'select', currentValue: 'default', options: [] })) });
      else if (m.method === 'session/set_config_option') ok({ configOptions: [] });
      else if (m.method === 'session/prompt') {
        const sid = m.params!.sessionId!;
        const n = ++calls;
        update(sid, { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'Deep thought.' } });
        update(sid, { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Let me check.' } });
        update(sid, { sessionUpdate: 'tool_call', toolCallId: `tc${n}`, title: 'Fetch https://example.com', kind: 'fetch', status: 'pending' });
        out({ jsonrpc: '2.0', id: 1000 + n, method: 'session/request_permission', params: { sessionId: sid, toolCall: { toolCallId: `tc${n}`, title: 'Fetch https://example.com', kind: 'fetch' }, options: [{ optionId: 'allow', name: 'Allow', kind: 'allow_once' }, { optionId: 'reject', name: 'Reject', kind: 'reject_once' }] } });
        const id = m.id!;
        pendingPrompt = () => {
          update(sid, { sessionUpdate: 'tool_call_update', toolCallId: `tc${n}`, status: 'completed' });
          for (const tok of ['The answer ', 'is 42.']) update(sid, { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: tok } });
          out({ jsonrpc: '2.0', id, result: { stopReason: 'end_turn' } }, 100);
        };
      } else if (m.id !== undefined && m.method === undefined && m.result) pendingPrompt?.();
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

const openai: LiveDriver = (req, sink) => {
  const r = req as { body: { messages: unknown[] } };
  const tokens = `Compat reply ${r.body.messages.length}. Short and sweet!`.match(/\S+\s*/g)!;
  let closed = false;
  let t = jitter(2, 3);
  const ev = (e: Json) => {
    t += jitter(0.5, 2);
    later(t, () => !closed && sink.recv(e));
  };
  for (const tok of tokens) ev({ id: 'c1', choices: [{ index: 0, delta: { content: tok }, finish_reason: null }] });
  ev({ id: 'c1', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
  ev({ $done: true });
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

/** Deepgram Aura: 16-bit PCM, streamed in chunks that split samples. */
const deepgramTts: LiveDriver = (req, sink) => {
  const { body, query } = req as { body: { text: string }; query: { sample_rate: number } };
  const n = Math.round(body.text.length * 0.004 * query.sample_rate);
  const pcm = new Int16Array(n).map((_, i) => Math.round(Math.sin(i / 10) * 3000));
  const bytes = new Uint8Array(pcm.buffer);
  let closed = false;
  let off = 0;
  let t = jitter(2, 4);
  while (off < bytes.length) {
    const len = Math.min(bytes.length - off, 1 + Math.floor(Math.random() * 3000));
    const chunk = bytes.slice(off, off + len);
    off += len;
    t += jitter(0.2, 1);
    later(t, () => !closed && sink.recv(null, chunk));
  }
  later(t + 0.5, () => !closed && sink.close());
  return { send() {}, close: () => void (closed = true) };
};

/** Mimics speechSynthesis: start, a boundary per word, end (fast, jittered). */
const speechSynthesis: LiveDriver = (req, sink) => {
  const { text } = req as { text: string };
  let closed = false;
  let t = jitter(2, 4);
  later(t, () => !closed && sink.recv({ type: 'start', voice: 'fake' }));
  let charIndex = 0;
  for (const word of text.split(' ')) {
    const at = charIndex;
    later(t + jitter(0.5, 1), () => !closed && sink.recv({ type: 'boundary', name: 'word', charIndex: at }));
    charIndex += word.length + 1;
    t += jitter(1, 2);
  }
  later(t + 1, () => !closed && sink.close());
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

export const fakeDrivers: DriverRegistry = { mic, silero, deepgram, anthropic, openai, cartesia, 'deepgram-tts': deepgramTts, 'speech-synthesis': speechSynthesis, rnnoise, speaker, 'turn-model': turnModel, acp, typesafe };
