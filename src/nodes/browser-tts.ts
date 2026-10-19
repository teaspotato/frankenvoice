import type { IOHandle, NodeDef } from '../core/node';
import type { Msg } from '../core/types';
import { SentenceSplitter } from './segment';

interface Segment {
  rid: string;
  seg: number;
  text: string;
}

/**
 * Text to speech with the browser's built-in speechSynthesis, standing in for
 * both the tts and speaker nodes. The browser plays the audio itself, so no
 * samples are recorded; what is recorded is when each sentence started, every
 * word boundary the browser reported, and when it ended. Replay reproduces the
 * playback events from that timing, and audible replay speaks the text again.
 */
export const browserTtsNode: NodeDef = {
  type: 'tts.browser',
  title: 'Browser TTS',
  category: 'tts',
  description: 'Built-in speechSynthesis. Speaks LLM text sentence by sentence and reports playback timing (no API key, no recorded audio).',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'playback', type: 'playback' }],
  params: [
    { key: 'voice', label: 'Voice', type: 'string', default: '', description: 'Voice name as listed by speechSynthesis.getVoices(); empty uses the default for the language.' },
    { key: 'lang', label: 'Language', type: 'string', default: 'en-US' },
    { key: 'rate', label: 'Rate', type: 'number', default: 1 },
    { key: 'pitch', label: 'Pitch', type: 'number', default: 1 },
    { key: 'minSegmentChars', label: 'Min segment chars', type: 'number', default: 1 },
  ],
  channels: [{ name: 'speech-synthesis', description: 'One utterance per sentence; recv start/boundary events, close on end', output: true }],
  create(ctx) {
    const p = ctx.params as { voice: string; lang: string; rate: number; pitch: number; minSegmentChars: number };
    let splitter = new SentenceSplitter(p.minSegmentChars);
    let queue: Segment[] = [];
    let current: IOHandle | undefined;
    let rid = '';
    let segNo = 0;
    let llmDone = false;
    let playing = false;
    /** What has been said of this reply: finished sentences, and the current one up to its last word boundary. */
    let said: string[] = [];
    let speaking: { text: string; upTo: number } | undefined;

    const heard = () => {
      if (!speaking) return said.join(' ');
      // A boundary marks the start of a word; count that word as heard.
      const rest = speaking.text.slice(speaking.upTo);
      const wordEnd = speaking.upTo + (rest.match(/^\S*/)?.[0].length ?? 0);
      return [...said, speaking.text.slice(0, wordEnd).trim()].filter(Boolean).join(' ');
    };

    const setStatus = (s: string) => ctx.setState('status', s);

    /** Speak the next queued sentence once the previous one has finished. */
    const pump = () => {
      if (current) return;
      const s = queue.shift();
      if (s) {
        setStatus('speaking');
        const io = ctx.io.open('speech-synthesis', { text: s.text, voice: p.voice, lang: p.lang, rate: p.rate, pitch: p.pitch }, {
          onRecv(data) {
            const d = data as { type: string; charIndex?: number };
            if (d.type === 'start') {
              speaking = { text: s.text, upTo: 0 };
              if (!playing) {
                playing = true;
                ctx.setState('playing', true);
                ctx.emit('playback', { type: 'playback.start', rid: s.rid, at: ctx.now() });
              }
              ctx.emit('playback', { type: 'playback.segment', rid: s.rid, seg: s.seg, text: s.text, at: ctx.now() });
            } else if (d.type === 'boundary') {
              if (speaking) speaking.upTo = d.charIndex ?? 0;
              ctx.emit('playback', { type: 'playback.boundary', rid: s.rid, seg: s.seg, charIndex: d.charIndex ?? 0, at: ctx.now() });
            }
          },
          onClose(data) {
            if (current !== io) return;
            current = undefined;
            if (speaking) said.push(speaking.text);
            speaking = undefined;
            if (data && (data as { error?: unknown }).error) ctx.log('error', 'speech synthesis failed', data);
            pump();
          },
        });
        current = io;
        return;
      }
      if (llmDone && rid) {
        if (playing) {
          playing = false;
          ctx.setState('playing', false);
          ctx.emit('playback', { type: 'playback.end', rid });
        }
        setStatus('idle');
        rid = '';
      }
    };

    const cancel = () => {
      const io = current;
      current = undefined;
      io?.close({ status: 'cancelled' });
      if (playing) {
        playing = false;
        ctx.setState('playing', false);
        ctx.emit('playback', { type: 'playback.interrupted', rid, at: ctx.now(), heard: heard() });
      }
      queue = [];
      rid = '';
      splitter = new SentenceSplitter(p.minSegmentChars);
      setStatus('idle');
    };

    const enqueue = (text: string) => {
      queue.push({ rid, seg: segNo++, text });
      pump();
    };

    return {
      onInput(port, msg: Msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') cancel();
          return;
        }
        switch (msg.type) {
          case 'llm.start':
            if (rid) cancel();
            rid = msg.rid as string;
            said = [];
            speaking = undefined;
            segNo = 0;
            llmDone = false;
            break;
          case 'opener.say':
          case 'llm.delta':
            if (msg.rid !== rid) return;
            for (const seg of splitter.push(msg.text as string)) enqueue(seg);
            if (msg.type === 'opener.say') {
              // An opener is said as soon as it arrives, whatever its punctuation.
              const rest = splitter.flush();
              if (rest) enqueue(rest);
            }
            break;
          case 'llm.end': {
            if (msg.rid !== rid) return;
            const rest = splitter.flush();
            llmDone = true;
            if (rest) enqueue(rest);
            else pump();
            break;
          }
        }
      },
      stop() {
        current?.close({ status: 'stopped' });
        current = undefined;
      },
    };
  },
};
