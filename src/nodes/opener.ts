import type { NodeDef, TimerHandle } from '../core/node';
import type { JsonObject } from '../core/types';

export const DEFAULT_FILLER_LINES = 'Let me see.\nHmm, let me think.\nOne moment.\nLet me think about that.';

const ports = {
  inputs: [{ name: 'text', type: 'text', description: "The LLM's reply stream (llm.start, llm.thinking, llm.delta, llm.end)" }],
  outputs: [{ name: 'text', type: 'text', description: 'opener.say: a line for TTS to speak on the reply, before its words' }],
};

/**
 * Says a filler ("Let me see.") on a reply while a reasoning model keeps the
 * user waiting: the model is thinking and no words have come `afterMs` after
 * the turn was committed. It only watches the LLM's stream and speaks through
 * TTS on the same reply; the LLM never sees it, so its prompt, history and
 * cache are untouched. Lines rotate and the last one is never said twice in a row.
 */
export const fillerOpenerNode: NodeDef = {
  type: 'opener.filler',
  title: 'Thinking filler',
  category: 'llm',
  description: 'Says a filler line while a reasoning model is still thinking after the turn. The LLM never knows.',
  ...ports,
  params: [
    { key: 'lines', label: 'Filler lines', type: 'text', default: DEFAULT_FILLER_LINES, description: 'One per line, said in turn.' },
    {
      key: 'afterMs',
      label: 'After (ms)',
      type: 'number',
      default: 500,
      description: 'Say a filler if the model is still thinking this long after the turn was committed, with no words yet.',
    },
  ],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const lines = String(p.lines ?? '')
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    let n = 0;
    let last = '';
    /** The reply in flight: when it started, whether the model is thinking, whether anything was said. */
    let cur: { rid: string; t0: number; thinking: boolean; spoken: boolean; timer?: TimerHandle } | undefined;

    const check = () => {
      if (!cur || !cur.thinking || cur.spoken || !lines.length) return;
      const waited = ctx.now() - cur.t0;
      if (waited < Number(p.afterMs)) return;
      let line = lines[n++ % lines.length];
      if (line === last && lines.length > 1) line = lines[n++ % lines.length];
      last = line;
      cur.spoken = true;
      ctx.emit('text', { type: 'opener.say', rid: cur.rid, text: `${line} `, line, reason: `still thinking ${Math.round(waited)} ms after the turn` });
    };
    const end = () => {
      cur?.timer?.clear();
      cur = undefined;
    };

    return {
      start() {
        // Tell TTS the lines, so it can have them ready.
        if (lines.length) ctx.emit('text', { type: 'opener.lines', lines: [...new Set(lines)] });
      },
      onInput(_port, msg) {
        switch (msg.type) {
          case 'llm.start':
            end();
            cur = { rid: msg.rid as string, t0: ctx.now(), thinking: false, spoken: false };
            cur.timer = ctx.setTimer(Number(p.afterMs), check, 'filler');
            break;
          case 'llm.thinking':
            if (!cur || cur.rid !== msg.rid) return;
            cur.thinking = true;
            check();
            break;
          case 'llm.delta':
            // The model is talking: too late for a filler.
            if (cur && cur.rid === msg.rid) cur.spoken = true;
            break;
          case 'llm.end':
            if (cur?.rid === msg.rid) end();
            break;
        }
      },
      stop: end,
    };
  },
};

export const noOpenerNode: NodeDef<Record<string, never>> = {
  type: 'opener.off',
  title: 'No opener',
  category: 'llm',
  description: 'Replies start with the LLM\'s own words.',
  ...ports,
  params: [],
  create() {
    return {};
  },
};
