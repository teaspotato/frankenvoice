import type { NodeDef } from '../core/node';

interface TurnParams {
  commitOn: string;
}

type Floor = 'listening' | 'thinking' | 'speaking';

/**
 * Half-duplex turn taking: the agent takes the floor when the user's utterance
 * is final and gives it back when playback ends. User speech while the agent
 * talks is ignored (and logged). Step 2 replaces this with a full-duplex floor controller.
 */
export const halfDuplexTurnNode: NodeDef<TurnParams> = {
  type: 'turn.half-duplex',
  title: 'Half-duplex turns',
  category: 'control',
  description: 'Commits a user turn on ASR endpointing; ignores the user while the agent is speaking.',
  inputs: [
    { name: 'transcript', type: 'transcript' },
    { name: 'vad', type: 'vad' },
    { name: 'playback', type: 'playback' },
  ],
  outputs: [{ name: 'turn', type: 'turn' }],
  params: [{ key: 'commitOn', label: 'Commit on', type: 'select', options: ['speech_final', 'utterance_end'], default: 'speech_final' }],
  create(ctx) {
    let floor: Floor = 'listening';
    let parts: string[] = [];
    let from = -1;
    let to = -1;
    let turnNo = 0;
    const setFloor = (f: Floor, why: string) => {
      floor = f;
      ctx.setState('floor', f);
      ctx.log('info', `floor -> ${f}`, { why });
    };
    const commit = (why: string) => {
      const text = parts.join(' ').trim();
      if (!text) return;
      const turnId = `${ctx.id}-u${turnNo++}`;
      ctx.emit('turn', { type: 'turn.user', turnId, text, from, to });
      parts = [];
      from = to = -1;
      ctx.setState('pending', '');
      setFloor('thinking', why);
    };
    ctx.setState('floor', floor);

    return {
      onInput(port, msg) {
        if (port === 'playback') {
          if (msg.type === 'playback.start') setFloor('speaking', 'playback started');
          else if (msg.type === 'playback.end' && floor === 'speaking') setFloor('listening', 'playback ended');
          return;
        }
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start' || msg.type === 'vad.speech_end') ctx.setState('userSpeaking', msg.type === 'vad.speech_start');
          return;
        }
        if (floor === 'speaking') {
          if (msg.type === 'asr.final' && msg.text) ctx.log('info', 'ignored user speech while agent speaking', { text: msg.text as string });
          return;
        }
        if (msg.type === 'asr.final') {
          const text = (msg.text as string).trim();
          if (text) {
            parts.push(text);
            if (from < 0) from = msg.start as number;
            to = msg.end as number;
            ctx.setState('pending', parts.join(' '));
          }
          if (msg.speechFinal && ctx.params.commitOn === 'speech_final') commit('speech_final');
        } else if (msg.type === 'asr.utterance_end') {
          commit('utterance_end');
        }
      },
    };
  },
};
