"""Synthesize short utterances that start with the vocabulary words, with Kokoro (Apache-2.0).

Each clip is one prompt in one voice at one speed: either a word on its own ("Yeah.",
"Mm-hmm.") or a sentence that starts with it ("So what about the timing?"). The label is
the first word's class plus roughly where that word ends (the opener synthesized alone
in the same voice and speed), so training knows whether more speech is audible inside a
given window. Non-English Kokoro voices reading English add accent variety.

Uses kokoro-onnx, which runs on CPU at ~0.3 s per clip:

  pip install kokoro-onnx
  curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
  python synth.py out.npz --kokoro-dir . [--per-prompt 3]
"""
import argparse, random
import numpy as np
from vocab import IDX

SR = 16000
VOICES = {
    'en-us': ['af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river',
              'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck'],
    'en-gb': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    # Accented English: these voices were trained on other languages.
    'accent': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'zf_xiaobei', 'zf_xiaoni', 'zm_yunxi',
               'ef_dora', 'em_alex', 'ff_siwis', 'hf_alpha', 'hm_omega', 'if_sara', 'im_nicola', 'pf_dora', 'pm_alex'],
}
HOLDOUT = ['af_nova', 'bm_george', 'jf_alpha']  # never trained on; scored as a sanity check

ALONE = {
    'yeah': ['Yeah.', 'Yeah!', 'Yeah?', 'Yeah, yeah.', 'Yes.', 'Yep.', 'Yup.', 'Yeah yeah yeah.'],
    'right': ['Right.', 'Right!', 'Right, right.', 'Right?'],
    'okay': ['Okay.', 'Okay!', 'OK.', 'Alright.', 'All right.', 'Okay okay.'],
    'mhm': ['Mm-hmm.', 'Mhm.', 'Uh-huh.', 'Mm.', 'Mm-hmm!'],
    'sure': ['Sure.', 'Sure!', 'Sure, sure.'],
    'cool': ['Cool.', 'Nice.', 'Great.', 'Nice!'],
    'so': ['So.', 'So…', 'So,', 'So?'],
    'wait': ['Wait.', 'Wait!', 'Wait, wait.', 'Wait…'],
    'but': ['But…', 'But,'],
    'actually': ['Actually.', 'Actually…'],
    'no': ['No.', 'No!', 'No, no.', 'Nope.'],
    'hold': ['Hold on.', 'Hold on!', 'Hold on…'],
    'sorry': ['Sorry.', 'Sorry…', 'Sorry?'],
    'well': ['Well.', 'Well…'],
    'um': ['Um.', 'Uh.', 'Hmm.', 'Um…', 'Uh…'],
    'and': ['And…', 'And,'],
}
TAILS = [
    "what about the timing?", "that's not what I meant.", "can you go back to the first part?", "I think it's the other one.",
    "how does that work?", "let me check that.", "it's usually the ASR problem.", "the second option is better.",
    "could you say that again?", "I was thinking about tomorrow.", "we already tried that.", "that makes sense.",
    "where did you get that?", "I clicked the wrong button.", "it keeps cutting me off.", "how long would it take?",
    "is that the default?", "I don't think so.", "tell me more about it.", "that sounds about right.",
]
OPENERS = {  # how each class starts a longer sentence
    'yeah': ['Yeah, ', 'Yes, ', 'Yeah. '], 'right': ['Right, ', 'Right. '], 'okay': ['Okay, ', 'Okay so ', 'Alright, '],
    'mhm': ['Mm-hmm, ', 'Uh-huh, '], 'sure': ['Sure, ', 'Sure but '], 'cool': ['Cool, ', 'Nice, '],
    'so': ['So ', 'So, ', 'So so '], 'wait': ['Wait, ', 'Wait '], 'but': ['But ', 'But, '], 'actually': ['Actually, ', 'Actually '],
    'no': ['No, ', 'No no, '], 'hold': ['Hold on, ', 'Hold on. '], 'sorry': ['Sorry, ', 'Sorry. '], 'well': ['Well, ', 'Well '],
    'um': ['Um, ', 'Uh, ', 'Hmm, '], 'and': ['And ', 'And, '],
}
OTHER = [  # sentences that start with none of the vocabulary words
    "I think that's wrong.", "Could you repeat that?", "That's interesting.", "What do you mean?", "How about New York?",
    "Can you see which part?", "There's a weird bug.", "Tell me more.", "I'm in Tokyo.", "Interesting.", "Exactly.", "Really?",
    "Oh.", "Oh really?", "Huh.", "Which one?", "Why?", "Thanks.", "Hello?", "Excuse me.", "Maybe.", "Probably.", "Got it.",
    "I see.", "True.", "Fine.", "Hey.", "What?", "Hang on.", "Let me think.", "It's broken.", "Could be.", "You do.",
]


def prompts():
    out = []
    for c, ss in ALONE.items():
        out += [(c, s, False) for s in ss]
    for c, ops in OPENERS.items():
        if not ops:
            continue
        for o in ops:
            for t in random.sample(TAILS, 4):
                out.append((c, o + t, True))
    out += [('other', s, None) for s in OTHER]
    return out


def trim(x):
    """Cut leading/trailing silence so sample 0 is the speech start (like an ASR word start)."""
    e = np.convolve(np.abs(x), np.ones(160) / 160, 'same')
    on = np.nonzero(e > 0.02 * e.max())[0]
    return x[max(0, on[0] - 160):on[-1] + 800] if len(on) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--kokoro-dir', default='.')
    ap.add_argument('--per-prompt', type=int, default=3, help='voices per prompt')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly
    k = Kokoro(f'{a.kokoro_dir}/kokoro-v1.0.onnx', f'{a.kokoro_dir}/voices-v1.0.bin')

    def say(text, voice, lang, speed):
        x, sr = k.create(text, voice=voice, speed=speed, lang='en-gb' if lang == 'en-gb' else 'en-us')
        return trim(resample_poly(x, 2, 3).astype(np.float32))  # 24 kHz -> 16 kHz

    rows, ps = [], prompts()
    for n, (c, text, cont) in enumerate(ps):
        for _ in range(a.per_prompt):
            lang = random.choices(list(VOICES), weights=[2, 1, 1])[0]
            voice = random.choice(VOICES[lang]); speed = random.uniform(0.8, 1.3)
            x = say(text, voice, lang, speed)
            if cont:  # where the first word ends: say the opener alone in the same voice
                first_end = len(say(text.split(' ')[0].rstrip(','), voice, lang, speed)) / SR - 0.05
            else:
                first_end = len(x) / SR
            more = cont if cont is not None else len(text.split()) > 1
            rows.append(dict(audio=x, y=IDX[c], text=text, voice=voice, first_end=first_end, more=bool(more)))
        if n % 50 == 0:
            print(n, '/', len(ps), text, flush=True)
    L = max(len(r['audio']) for r in rows)
    audio = np.zeros((len(rows), L), np.float16); lens = np.array([len(r['audio']) for r in rows])
    for i, r in enumerate(rows):
        audio[i, :lens[i]] = r['audio']
    np.savez_compressed(a.out, audio=audio, len=lens, y=np.array([r['y'] for r in rows]), text=np.array([r['text'] for r in rows]),
                        voice=np.array([r['voice'] for r in rows]), first_end=np.array([r['first_end'] for r in rows], np.float32),
                        more=np.array([r['more'] for r in rows]))
    print(len(rows), 'clips ->', a.out)


if __name__ == '__main__':
    main()
