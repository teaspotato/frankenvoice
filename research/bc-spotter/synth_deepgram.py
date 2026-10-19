"""Synthesize the synth.py prompts with Deepgram Aura-2 English voices (same .npz format).

A second TTS family, for testing whether the classifier generalizes to unseen engines and
speakers. Needs DEEPGRAM_API_KEY, or a proxy that adds it.

  python synth_deepgram.py out.npz [--per-prompt 3] [--voices aura-2-thalia-en,...]
"""
import argparse, json, os, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from synth import prompts, trim
from vocab import IDX

SR = 16000


def voices():
    req = urllib.request.Request('https://api.deepgram.com/v1/models')
    m = json.load(urllib.request.urlopen(req))
    return sorted({v['canonical_name'] for v in m.get('tts', []) if v['canonical_name'].startswith('aura-2')
                   and str((v.get('languages') or ['en'])[0]).startswith('en')})


def say(text, voice):
    url = f'https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&sample_rate={SR}&container=none'
    h = {'content-type': 'application/json'}
    if os.environ.get('DEEPGRAM_API_KEY'):
        h['authorization'] = 'Token ' + os.environ['DEEPGRAM_API_KEY']
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, json.dumps({'text': text}).encode(), h), timeout=30).read()
            return trim(np.frombuffer(raw, '<i2').astype(np.float32) / 32768)
        except Exception:
            if attempt == 3:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--per-prompt', type=int, default=3); ap.add_argument('--voices')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    vs = a.voices.split(',') if a.voices else voices()
    jobs = []
    for c, text, cont in prompts():
        for v in random.sample(vs, min(a.per_prompt, len(vs))):
            jobs.append((c, text, cont, v))

    def run(j):
        c, text, cont, v = j
        x = say(text, v)
        first_end = len(say(text.split(' ')[0].rstrip(','), v)) / SR - 0.05 if cont else len(x) / SR
        more = cont if cont is not None else len(text.split()) > 1
        return dict(audio=x, y=IDX[c], text=text, voice=v, first_end=first_end, more=bool(more))

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(run, jobs))
    L = max(len(r['audio']) for r in rows)
    audio = np.zeros((len(rows), L), np.float16); lens = np.array([len(r['audio']) for r in rows])
    for i, r in enumerate(rows):
        audio[i, :lens[i]] = r['audio']
    np.savez_compressed(a.out, audio=audio, len=lens, y=np.array([r['y'] for r in rows]), text=np.array([r['text'] for r in rows]),
                        voice=np.array([r['voice'] for r in rows]), first_end=np.array([r['first_end'] for r in rows], np.float32),
                        more=np.array([r['more'] for r in rows]))
    print(len(rows), 'clips in', len(vs), 'voices ->', a.out)


if __name__ == '__main__':
    main()
