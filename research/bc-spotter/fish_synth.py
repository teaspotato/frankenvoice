"""Synthesize the synth.py prompts with Fish Audio voices, split into train and unseen test voices.

Writes two files in synth.py's .npz format: <out>_train.npz and <out>_test.npz. Voices
come from fish_voices.json (public Fish models, character/celebrity voices dropped); a
seeded split keeps every test voice out of training. Clips are cached per request in
--cache, so a rerun resumes. Needs the Fish API key (the session proxy injects it):

  python fish_synth.py data/fish --per-prompt 3 [--test-frac 0.2] [--workers 4]
"""
import argparse, hashlib, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
from vocab import IDX
from synth import prompts, trim, SR

URL = 'https://api.fish.audio/v1/tts'
MODEL = 's2.1-pro-free'  # free tier; the account has no payment method


def tts(text, voice, speed, cache):
    key = hashlib.sha1(f'{voice}|{speed:.3f}|{text}'.encode()).hexdigest()
    path = f'{cache}/{key}.npy'
    if os.path.exists(path):
        return np.load(path)
    for _ in range(4):
        r = requests.post(URL, json=dict(text=text, reference_id=voice, format='pcm', sample_rate=SR, normalize=False,
                                         latency='normal', prosody=dict(speed=speed)), headers={'model': MODEL}, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5); continue
        r.raise_for_status()
        x = trim(np.frombuffer(r.content, np.int16).astype(np.float32) / 32768)
        np.save(path, x)
        return x
    r.raise_for_status()


def save(rows, out):
    L = max(len(r['audio']) for r in rows)
    audio = np.zeros((len(rows), L), np.float16); lens = np.array([len(r['audio']) for r in rows])
    for i, r in enumerate(rows):
        audio[i, :lens[i]] = r['audio']
    np.savez_compressed(out, audio=audio, len=lens, y=np.array([r['y'] for r in rows]), text=np.array([r['text'] for r in rows]),
                        voice=np.array([r['voice'] for r in rows]), first_end=np.array([r['first_end'] for r in rows], np.float32),
                        more=np.array([r['more'] for r in rows]))
    print(len(rows), 'clips ->', out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--voices', default=os.path.join(os.path.dirname(__file__), 'fish_voices.json'))
    ap.add_argument('--per-prompt', type=int, default=3); ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache', default='fish_cache')
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    random.seed(a.seed)
    voices = [v[0] for v in json.load(open(a.voices))]
    random.shuffle(voices)
    n_test = round(len(voices) * a.test_frac)
    split = {'test': voices[:n_test], 'train': voices[n_test:]}
    json.dump(split, open(f'{a.out}_split.json', 'w'), indent=1)
    ps = prompts()
    for name, vs in split.items():
        per = a.per_prompt if name == 'train' else 1
        jobs = [(c, text, cont, random.choice(vs), random.uniform(0.85, 1.25)) for c, text, cont in ps for _ in range(per)]

        done = []

        def run(j):
            c, text, cont, voice, speed = j
            try:
                x = tts(text, voice, speed, a.cache)
                first_end = (len(tts(text.split(' ')[0].rstrip(','), voice, speed, a.cache)) / SR - 0.05) if cont else len(x) / SR
                if cont and first_end > 0.8 * len(x) / SR:  # the lone opener sometimes runs on for seconds
                    print('skip', voice, text, 'first word', round(first_end, 2), flush=True); return None
            except requests.HTTPError as e:
                print('skip', voice, text, e, flush=True); return None
            more = cont if cont is not None else len(text.split()) > 1
            done.append(1)
            if len(done) % 50 == 0:
                print(name, len(done), '/', len(jobs), flush=True)
            return dict(audio=x, y=IDX[c], text=text, voice=voice, first_end=first_end, more=bool(more))

        with ThreadPoolExecutor(a.workers) as ex:
            rows = [r for r in ex.map(run, jobs) if r is not None]
        save(rows, f'{a.out}_{name}.npz')


if __name__ == '__main__':
    main()
