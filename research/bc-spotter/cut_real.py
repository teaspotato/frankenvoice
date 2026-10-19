"""Cut word-onset windows from recorded Frankenvoice sessions (.fvs, unzipped).

Every Deepgram word with timing becomes one example: mic audio from 0.2 s before the word
to 1.4 s after, its class (vocab.py), when the next word starts, and whether the word
opened a VAD segment and whether the agent was audible. Eval sessions keep only the
overlap onsets; train sessions keep everything.

  python cut_real.py out.npz SESSION_DIR [SESSION_DIR ...]
  python cut_real.py --overlaps-only eval.npz SESSION_DIR
"""
import argparse, json, sys
import numpy as np
from scipy.signal import resample_poly
from vocab import IDX, word_class

SR = 16000
PRE, POST = 0.2, 1.4  # seconds around the word start


def load(d):
    meta = json.load(open(f'{d}/meta.json'))
    mic = np.fromfile(f"{d}/{meta['tracks']['mic.audio']['file']}", dtype='<f4')
    tts = np.fromfile(f"{d}/{meta['tracks']['tts.audio']['file']}", dtype='<f4')
    ev = [json.loads(l) for l in open(f'{d}/events.jsonl')]
    # Session ms -> mic sample. The mic track drops samples now and then (up to ~1 s behind
    # the session clock by the end of a 4-minute session), so map through every mic chunk.
    mm = [e for e in ev if e['node'] == 'mic' and e['kind'] == 'msg' and 'media' in e]
    mt = np.array([e['t'] for e in mm]); ms = np.array([e['media']['end'] for e in mm], float)
    def sample(t):
        return float(np.interp(t, mt, ms, left=ms[0] - (mt[0] - t) * 16, right=ms[-1] + (t - mt[-1]) * 16))
    agent = np.zeros_like(mic)  # played TTS on the mic clock
    for e in ev:
        if e['node'] == 'speaker' and e.get('port') == 'played' and 'media' in e and (e.get('data') or {}).get('at') is not None:
            x = resample_poly(tts[e['media']['start']:e['media']['end']], 2, 3).astype(np.float32)
            i = int(round(sample(e['data']['at'])))
            if 0 <= i < len(agent):
                agent[i:i + len(x)] = x[:len(agent) - i]
    segs, cur = [], None
    for e in ev:
        if e['node'] == 'vad' and e['kind'] == 'state' and 'speaking' in (e.get('data') or {}):
            if e['data']['speaking']:
                cur = sample(e['t']) / SR
            elif cur is not None:
                segs.append((cur, sample(e['t']) / SR)); cur = None
    words = []
    for e in ev:
        dd = e.get('data') or {}
        if e['node'] == 'asr' and e['kind'] == 'msg' and dd.get('type') == 'asr.final' and dd.get('track', 'mic.audio') == 'mic.audio':
            for w in dd.get('words') or []:
                words.append((w['start'] / SR, w['end'] / SR, w['word']))
    words.sort()
    return mic, agent, segs, words


def agent_env(agent):
    return np.convolve(np.abs(agent), np.ones(1600) / 1600, 'same')


def cut(d, overlaps_only=False, exclude=()):
    mic, agent, segs, words = load(d)
    env = agent_env(agent)
    n = int((PRE + POST) * SR)
    out = []
    for k, (ws, we, w) in enumerate(words):
        if any(a - 3 <= ws <= b + 3 for a, b in exclude):
            continue
        onset = next((a for a, b in segs if a - 0.4 <= ws <= a + 0.6), None)
        nxt = words[k + 1][0] if k + 1 < len(words) else 1e9
        i = int(ws * SR)
        overlap = bool(env[max(0, i - 3200):i + 1600].max() > 0.003)
        if overlaps_only and not (overlap and onset is not None):
            continue
        a = int((ws - PRE) * SR)
        if a < 0 or a + n > len(mic):
            continue
        out.append(dict(audio=mic[a:a + n].copy(), agent=agent[a:a + n].copy(), y=IDX[word_class(w)], word=w,
                        dur=we - ws, next=min(nxt - ws, 9.0), onset=onset is not None, overlap=overlap, t=ws))
    # Pure silence/noise windows: VAD-quiet stretches, label `none`.
    if not overlaps_only:
        busy = np.zeros(len(mic), bool)
        for a, b in segs:
            busy[max(0, int((a - 0.5) * SR)):int((b + 0.5) * SR)] = True
        for a in range(0, len(mic) - n, int(2.0 * SR)):
            if not busy[a:a + n].any():
                out.append(dict(audio=mic[a:a + n].copy(), agent=agent[a:a + n].copy(), y=IDX['none'], word='', dur=0, next=9.0,
                                onset=False, overlap=bool(env[a:a + n].max() > 0.003), t=a / SR + PRE))
    return out


def save(path, rows, src):
    np.savez_compressed(path, audio=np.stack([r['audio'] for r in rows]).astype(np.float16),
                        agent=np.stack([r['agent'] for r in rows]).astype(np.float16),
                        y=np.array([r['y'] for r in rows]), word=np.array([r['word'] for r in rows]),
                        dur=np.array([r['dur'] for r in rows], np.float32), next=np.array([r['next'] for r in rows], np.float32),
                        onset=np.array([r['onset'] for r in rows]), overlap=np.array([r['overlap'] for r in rows]),
                        t=np.array([r['t'] for r in rows], np.float32), src=np.array(src), pre=PRE)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--overlaps-only', action='store_true')
    ap.add_argument('--exclude-overlaps-of', help='session whose overlap onsets (±3 s) are left out')
    ap.add_argument('out'); ap.add_argument('sessions', nargs='+')
    a = ap.parse_args()
    rows, src = [], []
    for d in a.sessions:
        ex = ()
        if a.exclude_overlaps_of and d.rstrip('/') == a.exclude_overlaps_of.rstrip('/'):
            ex = [(r['t'], r['t'] + 1) for r in cut(d, overlaps_only=True)]
        r = cut(d, a.overlaps_only, ex)
        print(d, len(r), 'examples', file=sys.stderr)
        rows += r; src += [d.rstrip('/').split('/')[-1]] * len(r)
    save(a.out, rows, src)
