"""Turn one recording of you reading takes.txt into labeled clips (same format as synth.py).

Read each line of takes.txt in order with a clear pause (~1 s) between lines; any format
ffmpeg reads works (record on your phone or with the browser). The script splits on
silence and labels the pieces in order, so it refuses to guess if the counts differ.

  python split_takes.py takes.txt recording.m4a rin_takes.npz [--voice rin]
  python train.py --synth synth.npz rin_takes.npz ...
"""
import argparse, subprocess, sys
import numpy as np
from vocab import IDX, word_class

SR = 16000


def read_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def split(x, min_gap=0.45, min_len=0.12):
    e = 10 * np.log10(np.convolve(x ** 2, np.ones(320) / 320, 'same') + 1e-12)
    on = e > max(np.percentile(e, 20) + 15, -55)
    segs, start, quiet = [], None, 0
    for i in range(0, len(x), 160):
        if on[i]:
            if start is None: start = i
            quiet = 0
        elif start is not None:
            quiet += 160
            if quiet >= min_gap * SR:
                end = i - quiet + 160
                if end - start >= min_len * SR: segs.append((start, end))
                start, quiet = None, 0
    if start is not None: segs.append((start, len(x)))
    return segs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('takes'); ap.add_argument('recording'); ap.add_argument('out'); ap.add_argument('--voice', default='rin')
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.takes) if l.strip() and not l.startswith('#')]
    x = read_audio(a.recording)
    segs = split(x)
    if len(segs) != len(lines):
        sys.exit(f'found {len(segs)} pieces for {len(lines)} lines; leave longer pauses, or trim the start/end, and retry')
    clips = [x[max(0, s - 160):e + 1600] for s, e in segs]  # sample 0 = speech start, as in synth.py
    L = max(len(c) for c in clips)
    audio = np.zeros((len(clips), L), np.float16)
    for i, c in enumerate(clips): audio[i, :len(c)] = c
    first = [word_class(l.split()[0].strip('.,!?…')) if not l.lower().startswith('hold on') else 'hold' for l in lines]
    np.savez_compressed(a.out, audio=audio, len=np.array([len(c) for c in clips]), y=np.array([IDX[c] for c in first]),
                        text=np.array(lines), voice=np.array([a.voice] * len(lines)),
                        # first word end is unknown for real takes: single words end at the clip end
                        first_end=np.array([len(c) / SR if len(l.split()) == 1 else 0.35 for c, l in zip(clips, lines)], np.float32),
                        more=np.array([len(l.split()) > 1 for l in lines]))
    print(len(lines), 'takes ->', a.out)
