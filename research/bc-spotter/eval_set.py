"""The 10 hand-labeled overlaps in "Full Duplex Voice Pipeline Bugs 2" (research/backchannel-vs-interruption.md).

Times are the VAD onsets on the mic track's own clock (cut_real.load()). `first` is the
first word class, `decision` what the agent should have done.

  python eval_set.py SESSION_DIR out.npz
"""
import sys
import numpy as np
from cut_real import load
from vocab import IDX

SR = 16000
PRE, POST = 0.6, 1.6  # seconds kept around the VAD onset
OVERLAPS = [
    (55.58, 'yeah', 'floor', 'Yeah. That\'s usually the…'),
    (74.14, 'other', 'floor', 'I clicked…'),
    (111.52, 'yeah', 'backchannel', 'Yeah.'),
    (115.04, 'right', 'backchannel', 'Right.'),
    (136.26, 'okay', 'floor', 'Okay. So so…'),
    (169.57, 'yeah', 'aside', 'Yeah. Could be. (answers the agent\'s question)'),
    (186.11, 'so', 'floor', 'So (gave up)'),
    (188.32, 'so', 'floor', 'So (gave up)'),
    (194.59, 'so', 'floor', 'So'),
    (227.62, 'yeah', 'backchannel', 'Yeah.'),
]

if __name__ == '__main__':
    mic, agent, segs, words = load(sys.argv[1])
    rows = []
    for t, first, dec, text in OVERLAPS:
        a = int((t - PRE) * SR)
        rows.append((mic[a:a + int((PRE + POST) * SR)], agent[a:a + int((PRE + POST) * SR)], IDX[first], dec, text, t))
    np.savez_compressed(sys.argv[2], audio=np.stack([r[0] for r in rows]), agent=np.stack([r[1] for r in rows]),
                        y=np.array([r[2] for r in rows]), decision=np.array([r[3] for r in rows]),
                        text=np.array([r[4] for r in rows]), t=np.array([r[5] for r in rows]), pre=PRE)
    print(len(rows), 'overlaps')
