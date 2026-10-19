"""Build banks.npz from OpenSLR 28 (RIRS_NOISES, Apache-2.0): real room impulse responses and real noises.

  curl -LO https://www.openslr.org/resources/28/rirs_noises.zip && unzip rirs_noises.zip
  python prep_banks.py RIRS_NOISES banks.npz
"""
import glob, os, random, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def load(path, ch=0):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x[:, ch % x.shape[1]]
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    return x


root, out = sys.argv[1], sys.argv[2]
random.seed(0)
rir_files = glob.glob(f'{root}/real_rirs_isotropic_noises/*.wav')
rir_files = [f for f in rir_files if 'noise' not in os.path.basename(f).lower()]
noise_files = glob.glob(f'{root}/pointsource_noises/*.wav') + \
    [f for f in glob.glob(f'{root}/real_rirs_isotropic_noises/*.wav') if 'noise' in os.path.basename(f).lower()]
rirs = []
for f in random.sample(rir_files, min(400, len(rir_files))):
    h = load(f, random.randrange(8))
    i = int(np.argmax(np.abs(h))); h = h[max(0, i - 16):i + int(0.8 * SR)]  # start at the direct path
    h = h / (np.abs(h).max() + 1e-9)
    rirs.append(np.pad(h, (0, int(0.8 * SR) + 16 - len(h))))
noises = []
L = 3 * SR
for f in random.sample(noise_files, min(600, len(noise_files))):
    x = load(f, random.randrange(8))
    if len(x) < SR // 2 or np.std(x) < 1e-5:
        continue
    i = random.randrange(max(1, len(x) - L)); x = x[i:i + L]
    x = np.tile(x, int(np.ceil(L / len(x))))[:L]
    noises.append(x / (np.std(x) + 1e-9))
np.savez_compressed(out, rirs=np.stack(rirs).astype(np.float16), noises=np.stack(noises).astype(np.float16))
print(len(rirs), 'rirs', len(noises), 'noises ->', out)
