"""Simulate the browser mic chain on clean speech: room, mic, echo leak, AEC, noise suppression, AGC.

What Rin's recorded mic shows during overlaps, and what each stage reproduces:
- the room and a laptop/headset mic:      reverb + band-limited, tilted response
- the agent leaking back from speakers:   echo through its own room path, partly cancelled
- double-talk suppression (AEC's residual echo suppressor): near-end speech pushed down
  10-30 dB while the agent is loud, which is why short words like "So" come out faint
- noise suppression gating:               frames below a noise-relative threshold drop to
  near digital silence (-80 to -90 dB), sometimes in the middle of a word
- AGC:                                    slow gain that starts low on a fresh onset

simulate(speech, agent, noise, rng, banks=None) -> the mic signal, all at 16 kHz.
banks (prep_banks.py) adds real room responses and real environmental noise.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt, stft, istft

SR = 16000


def db(x):
    return 10 ** (x / 20)


def rir(rng, rt60):
    """Synthetic room impulse response: direct path + exponentially decaying noise tail."""
    n = int(SR * min(rt60 * 1.2, 0.8))
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[:int(SR * rng.uniform(0.002, 0.01))] = 0  # gap before the first reflections
    h = tail * db(rng.uniform(-18, -6)) / (np.sqrt(np.sum(tail ** 2)) + 1e-9)
    h[0] = 1.0
    return h.astype(np.float32)


def mic_response(x, rng):
    """Band-limit and tilt like a cheap mic: high-pass, low-pass, and one resonance or dip."""
    hp = butter(2, rng.uniform(70, 300), 'highpass', fs=SR, output='sos')
    lp = butter(4, rng.uniform(3400, 7600), 'lowpass', fs=SR, output='sos')
    y = sosfilt(lp, sosfilt(hp, x))
    f0, q, g = rng.uniform(800, 5000), rng.uniform(0.7, 3), rng.uniform(-8, 8)
    w0 = 2 * np.pi * f0 / SR; A = db(g / 2); al = np.sin(w0) / (2 * q)  # peaking EQ (RBJ)
    b = [1 + al * A, -2 * np.cos(w0), 1 - al * A]; a = [1 + al / A, -2 * np.cos(w0), 1 - al / A]
    return lfilter(b, a, y).astype(np.float32)


def envelope(x, win=160):
    return np.sqrt(np.convolve(x ** 2, np.ones(win) / win, 'same') + 1e-12)


def smooth_gain(g_db, attack=0.005, release=0.08, hop=40):
    """One-pole smoothing of a per-sample gain in dB (fast to duck, slower to recover), at 2.5 ms hops."""
    v = g_db[::hop]; out = np.empty_like(v); cur = v[0]
    a_up, a_dn = np.exp(-hop / (release * SR)), np.exp(-hop / (attack * SR))
    for i, x in enumerate(v):
        k = a_dn if x < cur else a_up
        cur = k * cur + (1 - k) * x
        out[i] = cur
    return np.repeat(out, hop)[:len(g_db)]


def pink(n, rng):
    w = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(w)); f[0] = 1
    x = np.fft.irfft(w / np.sqrt(f), n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def double_talk(near, agent_at_mic, rng):
    """Residual echo suppression: attenuate everything while the agent is loud at the mic."""
    depth = rng.uniform(0, 30)
    if depth < 3:
        return near
    ae = 20 * np.log10(envelope(agent_at_mic, 320) + 1e-9)
    on = np.clip((ae - ae.max() + 25) / 15, 0, 1)  # 0..1 as the agent gets within 25 dB of its peak
    g = smooth_gain(-depth * on, attack=0.003, release=rng.uniform(0.03, 0.15))
    return near * db(g).astype(np.float32)


def noise_suppress(x, rng):
    """Spectral subtraction + a frame gate that drops quiet frames to near digital silence."""
    n = 256
    f, t, Z = stft(x, SR, nperseg=n, noverlap=n // 2)
    mag = np.abs(Z)
    floor = np.percentile(mag, rng.uniform(10, 30), axis=1, keepdims=True)
    over = rng.uniform(1.0, 3.0)
    g = np.clip(1 - over * floor / (mag + 1e-9), db(-rng.uniform(15, 30)), 1)
    fe = 20 * np.log10(np.sqrt(np.mean(mag ** 2, 0)) + 1e-9)
    gate_th = np.percentile(fe, rng.uniform(5, 30)) + rng.uniform(0, 10)
    gate = np.where(fe < gate_th, db(-rng.uniform(50, 70)), 1.0)
    gate = np.convolve(gate, np.ones(2) / 2, 'same')  # no clicks
    _, y = istft(Z * g * gate, SR, nperseg=n, noverlap=n // 2)
    y = y[:len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)


def agc(x, rng, target_db=-20):
    """Slow automatic gain toward a target RMS, starting from a low gain (fresh onset)."""
    start = rng.uniform(-15, 0)
    env = 20 * np.log10(envelope(x, 800) + 1e-9)
    want = np.clip(target_db - env, -10, 25)
    want = np.where(env < -60, start, want)  # don't chase silence
    g = np.empty_like(want); cur = start; a = np.exp(-1 / (rng.uniform(0.2, 0.8) * SR))
    for i, v in enumerate(want[::40]):
        cur = a ** 40 * cur + (1 - a ** 40) * v
        g[i * 40:(i + 1) * 40] = cur
    return x * db(g[:len(x)]).astype(np.float32)


def simulate(speech, agent, noise, rng, banks=None):
    """speech, agent, noise: float32 at 16 kHz, same length. Returns the simulated mic signal."""
    speech = speech.astype(np.float32); agent = agent.astype(np.float32); noise = noise.astype(np.float32)
    n = len(speech)
    # Room and mic for the user.
    def room_ir():
        if banks is not None and rng.random() < 0.6:
            return banks['rirs'][rng.integers(len(banks['rirs']))].astype(np.float32)
        return rir(rng, rng.uniform(0.15, 0.6))
    near = fftconvolve(speech, room_ir())[:n]
    near = near * db(rng.uniform(-34, -16)) / (np.sqrt(np.mean(near ** 2)) + 1e-9) * (np.abs(speech).max() > 0)
    # Agent from the speakers through its own path, mildly distorted, mostly cancelled by AEC.
    echo = fftconvolve(np.tanh(agent * rng.uniform(1, 4)) / rng.uniform(1, 4), room_ir())[:n]
    echo_at_mic = echo * db(rng.uniform(-25, -10))
    residual = echo_at_mic * db(-rng.uniform(15, 35))  # what the linear AEC leaves
    room = pink(n, rng) * db(rng.uniform(-62, -42))  # room noise before noise suppression
    if banks is not None and rng.random() < 0.7:  # real environmental noise at 0-30 dB SNR
        z = banks['noises'][rng.integers(len(banks['noises']))].astype(np.float32)
        i = rng.integers(0, max(1, len(z) - n)); z = np.pad(z[i:i + n], (0, max(0, n - len(z[i:i + n]))))
        room = room + z * np.sqrt(np.mean(near ** 2) + 1e-12) * db(-rng.uniform(0, 30))
    x = mic_response(near + residual + room + noise * db(rng.uniform(-6, 6)), rng)
    if rng.random() < 0.8:
        x = double_talk(x, echo_at_mic, rng)
    if rng.random() < 0.85:
        x = noise_suppress(x, rng)
    if rng.random() < 0.7:
        x = agc(x, rng)
    if rng.random() < 0.05:
        x = np.clip(x, -0.3, 0.3)
    # The browser never outputs true zeros: gated frames sit near -88 dB.
    x = x + rng.standard_normal(n).astype(np.float32) * db(rng.uniform(-92, -84))
    return x.astype(np.float32)
