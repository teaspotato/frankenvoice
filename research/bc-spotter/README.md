# Overlap word classifier on a small ASR encoder (study, 2026-10-19)

The question: can a small local ASR model, deliberately overfit to the ~16 words that
decide an overlap (yeah, right, okay, mhm… vs. so, wait, but, no, hold on…), name the
first word fast enough to replace the 1.4–1.7 s wait for Deepgram after the hard cut?

**Short answer: the setup works and is fast, but training on synthetic speech alone
does not transfer to your mic yet.** The model learns the TTS voices (about 90% on
voices it never saw) and gets only 4–6 of the 10 Bugs 2 overlaps, no better than the
untrained model. The gap is the channel: your browser-processed mic, not the voices.
The next step that matters is recording takes **through Frankenvoice while the agent
talks**, so training audio goes through the same echo cancellation and noise
suppression.

## Design

- **Fixed window, no streaming.** At VAD onset, take 0.5 s before it to W after it, run
  it once, and decide. One model serves any W from 0.25 to 1.0 s, so the turn stage can
  ask at +300 ms and again at +500 ms.
- **Substrate:** Moonshine tiny encoder (7.7M parameters, MIT, 40 frames/s), fine-tuned
  whole, with attention pooling and two heads:
  - the first word's class: 16 words, `other` and `none` (`vocab.py`);
  - `p(more)`: whether speech continues past that word inside the window.
- **Decision** (`decide()` in `train.py`):
  - a floor opener (so, wait, but, actually, no, hold on, sorry, well) means stop;
  - a backchannel word with no more speech means resume;
  - `none` (noise) means resume;
  - everything else means keep listening, which is today's behavior.
- **Cost:** about 8 ms per decision on one native CPU core with onnxruntime, for a
  0.8–1.1 s window. The model is 32 MB as fp32 ONNX. `train.py --export` writes it with a
  dynamic length.

## Data

| Source | Clips | Script |
|---|---|---|
| Kokoro TTS: 44 voices (US, UK and 17 accented non-English voices), speed 0.8–1.3, each word alone and as a sentence opener | 1,428 | `synth.py` (kokoro-onnx, CPU, ~25 min) |
| Your real word onsets from the 5 sessions, with Deepgram word timings, plus noise-only windows. Bugs 2 overlaps are excluded ±3 s. | 369 | `cut_real.py` |
| Mixing, per batch: your sessions' mic noise, agent playback as echo residue (−40 to −15 dB, random low-pass), random gain, and random 50–200 ms suppression dips | | `train.py` |
| **Eval:** the 10 labeled overlaps in Bugs 2 | 10 | `eval_set.py` |
| Your own takes: read `takes.txt` (108 lines) into any recorder | 0 so far | `split_takes.py` |

The real sessions contain almost none of the vocabulary words outside overlaps: 4 yeah,
3 okay, 3 mhm, 2 right, 1 so. That's why synthetic speech had to carry the training.

## Results on Bugs 2 (10 overlaps)

"Decided" means the classifier would act without waiting for Deepgram. "Wrong" means it
resumed on a floor-take or stopped on a backchannel.

| Model | W | First word right | Decided | Wrong |
|---|---|---|---|---|
| Moonshine tiny, untrained (transcript, then word list) | +0.4 s | 6/10 | n/a | misses all 3 "So" |
| Fine-tuned, 800 steps, 0.3 s pre-roll | +0.4 s | 6/10 | 5 | 1 |
| Fine-tuned, 3,000 steps, 0.5 s pre-roll | +0.4 s | 4/10 | 4 | 2 |

- **Backchannels:** the 800-step model got all three ("Yeah.", "Right.", "Yeah.") right
  and confidently at +0.4 s (p ≥ 0.89, p(more) ≤ 0.07). That is about 1–1.3 s sooner
  than Deepgram.
- **"So":** it caught one of three. It called the two quiet "So (gave up)" attempts `none`
  or `other`, so it would wrongly resume on the first one. Off-the-shelf Moonshine hears
  those two as "Yeah", "Wait" or "You know", and Deepgram returned nothing for them, so
  **their "So" label is itself a guess** from the earlier thread.
- **Longer training made the real score worse** while held-out TTS voices improved
  (0.88 to 0.93). That is the signature of a domain gap. Ten clips is also too few to
  rank models: one clip is 10 points.

### Unseen speakers

I trained on the 27 US and UK Kokoro voices only, for 1,500 steps, and held out all
17 accented voices (Japanese, Chinese, Spanish, French, Hindi, Italian and Portuguese
voices reading English):

- **Held-out voices:** 89–90% first-word accuracy, mixed under your mic noise and agent
  echo. That's close to the 88–93% on 3 held-out voices from the same accent groups.
- **Bugs 2:** still 6/10 at +0.4 s, deciding 6 times and wrongly 2 times.

So the model generalizes across speakers and accents of the same engine.

### Unseen engine: Deepgram Aura-2

I synthesized 952 clips in all 41 English Aura-2 voices (`synth_deepgram.py`) and mixed
them the same way.

| Trained on | Unseen test voices | First word right | Bugs 2 at +0.4 s |
|---|---|---|---|
| Kokoro only, 3,000 steps | all 41 Aura voices | **77%** | 4/10; decided 4 times, wrong 2 |
| Kokoro (3 voices held out) | 3 Kokoro voices | 95% | (same model) |
| Kokoro plus 20 Aura voices, 3,000 steps | the other 21 Aura voices | **85%** | 6/10; decided 6 times, wrong 1 |

- **A new engine costs about 18 points** where a new voice of a known engine costs
  almost nothing. Each extra engine in the training mix buys back about half of that.
- **On Bugs 2 the mixed model is the best so far.** At +0.4 s:
  - all 3 backchannels resumed correctly;
  - 2 of the 3 "So" attempts stopped correctly (one was heard as "wait", which is
    still a floor opener);
  - the one wrong action is resuming on the quietest "So (gave up)", which it heard as
    noise.

### Unseen engine and speakers: Fish Audio

The Fish clips come from `fish_synth.py`: 711 training clips in 107 voices and 237 test
clips in 27 other voices. The split is in `fish_split.json`.

| Trained on | Unseen Fish test voices (27) | Bugs 2 at +0.4 s | Bugs 2 at +0.5 s |
|---|---|---|---|
| Kokoro only | 78% | 4/10; decided 4 times, wrong 2 | |
| Kokoro + Aura (Fish never heard) | 86% | 6/10; decided 6 times, wrong 1 | 6 decided, 1 wrong |
| Kokoro + Aura + 107 Fish voices | **92%** | 6/10; decided 7 times, wrong 2 | 6 decided, 1 wrong |

- **Generalization to new engines rises with the number of engines:** 77–78% from
  one engine, 86% from two, and 92% on new voices of a third engine it has seen.
- **Bugs 2 doesn't move,** because the remaining errors aren't about the voice:
  - the quietest "So (gave up)" is still heard as noise;
  - with Fish added, "Okay. So so…" has p(more) = 0.48 at +0.4 s. That's just under the
    0.5 threshold, so the model resumes on it at +0.4 s and gets it right from +0.5 s.

This is consistent with the channel being the remaining gap (below).

### Browser-mic simulation (`mic_sim.py`)

Each synthetic clip, with probability 0.8, goes through a simulated browser mic chain:

1. A room impulse response: synthetic, or with `--banks`, one of 325 real ones.
2. A cheap-mic response: high-pass, low-pass and a resonance.
3. Agent echo through its own room path, distorted, then mostly cancelled.
4. Room noise: pink noise, or one of 586 real environmental noises at 0–30 dB SNR.
5. Double-talk suppression: −0 to −30 dB while the agent is loud.
6. Noise suppression: spectral subtraction plus a frame gate down to about −88 dB.
7. A slow AGC that starts low at the onset.

The real rooms and noises come from OpenSLR 28 (RIRS_NOISES, Apache-2.0) via
`prep_banks.py`.

**Calibration:** per 10 ms frame in the window around onset, the simulated clips are
36% near-silent (below −75 dB), with a median of −51 dB and a 90th percentile of
−23 dB. Your real overlaps are 28% near-silent, with a median of −46 dB and a 90th
percentile of −20 dB.

| Kokoro + Aura + Fish, 3,000 steps | Unseen Fish voices | Bugs 2 first word at +0.4 s | Decided / wrong at +0.4 s | Decided / wrong at +0.5 s |
|---|---|---|---|---|
| no simulation | 92% | 6/10 | 7 / 2 | 6 / 1 |
| mic simulation | 92% | **7/10** | 6 / 2 | 6 / 2 |
| mic simulation + real rooms and noises | 92% | **7/10** | 7 / 2 | 6 / 1 |

- **Simulation doesn't hurt the clean synthetic score, and it adds one first word on your
  audio:** the "Okay" in "Okay. So so…" is now right.
- **The decisions stay the same.** All three backchannels and two of the three "So"
  attempts are right. The quietest "So (gave up)" is heard as `none` or `mhm` in every
  version, and "Yeah. Could be." is heard as `um` or `none`. One clip is 10 points here,
  so these rows are not separable. A larger real eval set is now the limit.
## Why the channel matters (measured)

Per 100 ms frame, your mic during the overlaps:

- **It is often digital silence** (−85 to −90 dB) right before and after the word.
- **It drops to that silence mid-word on short words.** The browser's noise suppression
  and echo cancellation gate it while the agent plays.
- **Words are short and quiet.** The "So" attempts peak at −21 to −30 dB for about
  0.3 s.

The TTS clips are full words in clean audio, so the model never sees that gating. The
random dips in `train.py` only roughly imitate it.

**Side finding, a bug worth fixing separately:** the mic track drops samples. In
Bugs 2 it falls 1.05 s behind the session clock by 222 s, in steps (118 ms, 654 ms,
1046 ms). Anything that places mic audio by session time is off by up to a second late
in a session. That includes earlier analysis scripts and the agent channel built from
`played` events. `cut_real.load()` now maps time through every mic chunk instead.

## Next steps, in order of value

1. **Record takes inside Frankenvoice, over the agent.** This needs a small "takes" mode:
   - the agent reads a long text;
   - the UI flashes a word from `takes.txt` at random moments;
   - you say it over the agent.

   The session log then has the true channel, both tracks and the label (the prompt),
   with no Deepgram needed. About 100 takes is roughly 10 minutes of talking, and
   `cut_real.py` needs only a prompt-event reader added.
2. Re-train with those takes weighted heavily (`--p-real`), and evaluate on a fresh
   session, not Bugs 2.
3. Then try Moonshine base, or a frozen encoder plus head, if tiny plateaus. GPU (Colab)
   only matters from here on: tiny trains on this CPU in 4–15 minutes.

## Files

All in `research/bc-spotter/` on branch `claude/project-thread-5jvmuc`:

- `vocab.py`: classes and spellings.
- `cut_real.py`: word-onset windows from `.fvs` sessions.
- `eval_set.py`: the 10 labeled overlaps.
- `synth.py`: Kokoro generator; `synth_deepgram.py` and `fish_synth.py`: the same prompts in Deepgram Aura-2 and Fish voices.
- `split_takes.py` and `takes.txt`: your own recordings.
- `train.py`: training, evaluation and ONNX export (`--p-sim`, `--banks`).
- `mic_sim.py`: the browser-mic simulator; `prep_banks.py`: real rooms and noises from OpenSLR 28.

Running it:

```
python cut_real.py --exclude-overlaps-of BUGS2 real_train.npz SESSION_DIRS... BUGS2
python eval_set.py BUGS2 eval_bugs2.npz
python synth.py synth.npz --kokoro-dir KOKORO --per-prompt 6
python train.py --synth synth.npz [rin_takes.npz] --real real_train.npz --eval eval_bugs2.npz --steps 800 --export spotter.onnx
```

Session directories are unzipped `.fvs` files.

The project folder `frankenvoice/research/bc-spotter/` has a copy, plus:

- `data/`: the three `.npz` sets above;
- `ckpt-all/`: the Kokoro + Aura + Fish model (3,000 steps, `PRE = 0.5`).
- `ckpt-800/`: the 800-step model. It was trained with `PRE = 0.3`, so set that in
  `train.py` before loading it with `--load`.
