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

So the model generalizes across speakers and accents. What it doesn't cross is the
step from TTS to your browser-processed mic. Fish voices would be a stronger
unseen-speaker test, since they're a different TTS family; this session couldn't reach
the Fish API.

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
- `synth.py`: Kokoro generator.
- `split_takes.py` and `takes.txt`: your own recordings.
- `train.py`: training, evaluation and ONNX export.

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
- `ckpt-800/`: the 800-step model. It was trained with `PRE = 0.3`, so set that in
  `train.py` before loading it with `--load`.
