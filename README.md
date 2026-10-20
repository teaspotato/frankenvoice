# Frankenvoice

A research playground for full-duplex voice agents built from existing components (microphone, VAD, streaming ASR, turn-taking, LLM, TTS, speaker). The pipeline is fixed in code and iterated on by changing code. The web UI is a realtime observability view: a diagram of the pipeline with each stage's live state, and a time-aligned dashboard with one lane per stage. Every session is recorded so it can be replayed exactly.

## Quick start

```bash
pnpm install
pnpm dev               # server (pipeline) on :8787 + Vite on :5173
```

Everything but the TypeSafe stages runs on your own GPU box (`gpubox/README.md`): Qwen3-ASR, Gemma on llama.cpp and Kyutai TTS. Open **Settings** in the toolbar to enter its URL and token, the LLM endpoint and the TypeSafe key. The server saves them to `.frankenvoice/settings.json` (git-ignored, readable only by you); the page never gets a key back, only whether it is set. A `.env` file (see `.env.example`) still works, and values set in Settings override it. The server port (`FV_PROXY_PORT`) is the one thing that has to come from the environment, since the page needs it to reach the server.

Open http://localhost:5173 and press **Run live**. Allow microphone access. Press **Stop** to finish. The session is saved to `./sessions/<id>.fvs` (uncheck **autosave live sessions** to keep sessions only in the tab) and appears in the Sessions list, where **Replay** runs it again.

## The UI

- **Pipeline diagram** (top): read-only, drawn to a fixed layout in `src/ui/PipelineView.tsx` (the playback feedback runs above the row so it never crosses a stage). Each stage shows its live state. While a session runs, a connection animates only while data is flowing on it, and a stage lights up while its on/off state is on (VAD hears speech, ASR mid-utterance, turn-taking thinking or speaking, LLM generating, TTS synthesizing, speaker playing, AEC double talk); see `src/ui/activity.ts`. Click a stage to configure it in the right panel.
- **Dashboard** (middle): one lane per stage on a shared time axis, live while running and the same for replays and loaded sessions.
  - *Mic*: waveform. *Cleaned*: what the noise/echo stage passes on, with how much it removed (in − out, 0–30 dB line); red ticks mark double talk (echo canceller not adapting) or RNNoise gating. *VAD*: speech probability with the start/stop thresholds; detected speech is shaded across Mic and VAD.
  - *ASR*: each result drawn over the audio it covers (dashed while partial, solid when final), with a line to the dot where it arrived, so recognition latency is visible.
  - *Turn model*: end-of-turn (blue) and take-the-floor (orange) scores per 80 ms, the user solid and the agent faint, against the thresholds (dashed); diamonds for the user's committed events, hollow dots for the agent's. Hover for the values.
  - *Turn*: floor state (thinking, speaking) and committed user turns. Below it, each time the user talked over the agent: a band for how long, ending in the verdict (red *interrupt*, amber *continue*, grey *backchannel*). Hover for the reason and what the ASR had heard.
  - *LLM*: waiting for the first token (light), then generating (solid), labelled with time to first token, chunk count and chars/s. Below it, each streamed chunk gets a dot and its text at the moment it arrived (zoom in to read them); hover a chunk for its offset and the text so far.
  - *TTS*: a diamond per request sent and each sentence's audio stream. *Playback*: the scheduled agent audio, sentence starts (diamonds), and interruptions.
  - Scroll to zoom, drag to pan, double-click or **Fit** for the whole session, **Follow** to track the live edge. Hover for details; click a lane's name to select its stage, empty space to seek. Errors show as red dots on their stage's lane.
- **Stage panel** (right): pick a stage's implementation (for example Silero or energy VAD, Kyutai or the tone mock for TTS) and edit its params. Saved in the browser; **Save config** / **Open…** move it as a file, and **Use its config** on a session restores what that session ran with.
- **Event log** (bottom): every recorded event, filterable.
- **Hover**: hovering an item makes it glow and outlines what triggered it (amber) and what it produced (blue), following each event's recorded causes; the user audio behind it is shaded on the Mic and Cleaned lanes, and VAD/ASR items on the same audio get a dashed outline. The tooltip lists both sides. Hover a lane's name (ⓘ) for what its marks mean. With the event log open, clicking an item scrolls to and selects its first event and marks the rest, clearing filters that would hide it.
- **Transport** (dashboard header): ▶ replays the shown session, ❚❚ pauses a realtime replay (the session clock and any audio already scheduled freeze) and resumes it, ■ stops. Space does the same as ▶ / ❚❚. Click empty timeline space (or the time axis) to seek: a running replay jumps there, skipping ahead with audio muted or restarting if the point is earlier; when nothing runs it sets where ▶ starts (blue marker). Click a lane's name to select its stage.
- **Layout**: drag the edges between the diagram, dashboard, event log and side panel to resize them (double-click an edge to reset); **Hide panel** in the toolbar collapses the side panel (selecting a stage brings it back); the event log starts collapsed, click **Events** to open it. Sizes are remembered in the browser.
- **Sessions** (right): **New session** clears the dashboard; click a session's name to rename it (a saved session's file is renamed too); **Save** writes an unsaved live session to `./sessions` when autosave is off; **Delete** removes a session from the list and, for a recorded session, its file in `./sessions`.

The pipeline runs in the server (`server/`, Node), next to the models and providers: VAD (onnxruntime-node), the turn model, RNNoise, the sentence encoder, and the connections to the GPU box, the LLM and TypeSafe. The page is only the mic, the speaker and the observability view. It keeps one WebSocket to the server (`/api/live`, protocol in `src/live-protocol.ts`): mic frames go up (captured with the browser's echo cancellation, noise suppression and gain as the Noise / echo stage asks, since playback happens in the page); speaker commands with their session time come down and are scheduled on an audio clock synced to the server's by pings; and the run's trajectory (events, node states, media tracks) streams down so the page keeps a mirror and draws it as it happens. When a run stops the server sends its full recording, saves it (autosave) and names it. One page drives the server at a time: a page that connects takes over and stops a run the previous page had going. The server only answers the local UI: other websites (by `Origin`) and DNS-rebinding hosts (by `Host`) get 403.

Replays of a recording run in the page (they need no providers: the recorded agent audio plays locally). A replay with **live channels** runs on the server, in real time (it can pause, not seek).

To host the UI somewhere else (for example on the GPU box, behind a tunnel), build it with `pnpm build` and start the server with `FV_STATIC_DIR=dist` and `FV_HOST_TOKEN=<secret>`. It then serves the UI too, and instead of local-only access every request must be same-origin and carry the token: open the page once with `?token=<secret>`, which sets an HttpOnly cookie. Anyone with the token can change settings, so treat it like a password.

## Pipeline

```
                       ┌─> vad (Silero) ──┐
mic ─> noise/echo ─────┼─> turn model ────┼─> turn ─> llm ─> tts ─> speaker
          ^            └─> asr (streaming)┘    ^ │                   │  │
          │                     ^              │ └ cancel ─> llm, tts, speaker
          │                     │              └──── playback ───────┘  │
          └─────────────────────┴─ what was played (echo reference, agent audio)
```

The *Scheduler* stage (below, not drawn in the sketch) watches the reply in flight and paces the LLM.

The wiring lives in `src/pipeline.ts` (`STAGES` and `buildGraph`). Each stage lists the node types that can fill it; the UI only chooses among those and sets their params. To change the topology, edit that file.

## Noise / echo cancellation

The stage between the mic and VAD/ASR has four implementations, each with its knobs in the stage panel:

| implementation | what it does | knobs |
| --- | --- | --- |
| `browser` (default) | WebRTC echo cancellation, noise suppression and gain control, applied by the browser at capture | each of the three on/off |
| `aec` | NLMS echo canceller (a node, in the server). Its reference is the audio the speaker played, placed on the session clock. Browser processing is off, so the Mic lane is raw and the Cleaned lane is its output | filter length, step size, reference delay, double-talk threshold, residual suppression, adapt on/off |
| `rnnoise` | RNNoise neural noise suppression (WASM in the server, 48 kHz internally); also reports a voice probability | wet/dry mix, gate below a voice probability, gate depth, browser AEC in front |
| `off` | raw microphone, no processing | none |

The echo canceller is deterministic, so a replay with changed knobs (**replay with current config**) shows what they would have done to the same recorded audio. RNNoise's output is recorded, and its mix and gate are applied after the recording, so those two can also be retuned on a replay.

## Services

| stage | runs on | setting (Settings panel or env) |
| --- | --- | --- |
| `asr` | Qwen3-ASR-1.7B on the GPU box (`/v1/asr`) | `GPUBOX_BASE_URL`, `GPUBOX_TOKEN` |
| `llm` | an OpenAI-compatible chat endpoint: llama.cpp with Gemma 4 12B on the GPU box (model `gemma-4-12b-it`), or any local server | `OPENAI_COMPAT_BASE_URL` (+ optional `OPENAI_COMPAT_API_KEY`) |
| `tts.kyutai` | Kyutai TTS on the GPU box (`/v1/tts`): words stream in as the LLM writes them, word timings come back. Canned lines (backchannels, openers) render once per box: the server keeps their audio (`fv-tts-cache` next to the settings file, or `FV_TTS_CACHE_DIR`), keyed by the text, voice and the box's TTS settings, and renders a missing one as a background job the box drops whenever a reply needs it | `GPUBOX_BASE_URL`, `GPUBOX_TOKEN` |
| `opener.choice`, `backchannel` (*decide* `typesafe`), `interject` | TypeSafe Choice (`/v1/systemone`, model `jev-latest`) | `TYPESAFE_API_KEY` |

Backchannels are written by Gemma at the moment the stage decides to say one (*say* `generated`, the default; `canned` plays lines rendered ahead). The request is the reply's own (same system prompt and history, so llama.cpp reuses the cached prefix), then the user's words so far with a short instruction as the last user turn, and an assistant prefill `[i0g0] ` (the utterance tag: not interruptible, not gated) so the model writes only the words, under a GBNF grammar of at most *max words* words and no question. It runs on llama.cpp slot 1 (`id_slot`; replies and drafts run on slot 0), so it never evicts the reply's cache, and nothing of it enters the LLM's history. Kyutai speaks it on a socket of its own, opened at the decision, and the audio plays as it arrives on the speaker's cue output, which nothing the user does stops. A line that is late (*wait for the LLM*, 500 ms) or isn't a backchannel is replaced by a canned fallback; one whose audio hasn't started when the user's turn ends is dropped.

The toolbar shows which settings the current config still needs; click it to open Settings.

The GPU box protocols are in `gpubox/PROTOCOL.md`. ASR and the LLM each have a small pure adapter (`src/nodes/asr/gpubox.ts`, `src/nodes/llm/openai.ts`): it builds the request and parses the raw response payloads into the node's common messages. The network code lives in a driver per channel (`server/drivers/index.ts`). Recordings keep the raw payloads, so replay feeds the same bytes through the same adapter.

Turn-taking is full-duplex by default (`turn.duplex` in `src/nodes/turn.ts`); `half-duplex` is still selectable and ignores the user while the agent talks. The full-duplex stage is an explicit state machine (`FLOOR` in `src/nodes/turn.ts`): listening, thinking, speaking, and thinking+user / speaking+user while the user talks over the agent. Every transition is emitted as a `turn.transition` message with what triggered it.

### Tick scheduler

The *Scheduler* stage (`sched` in `src/nodes/sched.ts`) gives the pipeline a view of the GPU as 80 ms frames, one Kyutai frame each: TTS frame production comes first, LLM tokens and ASR partials go in the slack. While a reply is in flight (from `llm.start` until it is written, spoken and played) it ticks on an 80 ms grid from the reply's start, and logs every tick as a `sched.tick` message:

| field | what |
| --- | --- |
| `n`, `rid`, `ms` | tick number, the reply, ms since the reply started |
| `tts` | `audioMs` produced and `words` started during the tick, `queuedWords` (words Kyutai has been sent and not yet read: its `Consumed` count, plus words sent since), `bufferedMs` (how far the reply's audio is scheduled ahead of playback), `done` |
| `llm` | `tokens` arrived during the tick, `pending` (granted and not arrived), `decision` (`grant`, `hold`, `busy`, `done`, `free`) and `why`, `granted` |
| `asr` | `partials`, `finals` arrived during the tick |

The backends are separate processes, so what the scheduler governs is when the LLM may write. With *Pace the LLM* on, the LLM writes a reply in chunks: its first request (a draft's too) asks for *first burst* tokens (16); a request that stops at its limit leaves the reply open (`llm.chunk`), and each `sched.grant` continues it with *chunk* tokens (8): the same messages plus the text so far as the assistant's turn, which llama.cpp continues, prefilling a token or two thanks to its prompt cache. A grant goes out on a tick when TTS has fewer than *lookahead* words (3) to read, or playback is about to run dry; before the first audio, otherwise nothing more until that audio is out (*hold until first audio*); after it, while TTS has fewer than *max words queued* (6) and its audio runs at least *min audio ahead* (300 ms) ahead of playback. Drafts are capped at the first burst and paced once kept. Off, the LLM runs free and the ticks are only logged. Grants and ticks are ordinary messages driven by recorded IO and timers, so they replay exactly; a future engine running the models in one process would take the grants over its own channel and report what it ran per tick, without changing the pipeline or replay.

### Turn model

The *Turn model* stage (`turnmodel.ssl` in `src/nodes/turn-model.ts`) runs the ssl_turn model from the [turn-playground Space](https://huggingface.co/spaces/shinohararin/turn-playground) on the server's CPU (onnxruntime-node). Its two channels are the user (cleaned mic) and the agent (what the speaker played, placed on the same clock), and every 80 ms it scores both for *end of turn* and *take the floor*. The thresholds (TurnBench dev operating points 0.64 / 0.18, 0.5 s refractory) are applied in the node, on the recorded scores, so they can be retuned on a replay. The model files (213 MB) download from the Space and compile as soon as the stage is picked (or its model settings change), with a progress bar on the Turn model stage; the server caches the files (`fv-models` next to its settings file), and the compiled model is kept across sessions, so a session starts with it ready; *Model files URL* points elsewhere (for example a retrained export). The *Turn model* lane shows both scores for both speakers against the thresholds, and the events committed.

While the model runs, turn-taking uses it: the user's *take the floor* stops the agent at once, user speech over the agent that ends without one is a backchannel (whatever the words), and the user's *end of turn* commits the turn, waiting up to *wait for ASR* (600 ms) for the final words. *Max backchannel* still applies as a safety net, and so does ASR `utterance_end`. Until the model is ready, or if it fails or falls behind real time (it gives up 3.2 s behind), the rules below apply.

To try the model on a recorded session, tick **replay with current config**, put `turn-model` in **live channels**, and replay in realtime (a fast replay feeds it faster than it can run).

Without the model, the stage commits a user turn on ASR endpointing. When VAD hears the user while the agent is thinking or speaking, it waits *min speech to interrupt* (400 ms), then decides:

- **backchannel**: the speech ended first, or the ASR so far is only *backchannel words* (yeah, okay, mhm, …) and it is shorter than *max backchannel* (1200 ms). The agent keeps going.
- **interrupt**: the agent was speaking. It sends `cancel` to the LLM, TTS and speaker. The speaker reports how much of the reply was heard (whole sentences played, plus the words of the current one in proportion to its playback), and the LLM's history keeps only that, ending in "—".
- **continue**: the agent was still thinking (the user wasn't done). The reply is cancelled, and the previous turn is resent together with what the user says now (`turn.user` with `replaces`), replacing it in the LLM history.

**Yield if it just started** (`yieldWithinMs`, 1000 ms): speech within that long of the agent starting a reply stops it at once and gives the user the floor. With **Stop in the page** (`pageGate`, on by default) the stop itself doesn't wait for the network: at the start of a reply turn-taking arms the page (control `gate` with the window's end, passed on as a speaker command; disarmed when the agent stops holding the floor alone), and the page runs its own FireRedVAD (onnxruntime-web, loaded only when the option is on, same model and segmenter defaults as `vad.firered`) on the mic it captures. When it hears the user start while armed, it fades the reply out at once, drops the reply audio it is sent until the server acknowledges with a flush, and reports the stop (session time from the clock sync, and the mic sample where the speech started) as a speaker recv, so a replay reproduces it. The speaker takes that as a pause at the reported time (`playback.paused` with `gate`; what was heard is counted up to there). Turn-taking still leaves the decision to the VAD here: if it hears speech within *confirm within* (600 ms), that is the quick yield (cancel, the floor to the user, counted toward backing off only if a word was heard); if it hears none, the page heard noise or echo, the decision is `noise` and the agent plays on from the start of the word it stopped in. If the VAD here heard the speech first but past the window, the page's earlier onset still makes it a quick yield. The page gate only runs on 16 kHz capture with browser echo cancellation (otherwise it would hear the agent). The Playback lane marks each page stop with a red dot.

The speaker no longer reports the end of playback during a gap between sentences; it waits until the TTS reply is complete.

`llm.echo` and `tts.tone` (implementations `echo` and `tone`) are mocks for trying the pipeline without the GPU box.

## Time, logging and replay

The core rule is that **everything is replayable as-is**: ASR partials, LLM tokens and TTS audio come back exactly as they first arrived, at the same moments.

How the runtime ensures that:

- **One session clock.** Every event has `t`, its arrival time in ms on a monotonic session clock. Events about audio also carry `media: {track, start, end}`, a sample range on a recorded media track. So each event records both *when* it happened and *which audio* it refers to. The mic driver also logs the AudioContext time of every captured frame.
- **Everything external goes through recorded IO.** Nodes never call `fetch`, `setTimeout` or `Date.now`. They open *exchanges* on named channels (`mic`, `rnnoise`, `silero`, `gpubox-asr`, `gpubox-tts`, `tts-canned`, `openai`, `typesafe`, `speaker`). Every message coming back is an `io.recv` event with its payload and raw bytes, in the order and with the chunking it originally had. A canned line served from the server's cache is such an exchange too, so a replay gets it from the recording, not the cache. Timers are recorded the same way (`timer.set` / `timer.fire`).
- **Deterministic turns.** When an external event arrives, the runtime stamps it with the current time, hands it to its node, and drains every resulting message synchronously, all at that timestamp. Nodes are pure functions of what they receive, so the same external events at the same times always produce the same log.
- **Replay on a virtual clock.** A replay serves every recorded exchange and timer from the trajectory at its original time, with the original event order breaking ties. The result is checked against the source: the Sessions list shows *identical to source ✓* or the first event where the two differ.
- **Causality.** Every event lists the seqs of the events that caused it, so any output (a TTS chunk, say) can be traced back through the LLM tokens to the ASR result and the audio frames behind it.

Replay options:

- **real time / audible** paces the virtual clock against the wall clock and plays both sides again: the recorded mic, and the agent audio (output channels such as `speaker` run live as monitors). The 🔊 buttons on the Mic and Playback lanes mute either side.
- **replay with current config** runs the current stage config against the recorded world. Upstream nodes reproduce exactly; any exchange whose request differs from the recording is flagged as an `io.diverge` event.
- **live channels** (for example `openai`) serves those channels live during replay, so the same recorded user audio can be run against a new LLM or TTS.

### Session bundle format (`.fvs`, a zip)

| file | contents |
| --- | --- |
| `meta.json` | session id, mode, `replayOf`, the full graph and its hash, channel modes, track table |
| `events.jsonl` | one event per line: `{seq, t, node, kind, port?, data?, media?, bin?, cause?}` |
| `tracks/NNN.bin` | raw track bytes: float32 LE PCM for media tracks (`mic.audio`, `tts.audio`), payload bytes for IO tracks (`io/<node>/<channel>`) |

Event kinds: `session.*`, `node.*`, `msg` (a node emitted on a port), `state` (internal state change), `log`, `timer.*`, `io.open|send|recv|close|diverge`.

## Writing a node

A stage is filled by a node: a `NodeDef` (see `src/nodes/*.ts`): ports, params, the IO channels it uses, and `create(ctx)`, which returns `{start, onInput, stop}`. Rules that keep replay exact:

1. Touch the outside world only through `ctx.io.open(channel, request, handlers)`. Put everything that determines the response into `request`, since it is recorded and compared on replay.
2. Delay work only with `ctx.setTimer`, and read time only with `ctx.now()`.
3. Keep all state inside the node and expose what matters with `ctx.setState` so it lands in the log.
4. Append produced audio with `ctx.media.append` and emit `{type: 'audio', ...ref, samples}`.

The live side of a channel is a `LiveDriver` in `server/drivers/index.ts` (the mic and speaker are the page's, in `server/live.ts`). Register the node in `src/nodes/index.ts` and list it as a variant of its stage in `src/pipeline.ts`. To give it its own dashboard drawing, derive the data in `src/ui/timeline.ts` and draw it in `src/ui/Dashboard.tsx`.

## Layout

```
src/core/     runtime, scheduler (live + virtual clock), world (live/replay IO), trajectory, bundle, compare
src/pipeline.ts  the fixed pipeline: stages, their implementations, wiring
src/nodes/    node definitions
src/client/   the page's end of /api/live: mic capture (worklet), Web Audio playback, clock sync, trajectory mirror
src/ui/       pipeline diagram, dashboard (timeline.ts derives lanes, Dashboard.tsx draws them), stage panel, settings, sessions, event log
server/       the server: runs the pipeline for the page (live.ts), drivers (Silero/FireRedVAD/turn model on
              onnxruntime-node, RNNoise, GPU box WebSockets, OpenAI-compatible SSE, TypeSafe), settings, session storage
tests/        replay determinism tests with randomized fake drivers
```

## Roadmap

1. ~~Runtime with trajectory logging and a basic voice loop~~; fixed pipeline with a live per-stage dashboard, scrubbing and run comparison
2. ~~Full-duplex floor control: barge-in, with history cut back to what was heard; backchannels; every decision marked on the dashboard with its reason~~ (first version; tune on real sessions)
3. Tool calls mid-conversation, with filler speech and interruptible calls
4. Local models on a GPU box (VAD, ASR, LLM, TTS), with their internals visualized
