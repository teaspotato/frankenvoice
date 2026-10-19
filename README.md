# Frankenvoice

A research playground for full-duplex voice agents built from existing components (microphone, VAD, streaming ASR, turn-taking, LLM, TTS, speaker). The pipeline is fixed in code and iterated on by changing code. The web UI is a realtime observability view: a diagram of the pipeline with each stage's live state, and a time-aligned dashboard with one lane per stage. Every session is recorded so it can be replayed exactly.

## Quick start

```bash
pnpm install
pnpm dev               # local proxy on :8787 + Vite on :5173
```

Open **Settings** in the toolbar to enter API keys and base URLs. The proxy saves them to `.frankenvoice/settings.json` (git-ignored, readable only by you); the page never gets a key back, only whether it is set. A `.env` file (see `.env.example`) still works, and values set in Settings override it. The proxy port (`FV_PROXY_PORT`) is the one thing that has to come from the environment, since the page needs it to reach the proxy.

Open http://localhost:5173 and press **Run live**. Allow microphone access. Press **Stop** to finish. The session is saved to `./sessions/<id>.fvs` (uncheck **autosave live sessions** to keep sessions only in the tab) and appears in the Sessions list, where **Replay** runs it again.

## The UI

- **Pipeline diagram** (top): read-only, drawn to a fixed layout in `src/ui/PipelineView.tsx` (the playback feedback runs above the row so it never crosses a stage). Each stage shows its live state. While a session runs, a connection animates only while data is flowing on it, and a stage lights up while its on/off state is on (VAD hears speech, ASR mid-utterance, turn-taking thinking or speaking, LLM generating, TTS synthesizing, speaker playing, AEC double talk); see `src/ui/activity.ts`. Click a stage to configure it in the right panel.
- **Dashboard** (middle): one lane per stage on a shared time axis, live while running and the same for replays and loaded sessions.
  - *Mic*: waveform. *Cleaned*: what the noise/echo stage passes on, with how much it removed (in − out, 0–30 dB line); red ticks mark double talk (echo canceller not adapting) or RNNoise gating. *VAD*: speech probability with the start/stop thresholds; detected speech is shaded across Mic and VAD.
  - *ASR*: each result drawn over the audio it covers (dashed while partial, solid when final), with a line to the dot where it arrived, so recognition latency is visible.
  - *Turn model*: end-of-turn (blue) and take-the-floor (orange) scores per 80 ms, the user solid and the agent faint, against the thresholds (dashed); diamonds for the user's committed events, hollow dots for the agent's. Hover for the values.
  - *Turn*: floor state (thinking, speaking) and committed user turns. Below it, each time the user talked over the agent: a band for how long, ending in the verdict (red *interrupt*, amber *continue*, grey *backchannel*). Hover for the reason and what the ASR had heard.
  - *LLM*: waiting for the first token (light), then generating (solid), labelled with time to first token, chunk count and chars/s. Below it, each streamed chunk gets a dot and its text at the moment it arrived (zoom in to read them); hover a chunk for its offset and the text so far.
  - *TTS*: a diamond per request sent and each sentence's audio stream. *Playback*: the scheduled agent audio, sentence starts (diamonds), word boundaries (dots), and interruptions.
  - Scroll to zoom, drag to pan, double-click or **Fit** for the whole session, **Follow** to track the live edge. Hover for details; click a lane's name to select its stage, empty space to seek. Errors show as red dots on their stage's lane.
- **Stage panel** (right): pick a stage's implementation (for example Silero or energy VAD, API or browser TTS) and edit its params. Saved in the browser; **Save config** / **Open…** move it as a file, and **Use its config** on a session restores what that session ran with.
- **Event log** (bottom): every recorded event, filterable.
- **Hover**: hovering an item makes it glow and outlines what triggered it (amber) and what it produced (blue), following each event's recorded causes; the user audio behind it is shaded on the Mic and Cleaned lanes, and VAD/ASR items on the same audio get a dashed outline. The tooltip lists both sides. Hover a lane's name (ⓘ) for what its marks mean. With the event log open, clicking an item scrolls to and selects its first event and marks the rest, clearing filters that would hide it.
- **Transport** (dashboard header): ▶ replays the shown session, ❚❚ pauses a realtime replay (the session clock and any audio already scheduled freeze) and resumes it, ■ stops. Space does the same as ▶ / ❚❚. Click empty timeline space (or the time axis) to seek: a running replay jumps there, skipping ahead with audio muted or restarting if the point is earlier; when nothing runs it sets where ▶ starts (blue marker). Click a lane's name to select its stage.
- **Layout**: drag the edges between the diagram, dashboard, event log and side panel to resize them (double-click an edge to reset); **Hide panel** in the toolbar collapses the side panel (selecting a stage brings it back); the event log starts collapsed, click **Events** to open it. Sizes are remembered in the browser.
- **Sessions** (right): **New session** clears the dashboard; click a session's name to rename it (a saved session's file is renamed too); **Save** writes an unsaved live session to `./sessions` when autosave is off; **Delete** removes a session from the list and, for a recorded session, its file in `./sessions`.

The pipeline runs in the browser. The small proxy in `server/` keeps API keys out of the page, avoids CORS, and stores settings and session bundles. It only answers requests from the local UI: other websites (by `Origin`) and DNS-rebinding hosts (by `Host`) get 403. It forwards requests and streams responses unchanged, so the browser records exactly what each provider sent.

## Pipeline

```
                       ┌─> vad (Silero) ──┐
mic ─> noise/echo ─────┼─> turn model ────┼─> turn ─> llm ─> tts ─> speaker
          ^            └─> asr (streaming)┘    ^ │                   │  │
          │                     ^              │ └ cancel ─> llm, tts, speaker
          │                     │              └──── playback ───────┘  │
          └─────────────────────┴─ what was played (echo reference, agent audio)
```

The wiring lives in `src/pipeline.ts` (`STAGES` and `buildGraph`). Each stage lists the node types that can fill it; the UI only chooses among those and sets their params. To change the topology, edit that file.

## Noise / echo cancellation

The stage between the mic and VAD/ASR has four implementations, each with its knobs in the stage panel:

| implementation | what it does | knobs |
| --- | --- | --- |
| `browser` (default) | WebRTC echo cancellation, noise suppression and gain control, applied by the browser at capture | each of the three on/off |
| `aec` | NLMS echo canceller in the page. Its reference is the audio the speaker played, placed on the session clock. Browser processing is off, so the Mic lane is raw and the Cleaned lane is its output | filter length, step size, reference delay, double-talk threshold, residual suppression, adapt on/off |
| `rnnoise` | RNNoise neural noise suppression (WASM, 48 kHz internally); also reports a voice probability | wet/dry mix, gate below a voice probability, gate depth, browser AEC in front |
| `off` | raw microphone, no processing | none |

The echo canceller is deterministic, so a replay with changed knobs (**replay with current config**) shows what they would have done to the same recorded audio. RNNoise's output is recorded, and its mix and gate are applied after the recording, so those two can also be retuned on a replay.

## Providers

ASR, LLM and TTS each have a **Provider** param:

| node | providers | setting (Settings panel or env) |
| --- | --- | --- |
| `asr` | `deepgram` | `DEEPGRAM_API_KEY` |
| `llm` | `anthropic` | `ANTHROPIC_API_KEY` |
| | `openai-compatible`, endpoint `openrouter` / `openai` / `custom` | `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `OPENAI_COMPAT_BASE_URL` (+ optional key) |
| `tts` | `cartesia` | `CARTESIA_API_KEY` |
| | `deepgram` (Aura, `/v1/speak`, 16-bit PCM; default voice `aura-2-thalia-en`) | `DEEPGRAM_API_KEY` |
| | `openai-compatible` (`/audio/speech`, PCM), endpoint `openai` / `custom` | as above |
| `tts.browser` | the browser's built-in `speechSynthesis` | none |
| `opener.choice`, `backchannel` (*decide* `typesafe`) | TypeSafe Choice (`/v1/systemone`, model `jev-latest`) | `TYPESAFE_API_KEY` |

For OpenRouter, set the LLM stage provider to `openai-compatible`, endpoint `openrouter`, and a model slug. The toolbar shows which keys the current config still needs; click it to open Settings.

Each provider is a small pure adapter (`src/nodes/{asr,llm,tts}/`): it builds the request and parses the raw response payloads into the node's common messages. The network code lives in a driver per channel (`src/drivers/index.ts`). Recordings keep the provider's raw payloads, so replay feeds the same bytes through the same adapter. Adding a provider means one adapter, plus a driver and proxy route if it speaks a new protocol.

Turn-taking is full-duplex by default (`turn.duplex` in `src/nodes/turn.ts`); `half-duplex` is still selectable and ignores the user while the agent talks. The full-duplex stage is an explicit state machine (`FLOOR` in `src/nodes/turn.ts`): listening, thinking, speaking, and thinking+user / speaking+user while the user talks over the agent. Every transition is emitted as a `turn.transition` message with what triggered it.

### Turn model

The *Turn model* stage (`turnmodel.ssl` in `src/nodes/turn-model.ts`) runs the ssl_turn model from the [turn-playground Space](https://huggingface.co/spaces/shinohararin/turn-playground) in a worker (WebGPU, or WASM on the CPU). Its two channels are the user (cleaned mic) and the agent (what the speaker played, placed on the same clock), and every 80 ms it scores both for *end of turn* and *take the floor*. The thresholds (TurnBench dev operating points 0.64 / 0.18, 0.5 s refractory) are applied in the node, on the recorded scores, so they can be retuned on a replay. The model files (213 MB) download from the Space and compile as soon as the page opens (or the stage's model settings change), with a progress bar on the Turn model stage; the browser caches the files, and the compiled model is kept across sessions, so a session starts with it ready; *Model files URL* points elsewhere (for example a retrained export). The *Turn model* lane shows both scores for both speakers against the thresholds, and the events committed.

While the model runs, turn-taking uses it: the user's *take the floor* stops the agent at once, user speech over the agent that ends without one is a backchannel (whatever the words), and the user's *end of turn* commits the turn, waiting up to *wait for ASR* (600 ms) for the final words. *Max backchannel* still applies as a safety net, and so does ASR `utterance_end`. Until the model is ready, or if it fails or falls behind real time (it gives up 3.2 s behind), the rules below apply.

To try the model on a recorded session, tick **replay with current config**, put `turn-model` in **live channels**, and replay in realtime (a fast replay feeds it faster than it can run).

Without the model, the stage commits a user turn on ASR endpointing. When VAD hears the user while the agent is thinking or speaking, it waits *min speech to interrupt* (400 ms), then decides:

- **backchannel**: the speech ended first, or the ASR so far is only *backchannel words* (yeah, okay, mhm, …) and it is shorter than *max backchannel* (1200 ms). The agent keeps going.
- **interrupt**: the agent was speaking. It sends `cancel` to the LLM, TTS and speaker. The speaker reports how much of the reply was heard (whole sentences played, plus the words of the current one in proportion to its playback), and the LLM's history keeps only that, ending in "—".
- **continue**: the agent was still thinking (the user wasn't done). The reply is cancelled, and the previous turn is resent together with what the user says now (`turn.user` with `replaces`), replacing it in the LLM history.

The speaker no longer reports the end of playback during a gap between sentences; it waits until the TTS reply is complete.

`tts.browser` (TTS implementation `browser`) replaces both `tts` and `speaker`; the pipeline drops the speaker stage when it is selected. The browser plays the audio itself, so no samples are recorded. What is recorded is when each sentence started, each word boundary the browser reported, and when it ended, so replay reproduces the playback timing and audible replay speaks the text again. Voices differ between browsers and machines, and headless browsers usually have none.

`llm.echo` and `tts.tone` (implementations `echo` and `tone`) are mocks for trying the pipeline without API keys.

## Time, logging and replay

The core rule is that **everything is replayable as-is**: ASR partials, LLM tokens and TTS audio come back exactly as they first arrived, at the same moments.

How the runtime ensures that:

- **One session clock.** Every event has `t`, its arrival time in ms on a monotonic session clock. Events about audio also carry `media: {track, start, end}`, a sample range on a recorded media track. So each event records both *when* it happened and *which audio* it refers to. The mic driver also logs the AudioContext time of every captured frame.
- **Everything external goes through recorded IO.** Nodes never call `fetch`, `setTimeout` or `Date.now`. They open *exchanges* on named channels (`mic`, `rnnoise`, `silero`, `deepgram`, `anthropic`, `openai`, `cartesia`, `openai-tts`, `deepgram-tts`, `speech-synthesis`, `speaker`). Every message coming back is an `io.recv` event with its payload and raw bytes, in the order and with the chunking it originally had. Timers are recorded the same way (`timer.set` / `timer.fire`).
- **Deterministic turns.** When an external event arrives, the runtime stamps it with the current time, hands it to its node, and drains every resulting message synchronously, all at that timestamp. Nodes are pure functions of what they receive, so the same external events at the same times always produce the same log.
- **Replay on a virtual clock.** A replay serves every recorded exchange and timer from the trajectory at its original time, with the original event order breaking ties. The result is checked against the source: the Sessions list shows *identical to source ✓* or the first event where the two differ.
- **Causality.** Every event lists the seqs of the events that caused it, so any output (a TTS chunk, say) can be traced back through the LLM tokens to the ASR result and the audio frames behind it.

Replay options:

- **real time / audible** paces the virtual clock against the wall clock and plays both sides again: the recorded mic, and the agent audio (output channels such as `speaker` and `speech-synthesis` run live as monitors). The 🔊 buttons on the Mic and Playback lanes mute either side.
- **replay with current config** runs the current stage config against the recorded world. Upstream nodes reproduce exactly; any exchange whose request differs from the recording is flagged as an `io.diverge` event.
- **live channels** (for example `anthropic`) serves those channels live during replay, so the same recorded user audio can be run against a new LLM or TTS.

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

The browser side of a channel is a `LiveDriver` in `src/drivers/index.ts`. Register the node in `src/nodes/index.ts` and list it as a variant of its stage in `src/pipeline.ts`. To give it its own dashboard drawing, derive the data in `src/ui/timeline.ts` and draw it in `src/ui/Dashboard.tsx`.

## Layout

```
src/core/     runtime, scheduler (live + virtual clock), world (live/replay IO), trajectory, bundle, compare
src/pipeline.ts  the fixed pipeline: stages, their implementations, wiring
src/nodes/    node definitions
src/drivers/  browser drivers: mic worklet, Silero (onnxruntime-web), Deepgram WS, Claude SDK, OpenAI-compatible SSE, TTS byte streams, Web Audio
src/ui/       pipeline diagram, dashboard (timeline.ts derives lanes, Dashboard.tsx draws them), stage panel, settings, sessions, event log
server/       local proxy and session storage
tests/        replay determinism tests with randomized fake drivers
```

## Roadmap

1. ~~Runtime with trajectory logging and a basic voice loop~~; fixed pipeline with a live per-stage dashboard, scrubbing and run comparison
2. ~~Full-duplex floor control: barge-in, with history cut back to what was heard; backchannels; every decision marked on the dashboard with its reason~~ (first version; tune on real sessions)
3. Tool calls mid-conversation, with filler speech and interruptible calls
4. Local models on a GPU box (VAD, ASR, LLM, TTS), with their internals visualized
