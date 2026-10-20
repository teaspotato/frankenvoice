# GPU box

One authenticated HTTP/WebSocket server for Frankenvoice: streaming ASR (Qwen3-ASR-1.7B), streaming
TTS (Kyutai `tts-1.6b-en_fr`) and a streaming chat LLM (Gemma 4 12B, OpenAI-compatible). The ASR/TTS wire protocol
is in [PROTOCOL.md](PROTOCOL.md); the LLM is plain OpenAI `POST /v1/chat/completions` (`stream: true`) and
`GET /v1/models`, behind the same bearer token.

## Processes

| Process | Venv | Port | What it does |
|---|---|---|---|
| `asr` | `v-asr` (sglang-omni 0.1.8) | 127.0.0.1:8101 | `sgl-omni serve` Qwen3-ASR-1.7B (or `ASR_MODEL`) with FP8 weights, `mem_fraction_static 0.15`, `max_running_requests 8` |
| `llm` | none (prebuilt llama.cpp `b11753`, CUDA) | 127.0.0.1:8103 | `llama-server`: `google/gemma-4-12B-it-qat-q4_0-gguf` (official QAT Q4_0) + MTP draft head from `ggml-org/gemma-4-12B-it-GGUF`, 2 slots x 4k context, q8_0 KV cache with a full-size sliding-window cache (`--swa-full`, so a diverging prompt reuses its prefix), thinking off |
| `tts` | `v-tts` (moshi 0.2.13) | 127.0.0.1:8102 | `kyutai_tts.py`: Kyutai streaming generation, one reply at a time |
| `gateway` | `v-gw` (qwen-asr 0.0.6, FastAPI) | 0.0.0.0:8000 | `server.py`: auth, ASR segmenting/partials/finals, Qwen3-ForcedAligner-0.6B, relays `/v1/tts`, passes `/v1/chat/completions` and `/v1/models` through to `llm` |
| `tunnel` | none | none | `cloudflared tunnel --protocol http2 --url http://localhost:8000` (quick tunnel; over the default QUIC it moved only ~110 KB/s, random `*.trycloudflare.com`) |
| `ui` | Node 22 (`$ROOT/node`) | 127.0.0.1:8787 | The Frankenvoice server (`server/index.ts`) in hosted mode: runs the pipeline (VAD and the turn model on its CPU) against the gateway and llama.cpp over localhost, and serves the built UI from `$ROOT/frankenvoice/dist` behind the gpubox token; the page only sends the mic and plays the agent |
| `ui-tunnel` | none | none | `cloudflared tunnel --protocol http2 --url http://localhost:8787` (a second quick tunnel, printed as `UI_URL`) |

Moshi and qwen-asr need different torch/transformers versions, so TTS runs in its own venv and process.

GPU memory on a 24 GB L4 (measured, all four loaded):

| Process | GPU memory |
|---|---|
| `asr` (Qwen3-ASR-1.7B FP8, `ASR_MEM=0.15`: 2.5 GB weights, 0.6 GB KV = 5.8k tokens) | 5.1 to 5.4 GB |
| `llm` (Gemma 4 12B Q4_0: 6.6 GB weights, 1.2 GB KV, MTP head, audio/vision projector) | 9.0 GB |
| `tts` (Kyutai) | 4.5 GB |
| `gateway` (forced aligner) | 2.3 GB |
| **Total** | **about 21.2 GB of 22.5 GB usable (1.3 GB free)** |

`ASR_MEM` is a fraction of the GPU memory that is *free when ASR starts*, so `start` launches ASR first. Its old default
(0.35) reserved 9.5 GB, mostly an oversized KV pool (46k tokens). With `NO_LLM=1` the box needs about 13 GB.
`ASR_MODEL=Qwen/Qwen3-ASR-0.6B` frees about 1.3 GB more if needed (about 20% higher WER; see the ASR benchmark).

## Run

On any Ubuntu box with an NVIDIA GPU (24 GB or more), CUDA driver, `curl` and internet access:

```bash
# copy this folder to the box (e.g. /content/gpubox on Colab), then:
bash gpubox/setup.sh            # install (about 3 min) + start (about 3 min the first time)
bash gpubox/setup.sh status     # prints GPUBOX_URL and GPUBOX_TOKEN
```

Other commands: `setup.sh install | start | stop | restart | status | ui` (see [Hosted UI](#hosted-ui)). `start` only starts what isn't already
running. To restart just one process, kill its process group (`kill -- -$(cat $ROOT/tts.pid)`) and run `start` again.
The tunnel keeps its URL as long as the `tunnel` process lives.

Environment variables: `GPUBOX_ROOT` (venvs, logs, pid files; default `/content` if it exists, else `~/gpubox`),
`GPUBOX_TOKEN` (default: random, saved in `$ROOT/gpubox.token`), `ASR_MODEL` (`Qwen/Qwen3-ASR-1.7B`, or
`Qwen/Qwen3-ASR-0.6B`), `ASR_MEM` (0.15), `NO_LLM=1`, `LLM_CTX` (8192, shared by the slots), `LLM_SLOTS` (2), `LLM_SWA_FULL` (1; at 32k it needs 5.4 GB),
`LLM_MTP` (1), `NO_TUNNEL=1`, `KYUTAI_NQ` (codebooks, 32), `KYUTAI_WORD_OFFSET_S` (-0.04), `KYUTAI_IDLE_CLOSE_S` (30),
`NO_UI=1`, `FV_APP` (app archive, default `$ROOT/frankenvoice.tar.gz`).
Logs are in `$ROOT/logs/{asr,llm,tts,gateway,tunnel,ui,ui-tunnel}.log`. Model files for the LLM are in `$ROOT/models`, llama.cpp
in `$ROOT/llama.cpp`.

### Colab

1. Create an L4 runtime: `colab --auth=oauth2 new -s gpubox --gpu L4`.
2. Upload this folder: `colab --auth=oauth2 upload -s gpubox gpubox/setup.sh /content/gpubox/setup.sh` (same for
   `server.py` and `kyutai_tts.py`).
3. Run `bash /content/gpubox/setup.sh` **detached**: a `colab exec` call times out long before it finishes. Use
   `subprocess.Popen(['bash', '/content/gpubox/setup.sh'], start_new_session=True, stdout=open('/content/setup.log', 'w'), stderr=subprocess.STDOUT)`.
4. `tail /content/setup.log` until it prints `GPUBOX_URL=...` and `GPUBOX_TOKEN=...`.

Colab notes: Colab sets `UV_SYSTEM_PYTHON=true`; `setup.sh` unsets it and installs into venvs. llama.cpp finds
the driver's `libcuda` through Colab's `LD_LIBRARY_PATH` (`/usr/lib64-nvidia`), which `setup.sh` keeps.
`colab exec` has a 30 s default timeout and can wedge the kernel; `colab ssh` (one connection at a time, so use SSH
`ControlMaster`) is more reliable for long jobs. Don't use `pkill -f`
there, because it can match your own shell; kill by pid. The VM (and so the tunnel URL) goes away when the
runtime is recycled.

## Hosted UI

`setup.sh` can also serve the Frankenvoice web UI from the box, so the whole thing is one https URL and the
server, which runs the pipeline, reaches ASR, TTS and the LLM over localhost (no keys or tunnels on your laptop).

1. Upload the app (the repo is private, so the box can't clone it):
   `git archive --format=tar.gz -o frankenvoice.tar.gz HEAD`, then
   `colab --auth=oauth2 upload -s gpubox frankenvoice.tar.gz /content/frankenvoice.tar.gz` (or `scp` over the
   `colab ssh` connection). A `dist/` inside the archive is used as is; otherwise the box builds it.
2. `bash /content/gpubox/setup.sh ui` installs Node 22 and pnpm if missing (into `$ROOT/node`), unpacks the archive
   to `$ROOT/frankenvoice` (keeping `node_modules`), runs `pnpm install` and `pnpm build`, and starts only `ui` and
   `ui-tunnel` (about 15 s). A full `setup.sh` / `start` does the same when the archive is there, and `stop` stops
   them too. To pick up a new archive, kill `ui` (`kill -- -$(cat $ROOT/ui.pid)`) and run `setup.sh ui` again;
   `ui-tunnel` and its URL stay.
3. `setup.sh status` prints `UI_URL`. Open **`UI_URL/?token=<GPUBOX_TOKEN>`** once: the server sets an HttpOnly cookie
   (30 days) and redirects to `UI_URL/`. Without it every page and API call gets 403. Anyone with the token can
   change settings, so treat it like a password.

Preset from the environment (shown as `env` in Settings):

| Setting | Value |
|---|---|
| `GPUBOX_BASE_URL`, `GPUBOX_TOKEN` | `ws://127.0.0.1:8000` and the gpubox token: the GPU box ASR/TTS stages work as is |
| `OPENAI_COMPAT_BASE_URL` | `http://127.0.0.1:8103/v1` (llama.cpp directly; it needs no key): the LLM stage's default model `gemma-4-12b-it` works as is |
| `FV_SESSIONS_DIR` | `$ROOT/fv-sessions` |

Other keys (TypeSafe) are entered in the UI's Settings on the box and saved in
`$ROOT/fv-settings.json` (mode 600), which overrides the environment. The token itself is never written to the
app folder or the settings file; `setup.sh` reads it from `$ROOT/gpubox.token` and passes it in the environment.
`NO_UI=1` skips all of this. Log: `$ROOT/logs/ui.log`, install log `$ROOT/logs/install-ui.log`.

## Point Frankenvoice at it

Set the local server's GPU box URL to the `GPUBOX_URL` and its token to `GPUBOX_TOKEN`. Check it with
`curl -H "Authorization: Bearer $GPUBOX_TOKEN" $GPUBOX_URL/health`.

## Behavior notes

- **LLM**: Gemma 4 12B instruct, thinking off (`--reasoning off`; a request can turn it back on with
  `"chat_template_kwargs": {"enable_thinking": true}`, and the thoughts then arrive in `reasoning_content`). Use the
  model name `gemma-4-12b-it`; recommended sampling is temperature 1.0, top_p 0.95, top_k 64 (lower temperature is fine
  for a voice agent). Keep the system prompt and history byte-identical between turns so the prefix cache hits: with
  a 1.5k-token system prompt, time to first token is about 70 to 110 ms on a cache hit and about 0.9 s cold. Editing
  only the last ~500 tokens (for example a reply cut by barge-in) still hits the cache, because the sliding-window KV
  cache can roll back that far; context checkpoints are off because they cost about 400 ms per request.
  Decoding is about 60 tok/s for one stream (MTP speculative decoding roughly doubles it) and about 30 tok/s per
  stream with 4 streams. Closing the HTTP connection stops generation. The gateway's `/health` reports the LLM name.
- **Sharing the GPU**: while the LLM decodes, Kyutai TTS slows to about 1.0 to 1.1x realtime (from 0.7x) and its
  first audio arrives about 250 ms later; ASR finals take about 150 ms longer. See
  `gpubox-test/gemma-llm.md` in the project files.

- **ASR partials** re-decode the whole segment about every `partial_ms` of new audio. A partial runs only when no
  decode is in flight and new voiced audio has arrived since the last one.
- **ASR finals**: the server waits for any in-flight partial, decodes the segment, then times its words with the
  forced aligner. The final carries an extra `align_ms` field. A Finalize on a segment with no voiced audio returns
  an empty final right away, without decoding.
- **Energy gate**: 20 ms frames; a frame is voiced if it is more than 12 dB above the 10th-percentile level of the
  last 5 s and above -55 dBFS. For the first second of a connection, the floor is capped at -70 dBFS.
- **TTS text feeding**: words are queued as soon as they are complete (whitespace, Flush or Close ends a word). The
  model is not stepped while it could ask for a word and fewer than 3 words (the current word plus its 2-word
  lookahead) are queued, unless Close has come. A pause in the text therefore pauses the audio; it does not end
  the utterance.
- **TTS Words** times are `(text step + 19 - step of first audio frame) * 80 ms - 40 ms`. The -40 ms was calibrated
  against Qwen3-ForcedAligner on 6 renders (124 words). A word's `end` is the next word's start, so it includes any
  pause after the word.
- **TTS Done** carries an extra `stats` object (`ttfa_ms`, `rtf`, `audio_s`, ...).
- A TTS socket that sends no text for 30 s without Close is treated as Closed, so it can't hold the GPU forever.
