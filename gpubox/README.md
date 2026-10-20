# GPU box

One authenticated HTTP/WebSocket server for Frankenvoice: streaming ASR (Qwen3-ASR-1.7B) and streaming
TTS (Kyutai `tts-1.6b-en_fr`). The wire protocol is in [PROTOCOL.md](PROTOCOL.md).

## Processes

| Process | Venv | Port | What it does |
|---|---|---|---|
| `asr` | `v-asr` (sglang-omni 0.1.8) | 127.0.0.1:8101 | `sgl-omni serve` Qwen3-ASR-1.7B with FP8 weights, `mem_fraction_static 0.35`, `max_running_requests 8` |
| `tts` | `v-tts` (moshi 0.2.13) | 127.0.0.1:8102 | `kyutai_tts.py`: Kyutai streaming generation, one reply at a time |
| `gateway` | `v-gw` (qwen-asr 0.0.6, FastAPI) | 0.0.0.0:8000 | `server.py`: auth, ASR segmenting/partials/finals, Qwen3-ForcedAligner-0.6B, relays `/v1/tts` |
| `tunnel` | none | none | `cloudflared tunnel --url http://localhost:8000` (quick tunnel, random `*.trycloudflare.com`) |

Moshi and qwen-asr need different torch/transformers versions, so TTS runs in its own venv and process.
Everything uses about 17.5 GB of GPU memory, which fits a 24 GB L4.

## Run

On any Ubuntu box with an NVIDIA GPU (24 GB or more), CUDA driver, `curl` and internet access:

```bash
# copy this folder to the box (e.g. /content/gpubox on Colab), then:
bash gpubox/setup.sh            # install (about 3 min) + start (about 3 min the first time)
bash gpubox/setup.sh status     # prints GPUBOX_URL and GPUBOX_TOKEN
```

Other commands: `setup.sh install | start | stop | restart | status`. `start` only starts what isn't already
running. To restart just one process, kill its process group (`kill -- -$(cat $ROOT/tts.pid)`) and run `start` again.
The tunnel keeps its URL as long as the `tunnel` process lives.

Environment variables: `GPUBOX_ROOT` (venvs, logs, pid files; default `/content` if it exists, else `~/gpubox`),
`GPUBOX_TOKEN` (default: random, saved in `$ROOT/gpubox.token`), `ASR_MEM` (0.35), `NO_TUNNEL=1`,
`KYUTAI_NQ` (codebooks, 32), `KYUTAI_WORD_OFFSET_S` (-0.04), `KYUTAI_IDLE_CLOSE_S` (30).
Logs are in `$ROOT/logs/{asr,tts,gateway,tunnel}.log`.

### Colab

1. Create an L4 runtime: `colab --auth=oauth2 new -s gpubox --gpu L4`.
2. Upload this folder: `colab --auth=oauth2 upload -s gpubox gpubox/setup.sh /content/gpubox/setup.sh` (same for
   `server.py` and `kyutai_tts.py`).
3. Run `bash /content/gpubox/setup.sh` **detached**: a `colab exec` call times out long before it finishes. Use
   `subprocess.Popen(['bash', '/content/gpubox/setup.sh'], start_new_session=True, stdout=open('/content/setup.log', 'w'), stderr=subprocess.STDOUT)`.
4. `tail /content/setup.log` until it prints `GPUBOX_URL=...` and `GPUBOX_TOKEN=...`.

Colab notes: Colab sets `UV_SYSTEM_PYTHON=true`; `setup.sh` unsets it and installs into venvs. Don't use `pkill -f`
there, because it can match your own shell; kill by pid. The VM (and so the tunnel URL) goes away when the
runtime is recycled.

## Point Frankenvoice at it

Set the local proxy's GPU box URL to the `GPUBOX_URL` and its token to `GPUBOX_TOKEN`. Check it with
`curl -H "Authorization: Bearer $GPUBOX_TOKEN" $GPUBOX_URL/health`.

## Behavior notes

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
