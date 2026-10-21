#!/bin/bash
# GPU box for Frankenvoice: streaming ASR (Qwen3-ASR via sglang-omni) + streaming TTS (Kyutai)
# + a chat LLM (Gemma 4 12B QAT Q4_0 on llama.cpp) behind one authenticated gateway on :8000,
# exposed through a cloudflared quick tunnel.
#
# usage: bash setup.sh [all|install|fetch|start|stop|restart|restart-gpu|status|ui]   (default: all = install + start)
#        fetch = download the model weights (install does it too, unless NO_FETCH=1)
#        ui = (re)install the app from FV_APP and start only the ui + ui-tunnel processes
# Run from the directory holding server.py / kyutai_tts.py, or set GPUBOX_SRC.
# Tested on a Colab L4 (Ubuntu 22.04, CUDA 12/13 driver). Needs ~22 GB GPU memory (~13 GB with NO_LLM=1).
#
# env:
#   GPUBOX_ROOT   where venvs, logs and state go (default /content on Colab, else ~/gpubox)
#   GPUBOX_DATA   where the token, logs, LLM weights and UI settings/sessions go (default GPUBOX_ROOT; the Docker image
#                 uses the /data volume). Set HF_HOME too to keep the Hugging Face cache there.
#   GPUBOX_TOKEN  bearer token (default: random, saved to $DATA/gpubox.token)
#   ASR_MODEL     Qwen/Qwen3-ASR-1.7B (default) or Qwen/Qwen3-ASR-0.6B
#   ASR_MEM       sglang mem_fraction_static (default 0.15). It is a fraction of the GPU memory that is FREE
#                 when ASR starts, which is why ASR starts first.
#   NO_LLM=1      skip the LLM;  LLM_CTX (8192 tokens, shared by the slots), LLM_SLOTS (2), LLM_SWA_FULL (1), LLM_MTP (1 = MTP
#                 speculative decoding on)
#   TTS_ENGINE   kyutai (default) or qwen (Qwen3-TTS, QWEN_TTS_MODEL, default the 0.6B CustomVoice; QWEN_TTS_VOICE).
#                 Only one is loaded: both do not fit next to ASR and the LLM on an L4. `setup.sh tts qwen|kyutai` swaps
#                 the running one (and remembers it in $ROOT/tts.engine)
#   NO_TUNNEL=1   skip cloudflared;  NO_GW_TUNNEL=1 skips only the gateway's tunnel (the hosted UI doesn't need it)
#   LLAMA_CUDA    12.8 or 13.4: which llama.cpp CUDA build to use (default: picked from the driver)
#   UI_HOST       address the hosted UI listens on (default 127.0.0.1; the Docker image uses 0.0.0.0)
#   NO_UI=1       skip the hosted Frankenvoice UI (proxy on :8787 + its own tunnel). The app comes from
#                 FV_APP (default $ROOT/frankenvoice.tar.gz, a `git archive` of the repo, optionally with a
#                 built dist/), unpacked to $ROOT/frankenvoice.
set -u
SRC=${GPUBOX_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [ -z "${GPUBOX_ROOT:-}" ]; then [ -d /content ] && GPUBOX_ROOT=/content || GPUBOX_ROOT=$HOME/gpubox; fi
ROOT=$GPUBOX_ROOT                                  # venvs, llama.cpp, node, pid files
DATA=${GPUBOX_DATA:-$ROOT}; mkdir -p "$DATA/logs"  # token, logs, LLM weights, UI settings and sessions
export UV_SYSTEM_PYTHON=false   # Colab sets it to true; we always install into venvs
export HF_HUB_DISABLE_PROGRESS_BARS=1
ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}
ALIGN_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
TTS_REPO=kyutai/tts-1.6b-en_fr
QWEN_TTS_MODEL=${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}
QWEN_TTS_PKG=qwen-tts==0.1.1
ASR_PORT=8101; TTS_PORT=8102; LLM_PORT=8103; GW_PORT=8000; UI_PORT=8787
NODE_VERSION=v22.22.0
FV_APP=${FV_APP:-$ROOT/frankenvoice.tar.gz}
APP=$ROOT/frankenvoice
ASR_MEM=${ASR_MEM:-0.15}                           # of the whole GPU, counting what other processes hold: restarting asr alone
                                                   # after the others are up needs ~0.46 for the same 4.6 GB
TORCH=2.9.1                                        # shared by the gw and tts venvs (moshi 0.2.13 resolves to it)
LLAMA_CPP=b11753                                   # llama.cpp release (prebuilt CUDA binaries)
LLM_REPO=google/gemma-4-12B-it-qat-q4_0-gguf       # official QAT Q4_0 GGUF, with the audio/vision projector
LLM_FILE=gemma-4-12b-it-qat-q4_0.gguf
LLM_MMPROJ=mmproj-gemma-4-12b-it-qat-q4_0.gguf
LLM_MTP_REPO=ggml-org/gemma-4-12B-it-GGUF          # MTP draft head for speculative decoding
LLM_MTP_FILE=mtp-gemma-4-12B-it-Q4_0.gguf
LLM_NAME=gemma-4-12b-it
CUDA_MPS=${CUDA_MPS:-1}                            # CUDA MPS: the GPU services' kernels run concurrently instead of time-sliced
ASR_MPS_PRIO=${ASR_MPS_PRIO:-1}                    # same for ASR: its partials re-decode whole turns and made Kyutai steps spike (issue #23)
LLM_MPS_PRIO=${LLM_MPS_PRIO:-1}                    # MPS client priority for the LLM: 1 = below normal, so TTS kernels go first
LLM_MPS_PCT=${LLM_MPS_PCT:-}                       # optional cap on the LLM's share of SMs (40 helped, priority alone helps more: issue #23)
LLM_CTX=${LLM_CTX:-8192}; LLM_SLOTS=${LLM_SLOTS:-2}; LLM_MTP=${LLM_MTP:-1}; LLM_SWA_FULL=${LLM_SWA_FULL:-1}

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

venv() {  # venv NAME pkgs...
  local n=$1; shift
  [ -x "$ROOT/v-$n/bin/python" ] || uv venv -q -p 3.12 "$ROOT/v-$n"
  uv pip install -q --python "$ROOT/v-$n/bin/python" "$@"
}

install() {
  local t0=$SECONDS
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
  command -v ffmpeg >/dev/null || { (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1 || true; }
  # The venvs are the slow part (~19 GB), so the small downloads (cloudflared, llama.cpp, node + the UI) run alongside
  # them. With 8 vCPUs, install is bound by the network (issue #32)
  log "installing venvs (asr: sglang-omni, gw: gateway + forced aligner, tts: moshi) in parallel"
  venv asr --prerelease=allow sglang-omni==0.1.8 > "$DATA/logs/install-asr.log" 2>&1 &
  local a=$!
  # gw and tts pin the same torch, so its ~3.7 GB of CUDA wheels download and unpack once (uv hardlinks them from
  # its cache); unpinned, gw drifts to whatever torch is newest (2.14 + CUDA 13 wheels on 2026-10-21)
  venv gw qwen-asr==0.0.6 torch==$TORCH "fastapi>=0.110" "uvicorn[standard]" websockets httpx numpy soundfile "huggingface_hub[cli]" > "$DATA/logs/install-gw.log" 2>&1 &
  local g=$!
  venv tts moshi==0.2.13 torch==$TORCH "fastapi>=0.110" "uvicorn[standard]" websockets numpy soundfile > "$DATA/logs/install-tts.log" 2>&1 &
  local t=$!
  local q=
  if [ "$(tts_engine)" = qwen ]; then
    venv qtts "$QWEN_TTS_PKG" torch==$TORCH "fastapi>=0.110" "uvicorn[standard]" websockets numpy soundfile httpx > "$DATA/logs/install-qtts.log" 2>&1 &
    q=$!
  fi
  local ok=1
  if [ ! -x "$ROOT/cloudflared" ]; then
    curl -fsSL -o "$ROOT/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$ROOT/cloudflared"
  fi
  install_llama || ok=
  install_ui || ok=
  wait $a || { log "asr install failed"; tail -20 "$DATA/logs/install-asr.log"; ok=; }
  wait $g || { log "gw install failed"; tail -20 "$DATA/logs/install-gw.log"; ok=; }
  wait $t || { log "tts install failed"; tail -20 "$DATA/logs/install-tts.log"; ok=; }
  [ -z "$q" ] || wait $q || { log "qwen-tts install failed"; tail -20 "$DATA/logs/install-qtts.log"; ok=; }
  [ -n "$ok" ] || return 1
  log "venvs ready ($((SECONDS - t0)) s)"
  [ -n "${NO_FETCH:-}" ] || fetch || return 1
  log "install done ($((SECONDS - t0)) s)"
}

fetch() {  # model weights: Hugging Face cache (HF_HOME) for ASR/aligner/TTS, $DATA/models for the LLM
  # One file at a time already fills the link (~340 MB/s on an 8-vCPU Colab). Parallel files, HF_XET_HIGH_PERFORMANCE
  # and fetching while the venvs install were no faster (issue #32)
  local t0=$SECONDS
  log "fetching weights"
  "$ROOT/v-gw/bin/python" - <<EOF || { log "weight download failed"; return 1; }
from huggingface_hub import snapshot_download, hf_hub_download
for r in ["$ASR_MODEL", "$ALIGN_MODEL", "$TTS_REPO"]:
    print(r, snapshot_download(r))
snapshot_download("kyutai/tts-voices", allow_patterns=["expresso/ex03-ex01_happy_001_channel1_334s*"])
if "$(tts_engine)" == "qwen":
    print("$QWEN_TTS_MODEL", snapshot_download("$QWEN_TTS_MODEL"))
if not "${NO_LLM:-}":
    for r, f in [("$LLM_REPO", "$LLM_FILE"), ("$LLM_REPO", "$LLM_MMPROJ"), ("$LLM_MTP_REPO", "$LLM_MTP_FILE")]:
        print(f, hf_hub_download(r, f, local_dir="$DATA/models"))
EOF
  log "weights ready ($((SECONDS - t0)) s)"
}

llama_dir() {  # where the llama.cpp release for this driver lives
  local cu=12.8
  if [ -n "${LLAMA_CUDA:-}" ]; then echo "$ROOT/llama.cpp/$LLAMA_CPP-cuda-$LLAMA_CUDA"; return; fi
  [ "$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$')" -ge 13 ] 2>/dev/null && cu=13.4
  echo "$ROOT/llama.cpp/$LLAMA_CPP-cuda-$cu"
}

install_llama() {  # prebuilt llama.cpp (server + CUDA runtime libs); a few seconds, no compiler needed
  [ -n "${NO_LLM:-}" ] && return 0
  local d; d=$(llama_dir); [ -x "$d/llama-server" ] && return 0
  local cu=${d##*-cuda-} url=https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_CPP
  mkdir -p "$d"
  curl -fsSL "$url/llama-$LLAMA_CPP-bin-ubuntu-cuda-$cu-x64.tar.gz" | tar xz -C "$d" --strip-components=1
  curl -fsSL "$url/cudart-llama-$LLAMA_CPP-bin-ubuntu-cuda-$cu-x64.tar.gz" | tar xz -C "$d" --strip-components=1
  log "llama.cpp $LLAMA_CPP (CUDA $cu) in $d"
}

install_ui() {  # Node 22 + pnpm, then the app from $FV_APP: pnpm install, and pnpm build unless dist/ came with it
  [ -n "${NO_UI:-}" ] && return 0
  [ -f "$FV_APP" ] || { log "no $FV_APP: skipping the UI (upload a git archive of the repo there)"; return 0; }
  export PATH=$ROOT/node/bin:$PATH
  if ! node -v 2>/dev/null | grep -q '^v2[2-9]'; then
    mkdir -p "$ROOT/node"
    curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" | tar xJ -C "$ROOT/node" --strip-components=1
    log "node $(node -v) in $ROOT/node"
  fi
  command -v pnpm >/dev/null || npm install -g --silent pnpm@10.28.0 > "$DATA/logs/install-ui.log" 2>&1
  mkdir -p "$APP"                             # keep node_modules; drop files the new archive may no longer have
  find "$APP" -mindepth 1 -maxdepth 1 ! -name node_modules -exec rm -rf {} +
  tar xzf "$FV_APP" -C "$APP"
  log "installing the UI in $APP"
  (cd "$APP" && CI=1 pnpm install --frozen-lockfile && { [ -f dist/index.html ] || pnpm build; }) >> "$DATA/logs/install-ui.log" 2>&1 \
    || { log "ui install failed"; tail -20 "$DATA/logs/install-ui.log"; return 1; }
  log "ui installed"
}

start_ui() {  # the Frankenvoice proxy, serving the built UI behind the gpubox token, plus its own quick tunnel
  [ -n "${NO_UI:-}" ] && return 0
  [ -f "$APP/dist/index.html" ] || { log "no $APP/dist: run install (or ui) first"; return 0; }
  local TOKEN; TOKEN=$(cat "$DATA/gpubox.token")
  local llm=; [ -z "${NO_LLM:-}" ] && llm=http://127.0.0.1:$LLM_PORT/v1
  # Settings preset from the environment; anything entered in the UI's Settings wins and is saved to fv-settings.json.
  alive ui || (cd "$APP" && PATH=$ROOT/node/bin:$PATH FV_PROXY_PORT=$UI_PORT FV_PROXY_HOST=${UI_HOST:-127.0.0.1} FV_STATIC_DIR=dist FV_HOST_TOKEN=$TOKEN \
      FV_SETTINGS_FILE=$DATA/fv-settings.json FV_SESSIONS_DIR=$DATA/fv-sessions \
      GPUBOX_BASE_URL=ws://127.0.0.1:$GW_PORT GPUBOX_TOKEN=$TOKEN OPENAI_COMPAT_BASE_URL=$llm \
      spawn ui node_modules/.bin/tsx server/index.ts)
  wait_http ui "http://127.0.0.1:$UI_PORT/index.html?token=$TOKEN" 60 || return 1
  if [ -z "${NO_TUNNEL:-}" ] && ! alive ui-tunnel; then
    spawn ui-tunnel "$ROOT/cloudflared" tunnel --no-autoupdate --protocol http2 --url http://localhost:$UI_PORT
    for _ in $(seq 1 60); do grep -qo 'https://[a-z0-9-]*\.trycloudflare\.com' "$DATA/logs/ui-tunnel.log" && break; sleep 1; done
  fi
}

spawn() {  # spawn NAME cmd... : detached, own session, pid in $ROOT/NAME.pid
  local n=$1; shift
  setsid nohup "$@" > "$DATA/logs/$n.log" 2>&1 < /dev/null &
  echo $! > "$ROOT/$n.pid"
  log "started $n pid $(cat "$ROOT/$n.pid") (log $DATA/logs/$n.log)"
}

alive() { [ -f "$ROOT/$1.pid" ] && kill -0 "$(cat "$ROOT/$1.pid")" 2>/dev/null; }

wait_http() {  # wait_http NAME URL TIMEOUT_S
  for _ in $(seq 1 "$3"); do
    curl -sf -o /dev/null "$2" && return 0
    alive "$1" || { log "$1 died"; tail -40 "$DATA/logs/$1.log"; return 1; }
    sleep 1
  done
  log "$1 not up after $3 s"; return 1
}

# CUDA MPS. Without it, the separate CUDA processes time-slice the GPU, and Kyutai's many small per-frame
# kernels wait behind llama.cpp's decode. Clients find the daemon through these two variables.
export CUDA_MPS_PIPE_DIRECTORY=$ROOT/mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/mps/log
mps_up() { echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1; }
mps_start() {
  if [ "$CUDA_MPS" != 1 ]; then unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; return 0; fi
  mps_up && return 0
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && log "CUDA MPS daemon started" || { log "CUDA MPS failed to start"; return 1; }
}
mps_stop() { mps_up && echo quit | nvidia-cuda-mps-control && log "CUDA MPS daemon stopped"; return 0; }

tts_engine() { local e=${TTS_ENGINE:-}; [ -z "$e" ] && [ -f "$ROOT/tts.engine" ] && e=$(cat "$ROOT/tts.engine"); echo "${e:-kyutai}"; }
tts_model() { [ "$(tts_engine)" = qwen ] && echo "$QWEN_TTS_MODEL" || echo "$TTS_REPO"; }

install_qwen_tts() {  # its own venv: qwen-tts pins a transformers that moshi and qwen-asr do not share
  [ -x "$ROOT/v-qtts/bin/python" ] && "$ROOT/v-qtts/bin/python" -c "import qwen_tts" 2>/dev/null && return 0
  log "installing qwen-tts"
  venv qtts "$QWEN_TTS_PKG" torch==$TORCH "fastapi>=0.110" "uvicorn[standard]" websockets numpy soundfile httpx > "$DATA/logs/install-qtts.log" 2>&1 \
    || { log "qwen-tts install failed"; tail -20 "$DATA/logs/install-qtts.log"; return 1; }
}

start_tts() {  # the TTS worker of $(tts_engine) on $TTS_PORT
  alive tts && return 0
  local TOKEN; TOKEN=$(cat "$DATA/gpubox.token")
  if [ "$(tts_engine)" = qwen ]; then
    install_qwen_tts || return 1
    spawn tts env GPUBOX_TOKEN="$TOKEN" ALIGN_URL=http://127.0.0.1:$GW_PORT/v1/align QWEN_TTS_MODEL="$QWEN_TTS_MODEL" \
      "$ROOT/v-qtts/bin/python" "$SRC/qwen3_tts_worker.py" --port $TTS_PORT
  else
    spawn tts "$ROOT/v-tts/bin/python" "$SRC/kyutai_tts.py" --port $TTS_PORT
  fi
  wait_http tts "http://127.0.0.1:$TTS_PORT/health" 900
}

swap_tts() {  # swap_tts kyutai|qwen : replace the running TTS worker; the gateway, LLM, ASR and tunnels stay up
  case "$1" in kyutai|qwen) ;; *) echo "usage: $0 tts kyutai|qwen"; return 2 ;; esac
  echo "$1" > "$ROOT/tts.engine"
  TTS_ENGINE=$1
  if alive tts; then
    local p; p=$(cat "$ROOT/tts.pid"); kill -TERM -- "-$p" 2>/dev/null || kill "$p"; log "stopped tts ($p)"
    for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    rm -f "$ROOT/tts.pid"
  fi
  local t=$SECONDS
  start_tts && log "tts is $1 ($((SECONDS - t)) s)"
}

stop_gpu() {  # the GPU services only; tunnels and the UI keep their URLs
  for n in gateway tts llm asr; do
    if alive $n; then
      local p; p=$(cat "$ROOT/$n.pid")
      kill -TERM -- "-$p" 2>/dev/null || kill "$p"
      log "stopped $n ($p)"
    fi
    rm -f "$ROOT/$n.pid"
  done
  ports_free
  mps_stop
}

PORTS_FILTER="( sport = :$ASR_PORT or sport = :$TTS_PORT or sport = :$LLM_PORT or sport = :$GW_PORT )"
ports_free() {  # wait for the services' ports, so restarted ones don't fall back to random ports
  for _ in $(seq 1 60); do
    # Any socket, not just listeners: sglang refuses a port with connections still in TIME_WAIT and
    # silently serves on a random one instead. Local ports only: a client socket whose peer is one
    # of these ports holds nothing here.
    [ -z "$(ss -tanH "$PORTS_FILTER")" ] && return 0
    sleep 1
  done
  log "ports still busy: $(ss -tanH "$PORTS_FILTER" | awk '{print $1, $4}' | sort -u | tr '\n' ' ')"
}

stop() {
  for n in ui-tunnel ui tunnel gateway tts llm asr; do
    if alive $n; then
      local p; p=$(cat "$ROOT/$n.pid")
      kill -TERM -- "-$p" 2>/dev/null || kill "$p"   # whole process group (setsid)
      log "stopped $n ($p)"
    fi
    rm -f "$ROOT/$n.pid"
  done
  ports_free
  mps_stop
}

start() {
  mps_start || return 1
  [ -f "$DATA/gpubox.token" ] || echo "${GPUBOX_TOKEN:-$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)}" > "$DATA/gpubox.token"
  [ -n "${GPUBOX_TOKEN:-}" ] && echo "$GPUBOX_TOKEN" > "$DATA/gpubox.token"
  local TOKEN; TOKEN=$(cat "$DATA/gpubox.token")
  alive asr || PATH="$ROOT/v-asr/bin:$PATH" spawn asr env CUDA_MPS_CLIENT_PRIORITY="$ASR_MPS_PRIO" "$ROOT/v-asr/bin/sgl-omni" serve --model-path $ASR_MODEL --model-name $ASR_MODEL --port $ASR_PORT \
      --asr.engine.quantization fp8 --asr.engine.mem_fraction_static "$ASR_MEM" --asr.engine.max_running_requests 8
  wait_http asr "http://127.0.0.1:$ASR_PORT/health" 900 || return 1
  if [ -z "${NO_LLM:-}" ] && ! alive llm; then
    # q8_0 KV; --ctx-checkpoints 0: Gemma's sliding-window cache still allows rolling back ~500 tokens
    # (an edited last turn), and checkpoints cost ~400 ms per request; --cache-ram 0: no host-RAM copy per request
    local d; d=$(llama_dir)
    local mtp=(); [ "$LLM_MTP" = 1 ] && mtp=(--spec-type draft-mtp -md "$DATA/models/$LLM_MTP_FILE")
    # --swa-full: a full-size sliding-window cache, so a prompt that diverges from the slot's cached
    # tokens (a cancelled draft, a new user turn after a reply) reuses the shared prefix instead of re-reading it all
    local swa=(); [ "$LLM_SWA_FULL" = 1 ] && swa=(--swa-full)
    local pct=(env CUDA_MPS_CLIENT_PRIORITY="$LLM_MPS_PRIO"); [ -n "$LLM_MPS_PCT" ] && pct+=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$LLM_MPS_PCT")
    LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}" spawn llm "${pct[@]}" "$d/llama-server" -m "$DATA/models/$LLM_FILE" \
        --mmproj "$DATA/models/$LLM_MMPROJ" --alias $LLM_NAME --host 127.0.0.1 --port $LLM_PORT \
        -ngl 999 -fa on -c "$LLM_CTX" -np "$LLM_SLOTS" -ctk q8_0 -ctv q8_0 --ctx-checkpoints 0 --cache-ram 0 "${swa[@]}" \
        --jinja --reasoning off "${mtp[@]}"
    wait_http llm "http://127.0.0.1:$LLM_PORT/health" 600 || return 1
  fi
  start_tts || return 1
  local llm_name=$LLM_NAME; [ -n "${NO_LLM:-}" ] && llm_name=
  alive gateway || GPUBOX_TOKEN=$TOKEN ASR_URL=http://127.0.0.1:$ASR_PORT TTS_URL=ws://127.0.0.1:$TTS_PORT \
      LLM_URL=http://127.0.0.1:$LLM_PORT LLM_MODEL=$llm_name ASR_MODEL=$ASR_MODEL ALIGN_MODEL=$ALIGN_MODEL TTS_MODEL=$(tts_model) \
      spawn gateway "$ROOT/v-gw/bin/python" "$SRC/server.py" --port $GW_PORT
  wait_http gateway "http://127.0.0.1:$GW_PORT/health?token=$TOKEN" 600 || return 1
  if [ -z "${NO_TUNNEL:-}${NO_GW_TUNNEL:-}" ] && ! alive tunnel; then
    # --protocol http2: over QUIC a quick tunnel moved ~110 KB/s (2.3 MB in 20 s); over HTTP/2, 3.4 MB/s
    spawn tunnel "$ROOT/cloudflared" tunnel --no-autoupdate --protocol http2 --url http://localhost:$GW_PORT
    for _ in $(seq 1 60); do grep -qo 'https://[a-z0-9-]*\.trycloudflare\.com' "$DATA/logs/tunnel.log" && break; sleep 1; done
  fi
  start_ui
  status
}

status() {
  for n in asr llm tts gateway tunnel ui ui-tunnel; do alive $n && echo "$n: up (pid $(cat "$ROOT/$n.pid"))" || echo "$n: down"; done
  local url; url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$DATA/logs/tunnel.log" 2>/dev/null | head -1)
  echo "GPUBOX_URL=${url:-http://localhost:$GW_PORT}"
  echo "GPUBOX_TOKEN=$(cat "$DATA/gpubox.token" 2>/dev/null)"
  [ -n "$url" ] && echo "$url" > "$DATA/gpubox.url"
  if [ -z "${NO_UI:-}" ]; then
    local ui; ui=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$DATA/logs/ui-tunnel.log" 2>/dev/null | head -1)
    echo "UI_URL=${ui:-http://localhost:$UI_PORT}   (open once as UI_URL/?token=GPUBOX_TOKEN)"
  fi
}

case "${1:-all}" in
  install) install ;;
  fetch) fetch ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  restart-gpu) stop_gpu; start ;;
  status) status ;;
  ui) install_ui && start_ui && status ;;
  tts) mps_start && swap_tts "${2:-}" ;;
  all) install && start ;;
  *) echo "usage: $0 [all|install|fetch|start|stop|restart|restart-gpu|status|ui]"; exit 2 ;;
esac
