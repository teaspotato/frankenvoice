#!/bin/bash
# GPU box for Frankenvoice: streaming ASR (Qwen3-ASR via sglang-omni) + streaming TTS (Kyutai)
# + a chat LLM (Gemma 4 12B QAT Q4_0 on llama.cpp) behind one authenticated gateway on :8000,
# exposed through a cloudflared quick tunnel.
#
# usage: bash setup.sh [all|install|start|stop|status]     (default: all = install + start)
# Run from the directory holding server.py / kyutai_tts.py, or set GPUBOX_SRC.
# Tested on a Colab L4 (Ubuntu 22.04, CUDA 12/13 driver). Needs ~22 GB GPU memory (~13 GB with NO_LLM=1).
#
# env:
#   GPUBOX_ROOT   where venvs, logs and state go (default /content on Colab, else ~/gpubox)
#   GPUBOX_TOKEN  bearer token (default: random, saved to $ROOT/gpubox.token)
#   ASR_MODEL     Qwen/Qwen3-ASR-1.7B (default) or Qwen/Qwen3-ASR-0.6B
#   ASR_MEM       sglang mem_fraction_static (default 0.15). It is a fraction of the GPU memory that is FREE
#                 when ASR starts, which is why ASR starts first.
#   NO_LLM=1      skip the LLM;  LLM_CTX (32768 tokens, shared by the slots), LLM_SLOTS (4), LLM_MTP (1 = MTP
#                 speculative decoding on)
#   NO_TUNNEL=1   skip cloudflared
set -u
SRC=${GPUBOX_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [ -z "${GPUBOX_ROOT:-}" ]; then [ -d /content ] && GPUBOX_ROOT=/content || GPUBOX_ROOT=$HOME/gpubox; fi
ROOT=$GPUBOX_ROOT; mkdir -p "$ROOT/logs"
export UV_SYSTEM_PYTHON=false   # Colab sets it to true; we always install into venvs
export HF_HUB_DISABLE_PROGRESS_BARS=1
ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}
ALIGN_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
TTS_REPO=kyutai/tts-1.6b-en_fr
ASR_PORT=8101; TTS_PORT=8102; LLM_PORT=8103; GW_PORT=8000
ASR_MEM=${ASR_MEM:-0.15}
LLAMA_CPP=b11753                                   # llama.cpp release (prebuilt CUDA binaries)
LLM_REPO=google/gemma-4-12B-it-qat-q4_0-gguf       # official QAT Q4_0 GGUF, with the audio/vision projector
LLM_FILE=gemma-4-12b-it-qat-q4_0.gguf
LLM_MMPROJ=mmproj-gemma-4-12b-it-qat-q4_0.gguf
LLM_MTP_REPO=ggml-org/gemma-4-12B-it-GGUF          # MTP draft head for speculative decoding
LLM_MTP_FILE=mtp-gemma-4-12B-it-Q4_0.gguf
LLM_NAME=gemma-4-12b-it
LLM_CTX=${LLM_CTX:-32768}; LLM_SLOTS=${LLM_SLOTS:-4}; LLM_MTP=${LLM_MTP:-1}

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

venv() {  # venv NAME pkgs...
  local n=$1; shift
  [ -x "$ROOT/v-$n/bin/python" ] || uv venv -q -p 3.12 "$ROOT/v-$n"
  uv pip install -q --python "$ROOT/v-$n/bin/python" "$@"
}

install() {
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
  command -v ffmpeg >/dev/null || { (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1 || true; }
  if [ ! -x "$ROOT/cloudflared" ]; then
    curl -fsSL -o "$ROOT/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$ROOT/cloudflared"
  fi
  install_llama
  log "installing venvs (asr: sglang-omni, gw: gateway + forced aligner, tts: moshi) in parallel"
  venv asr --prerelease=allow sglang-omni==0.1.8 > "$ROOT/logs/install-asr.log" 2>&1 &
  local a=$!
  venv gw qwen-asr==0.0.6 "fastapi>=0.110" "uvicorn[standard]" websockets httpx numpy soundfile "huggingface_hub[cli]" > "$ROOT/logs/install-gw.log" 2>&1 &
  local g=$!
  venv tts moshi==0.2.13 "fastapi>=0.110" "uvicorn[standard]" websockets numpy soundfile > "$ROOT/logs/install-tts.log" 2>&1 &
  local t=$!
  wait $a || { log "asr install failed"; tail -20 "$ROOT/logs/install-asr.log"; return 1; }
  wait $g || { log "gw install failed"; tail -20 "$ROOT/logs/install-gw.log"; return 1; }
  wait $t || { log "tts install failed"; tail -20 "$ROOT/logs/install-tts.log"; return 1; }
  log "prefetching weights"
  "$ROOT/v-gw/bin/python" - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download
for r in ["$ASR_MODEL", "$ALIGN_MODEL", "$TTS_REPO"]:
    print(r, snapshot_download(r))
snapshot_download("kyutai/tts-voices", allow_patterns=["expresso/ex03-ex01_happy_001_channel1_334s*"])
if not "${NO_LLM:-}":
    for r, f in [("$LLM_REPO", "$LLM_FILE"), ("$LLM_REPO", "$LLM_MMPROJ"), ("$LLM_MTP_REPO", "$LLM_MTP_FILE")]:
        print(f, hf_hub_download(r, f, local_dir="$ROOT/models"))
EOF
  log "install done"
}

llama_dir() {  # where the llama.cpp release for this driver lives
  local cu=12.8
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

spawn() {  # spawn NAME cmd... : detached, own session, pid in $ROOT/NAME.pid
  local n=$1; shift
  setsid nohup "$@" > "$ROOT/logs/$n.log" 2>&1 < /dev/null &
  echo $! > "$ROOT/$n.pid"
  log "started $n pid $(cat "$ROOT/$n.pid") (log $ROOT/logs/$n.log)"
}

alive() { [ -f "$ROOT/$1.pid" ] && kill -0 "$(cat "$ROOT/$1.pid")" 2>/dev/null; }

wait_http() {  # wait_http NAME URL TIMEOUT_S
  for _ in $(seq 1 "$3"); do
    curl -sf -o /dev/null "$2" && return 0
    alive "$1" || { log "$1 died"; tail -40 "$ROOT/logs/$1.log"; return 1; }
    sleep 1
  done
  log "$1 not up after $3 s"; return 1
}

stop() {
  for n in tunnel gateway tts llm asr; do
    if alive $n; then
      local p; p=$(cat "$ROOT/$n.pid")
      kill -TERM -- "-$p" 2>/dev/null || kill "$p"   # whole process group (setsid)
      log "stopped $n ($p)"
    fi
    rm -f "$ROOT/$n.pid"
  done
  sleep 3
}

start() {
  [ -f "$ROOT/gpubox.token" ] || echo "${GPUBOX_TOKEN:-$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)}" > "$ROOT/gpubox.token"
  [ -n "${GPUBOX_TOKEN:-}" ] && echo "$GPUBOX_TOKEN" > "$ROOT/gpubox.token"
  local TOKEN; TOKEN=$(cat "$ROOT/gpubox.token")
  alive asr || PATH="$ROOT/v-asr/bin:$PATH" spawn asr "$ROOT/v-asr/bin/sgl-omni" serve --model-path $ASR_MODEL --model-name $ASR_MODEL --port $ASR_PORT \
      --asr.engine.quantization fp8 --asr.engine.mem_fraction_static "$ASR_MEM" --asr.engine.max_running_requests 8
  wait_http asr "http://127.0.0.1:$ASR_PORT/health" 900 || return 1
  if [ -z "${NO_LLM:-}" ] && ! alive llm; then
    # q8_0 KV; --ctx-checkpoints 0: Gemma's sliding-window cache still allows rolling back ~500 tokens
    # (an edited last turn), and checkpoints cost ~400 ms per request; --cache-ram 0: no host-RAM copy per request
    local d; d=$(llama_dir)
    local mtp=(); [ "$LLM_MTP" = 1 ] && mtp=(--spec-type draft-mtp -md "$ROOT/models/$LLM_MTP_FILE")
    LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}" spawn llm "$d/llama-server" -m "$ROOT/models/$LLM_FILE" \
        --mmproj "$ROOT/models/$LLM_MMPROJ" --alias $LLM_NAME --host 127.0.0.1 --port $LLM_PORT \
        -ngl 999 -fa on -c "$LLM_CTX" -np "$LLM_SLOTS" -ctk q8_0 -ctv q8_0 --ctx-checkpoints 0 --cache-ram 0 \
        --jinja --reasoning off "${mtp[@]}"
    wait_http llm "http://127.0.0.1:$LLM_PORT/health" 600 || return 1
  fi
  alive tts || spawn tts "$ROOT/v-tts/bin/python" "$SRC/kyutai_tts.py" --port $TTS_PORT
  wait_http tts "http://127.0.0.1:$TTS_PORT/health" 900 || return 1
  local llm_name=$LLM_NAME; [ -n "${NO_LLM:-}" ] && llm_name=
  alive gateway || GPUBOX_TOKEN=$TOKEN ASR_URL=http://127.0.0.1:$ASR_PORT TTS_URL=ws://127.0.0.1:$TTS_PORT \
      LLM_URL=http://127.0.0.1:$LLM_PORT LLM_MODEL=$llm_name ASR_MODEL=$ASR_MODEL ALIGN_MODEL=$ALIGN_MODEL TTS_MODEL=$TTS_REPO \
      spawn gateway "$ROOT/v-gw/bin/python" "$SRC/server.py" --port $GW_PORT
  wait_http gateway "http://127.0.0.1:$GW_PORT/health?token=$TOKEN" 600 || return 1
  if [ -z "${NO_TUNNEL:-}" ] && ! alive tunnel; then
    spawn tunnel "$ROOT/cloudflared" tunnel --no-autoupdate --url http://localhost:$GW_PORT
    for _ in $(seq 1 60); do grep -qo 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/logs/tunnel.log" && break; sleep 1; done
  fi
  status
}

status() {
  for n in asr llm tts gateway tunnel; do alive $n && echo "$n: up (pid $(cat "$ROOT/$n.pid"))" || echo "$n: down"; done
  local url; url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/logs/tunnel.log" 2>/dev/null | head -1)
  echo "GPUBOX_URL=${url:-http://localhost:$GW_PORT}"
  echo "GPUBOX_TOKEN=$(cat "$ROOT/gpubox.token" 2>/dev/null)"
  [ -n "$url" ] && echo "$url" > "$ROOT/gpubox.url"
}

case "${1:-all}" in
  install) install ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  all) install && start ;;
  *) echo "usage: $0 [all|install|start|stop|restart|status]"; exit 2 ;;
esac
