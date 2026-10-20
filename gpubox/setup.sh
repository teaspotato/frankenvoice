#!/bin/bash
# GPU box for Frankenvoice: streaming ASR (Qwen3-ASR-1.7B via sglang-omni) + streaming TTS (Kyutai)
# behind one authenticated gateway on :8000, exposed through a cloudflared quick tunnel.
#
# usage: bash setup.sh [all|install|start|stop|status]     (default: all = install + start)
# Run from the directory holding server.py / kyutai_tts.py, or set GPUBOX_SRC.
# Tested on a Colab L4 (Ubuntu 22.04, CUDA 12/13 driver). Needs ~20 GB GPU memory.
#
# env:
#   GPUBOX_ROOT   where venvs, logs and state go (default /content on Colab, else ~/gpubox)
#   GPUBOX_TOKEN  bearer token (default: random, saved to $ROOT/gpubox.token)
#   ASR_MEM       sglang mem_fraction_static (default 0.35)
#   NO_TUNNEL=1   skip cloudflared
set -u
SRC=${GPUBOX_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [ -z "${GPUBOX_ROOT:-}" ]; then [ -d /content ] && GPUBOX_ROOT=/content || GPUBOX_ROOT=$HOME/gpubox; fi
ROOT=$GPUBOX_ROOT; mkdir -p "$ROOT/logs"
export UV_SYSTEM_PYTHON=false   # Colab sets it to true; we always install into venvs
export HF_HUB_DISABLE_PROGRESS_BARS=1
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
ALIGN_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
TTS_REPO=kyutai/tts-1.6b-en_fr
ASR_PORT=8101; TTS_PORT=8102; GW_PORT=8000
ASR_MEM=${ASR_MEM:-0.35}

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
from huggingface_hub import snapshot_download
for r in ["$ASR_MODEL", "$ALIGN_MODEL", "$TTS_REPO"]:
    print(r, snapshot_download(r))
snapshot_download("kyutai/tts-voices", allow_patterns=["expresso/ex03-ex01_happy_001_channel1_334s*"])
EOF
  log "install done"
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
  for n in tunnel gateway tts asr; do
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
  alive tts || spawn tts "$ROOT/v-tts/bin/python" "$SRC/kyutai_tts.py" --port $TTS_PORT
  wait_http tts "http://127.0.0.1:$TTS_PORT/health" 900 || return 1
  alive gateway || GPUBOX_TOKEN=$TOKEN ASR_URL=http://127.0.0.1:$ASR_PORT TTS_URL=ws://127.0.0.1:$TTS_PORT \
      ASR_MODEL=$ASR_MODEL ALIGN_MODEL=$ALIGN_MODEL TTS_MODEL=$TTS_REPO \
      spawn gateway "$ROOT/v-gw/bin/python" "$SRC/server.py" --port $GW_PORT
  wait_http gateway "http://127.0.0.1:$GW_PORT/health?token=$TOKEN" 600 || return 1
  if [ -z "${NO_TUNNEL:-}" ] && ! alive tunnel; then
    spawn tunnel "$ROOT/cloudflared" tunnel --no-autoupdate --url http://localhost:$GW_PORT
    for _ in $(seq 1 60); do grep -qo 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/logs/tunnel.log" && break; sleep 1; done
  fi
  status
}

status() {
  for n in asr tts gateway tunnel; do alive $n && echo "$n: up (pid $(cat "$ROOT/$n.pid"))" || echo "$n: down"; done
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
