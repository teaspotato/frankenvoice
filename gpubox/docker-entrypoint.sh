#!/bin/bash
# Container entrypoint: fetch weights (first start only downloads), start every process with setup.sh,
# print where to open the UI, then stay up while they run. Any process dying stops the container, so
# `docker run --restart unless-stopped` brings the whole box back.
set -u
SETUP=$GPUBOX_SRC/setup.sh
log() { echo "[docker $(date +%H:%M:%S)] $*"; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || { log "no GPU visible: run with --gpus all (and the NVIDIA Container Toolkit installed)"; exit 1; }

rm -f "$GPUBOX_ROOT"/*.pid   # left over from the previous run of this container
rm -rf "$GPUBOX_ROOT/mps"     # setup.sh runs the GPU services under CUDA MPS (CUDA_MPS=0 turns it off)
stop() { bash "$SETUP" stop; }
trap 'log "stopping"; stop; exit 0' TERM INT

# in the background + wait, so docker stop interrupts a long first start instead of waiting it out
bash "$SETUP" fetch & wait $! || log "weight download failed; starting with what is already in /data"
bash "$SETUP" start & wait $! || { log "start failed (logs in /data/logs)"; stop; exit 1; }

TOKEN=$(cat "$GPUBOX_DATA/gpubox.token")
UI_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$GPUBOX_DATA/logs/ui-tunnel.log" 2>/dev/null | head -1)
cat <<MSG

  Frankenvoice is up. Open the web UI with the token once (it is kept in a cookie for 30 days):
    on this machine:  http://localhost:8787/?token=$TOKEN${UI_URL:+
    anywhere:         $UI_URL/?token=$TOKEN}
  The microphone needs https or localhost: from another machine use the https link
  or an SSH tunnel (ssh -L 8787:localhost:8787 <gpu-host>).

MSG

procs=(asr tts gateway); [ -z "${NO_LLM:-}" ] && procs+=(llm); [ -z "${NO_UI:-}" ] && procs+=(ui)
while true; do
  sleep 10 & wait $!
  for n in "${procs[@]}"; do
    if ! { [ -f "$GPUBOX_ROOT/$n.pid" ] && kill -0 "$(cat "$GPUBOX_ROOT/$n.pid")" 2>/dev/null; }; then
      log "$n exited; last lines of /data/logs/$n.log:"; tail -30 "$GPUBOX_DATA/logs/$n.log"
      stop; exit 1
    fi
  done
done
