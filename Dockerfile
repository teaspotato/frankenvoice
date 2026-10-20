# Frankenvoice on one GPU: Qwen3-ASR (sglang-omni), Gemma 4 12B (llama.cpp), Kyutai TTS, the gateway and the
# hosted web UI, all in one image. Model weights download on first start into the /data volume.
#
#   docker build -t frankenvoice .
#   docker run --gpus all -p 8787:8787 -v frankenvoice:/data frankenvoice
#
# Needs an NVIDIA GPU with 24 GB (L4, A10, 3090/4090 or bigger) and a driver for CUDA 12.8 or newer.
# See gpubox/README.md#docker.
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg git iproute2 tini xz-utils build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG NODE_VERSION=v22.22.0
RUN curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" \
      | tar xJ -C /usr/local --strip-components=1 \
    && npm install -g --silent pnpm@10.28.0 \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# /opt/gpubox holds what the image installs (venvs, llama.cpp, cloudflared, the app); /data holds what a run
# creates (weights, Hugging Face cache, token, logs, UI settings and recorded sessions).
ENV GPUBOX_ROOT=/opt/gpubox \
    GPUBOX_DATA=/data \
    GPUBOX_SRC=/opt/gpubox/frankenvoice/gpubox \
    HF_HOME=/data/hf \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    LLAMA_CUDA=12.8 \
    UI_HOST=0.0.0.0 \
    NO_GW_TUNNEL=1

# Python venvs (sglang-omni, gateway + aligner, moshi), llama.cpp and cloudflared. Only setup.sh is copied
# first, so app changes don't rebuild this (slow) layer.
COPY gpubox/setup.sh /tmp/setup/setup.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    GPUBOX_DATA=/tmp/setup NO_FETCH=1 NO_UI=1 bash /tmp/setup/setup.sh install \
    && rm -rf /tmp/setup

# The web UI and the proxy that serves it.
WORKDIR /opt/gpubox/frankenvoice
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store CI=1 pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

VOLUME /data
# 8787: the web UI (open http://localhost:8787/?token=<token printed at start>); 8000: the gateway (ASR/TTS/LLM API)
EXPOSE 8787 8000
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/opt/gpubox/frankenvoice/gpubox/docker-entrypoint.sh"]
