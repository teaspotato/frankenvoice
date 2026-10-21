#!/bin/bash
# llama-server at setup.sh's llama.cpp release, patched to score the tape probe (llama-probe.patch: a request with
# "probe": true gets "probe", the logistic probe of LLAMA_PROBE_FILE on the residual stream at the prompt's last
# token; research: duplex-text-llm/steering). Built for this GPU into $GPUBOX_ROOT/llama.cpp/<release>-cuda-probe,
# where setup.sh looks with LLAMA_CUDA=probe. Then restart the LLM with the probe:
#   LLAMA_CUDA=probe LLAMA_PROBE_FILE=$PWD/tape-probe-L28.bin bash ../setup.sh start   (after stopping llm)
# and set turn.tape's "Probe: answer at" (probeAt) to 0.5.
# tape-probe-L28.bin: int32 layer, int32 n_embd, f32 bias, f32 w[n_embd]; p = sigmoid(w . l_out-28 + bias).
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${GPUBOX_ROOT:-}" ]; then [ -d /content ] && GPUBOX_ROOT=/content || GPUBOX_ROOT=$HOME/gpubox; fi
TAG=$(grep -o '^LLAMA_CPP=b[0-9]*' "$HERE/../setup.sh" | cut -d= -f2)
OUT=$GPUBOX_ROOT/llama.cpp/$TAG-cuda-probe
SRC=$GPUBOX_ROOT/llama.cpp/src-$TAG-probe
export PATH=/usr/local/cuda/bin:$PATH
[ -d "$SRC" ] || git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
git -C "$SRC" apply --reverse --check "$HERE/llama-probe.patch" 2>/dev/null || git -C "$SRC" apply "$HERE/llama-probe.patch"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > "$SRC/cmake.log"
cmake --build "$SRC/build" --target llama-server -j"$(nproc)" > "$SRC/build.log"
mkdir -p "$OUT"
cp -a "$SRC/build/bin/." "$OUT/"
echo "built $OUT/llama-server ($TAG + probe, sm_$ARCH)"
