#!/usr/bin/env bash
# Launch the pinned NanoEngine single-request SP2 GQA microbenchmark.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
upstream=${NANOENGINE_ROOT:-"$here/../../../NanoEngine"}
revision=8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5
entry=tests/test_sp_attention_cudagraph.py

cd "$upstream"
if [[ $(git rev-parse HEAD) != "$revision" ]]; then
    echo "Expected NanoEngine revision $revision in $PWD" >&2
    exit 1
fi
git diff --quiet HEAD -- tests/test_sp_attention_cudagraph.py nanodeploy || {
    echo "Upstream benchmark or runtime has local changes" >&2
    exit 1
}
test -f "$entry"
if [[ ${1:-} == --check && $# == 1 ]]; then
    echo "NanoEngine $revision: $PWD/$entry"
    exit 0
fi

length=${1:-32768}
if [[ $# -gt 1 || ! $length =~ ^[1-9][0-9]*$ ]] || (( length % 32 != 0 )); then
    echo "Usage: bash run_upstream_gqa.sh [--check | sequence-length divisible by 32]" >&2
    exit 2
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nnodes=1 --nproc-per-node=2 "$entry" \
    --attention_type GQA --attention_sp 2 --num_heads 16 --num_kv_heads 2 \
    --head_dim 256 --seq_len "$length" 2>&1 | tee "$log"
# The upstream script can return normally after a setup error. Each rank
# prints its event measurement once after completing the graph replay.
samples=$(grep -Eo 'Average time with CUDA Graph: [0-9]+[.][0-9]+ us' "$log" | wc -l) || true
if (( samples != 2 )); then
    echo "Expected two CUDA graph timings, received $samples" >&2
    exit 1
fi
