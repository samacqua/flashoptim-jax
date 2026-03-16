#!/usr/bin/env bash
set -euo pipefail

export TF_CPP_MIN_LOG_LEVEL=3
export TORCH_LOGS="-all"
export PYTHONWARNINGS=ignore
export NCCL_NET=Socket

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$ROOT_DIR/.." && pwd)"
source "$WORKSPACE_DIR/.venv/bin/activate"
mkdir -p "$ROOT_DIR/examples/out/gpt2"

DATA_DIR="${DATA_DIR:-$HOME/data/fineweb10b-gpt2}"
if [[ $# -gt 0 ]]; then
  DATA_DIR="$1"
  shift
fi

# Determine GPU count automatically if NPROC_PER_NODE is not set
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi &> /dev/null; then
    NPROC_PER_NODE=$(nvidia-smi -L | grep -c GPU || echo 1)
  else
    NPROC_PER_NODE=1
  fi
fi

if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  launcher=(python)
else
  launcher=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
fi


# for optimizer in lion sgd adamw; do
for optimizer in sgd; do
  for impl in flash reference; do
    if [[ "$optimizer" == "adamw" && "$impl" == "flash" ]]; then
      continue
    fi
    metrics_jsonl="$ROOT_DIR/examples/out/gpt2/${optimizer}-${impl}-torch.jsonl"
    echo "=== impl=$impl optimizer=$optimizer nproc_per_node=$NPROC_PER_NODE ==="
    "${launcher[@]}" "$ROOT_DIR/examples/train_gpt2_pretrain.py" \
      --data-dir "$DATA_DIR" \
      --impl "$impl" \
      --optimizer "$optimizer" \
      --metrics-jsonl "$metrics_jsonl" \
      --seed 5 \
      --batch-size 16 \
      "$@"
  done
done
