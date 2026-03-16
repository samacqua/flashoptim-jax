#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

SHAPES="${SHAPES:-1024x1024,2048x2048,4096x4096,8192x8192,16384x16384}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/examples/out/benchmark_step_square}"
mkdir -p "$OUT_DIR"

IFS=',' read -r -a optimizers <<< "${OPTIMIZERS:-adamw,sgd,lion}"
IFS=',' read -r -a weights <<< "${WEIGHTS:-bf16,fp16,fp32}"
failures=()

export TORCH_LOGS="-all"
export PYTHONWARNINGS="ignore"
# XLA_FLAGS='--xla_allow_excess_precision=false'

for optimizer in "${optimizers[@]}"; do
  for weight in "${weights[@]}"; do
    log_file="$OUT_DIR/${optimizer}-${weight}.log"
    echo "=== optimizer=$optimizer weights=$weight ==="
    if ! python "$ROOT_DIR/examples/benchmark_step.py" \
      --optimizer "$optimizer" \
      --weights "$weight" \
      --shapes "$SHAPES" \
      "$@" 2>&1 | tee "$log_file"; then
      failures+=("${optimizer}/${weight}")
    fi
  done
done

if ((${#failures[@]})); then
  printf '\nFailed runs:\n' >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
