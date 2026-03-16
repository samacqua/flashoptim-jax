#!/usr/bin/env bash
set -euo pipefail

export TF_CPP_MIN_LOG_LEVEL=3
export TORCH_LOGS="-all"
export PYTHONWARNINGS=ignore
export NCCL_NET=Socket

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_ROOT_DIR="$ROOT_DIR/flashoptim_torch"
source "$ROOT_DIR/.venv/bin/activate"

usage() {
  cat >&2 <<'EOF'
Usage: examples/run_benchmark.sh <gpt2|mnist|imagenette> [options] [extra args...]

Options:
  --jax-only     Only run JAX experiments
  --torch-only   Only run PyTorch experiments
  --data-dir DIR Override default data directory

Examples:
  examples/run_benchmark.sh gpt2                       # JAX only (no Torch GPT-2)
  examples/run_benchmark.sh mnist                       # JAX + Torch, 100 epochs
  examples/run_benchmark.sh mnist --jax-only            # JAX only
  examples/run_benchmark.sh imagenette                  # JAX + Torch, 34 epochs
  examples/run_benchmark.sh imagenette --jax-only       # JAX only
EOF
  exit 1
}

[[ $# -eq 0 ]] && usage

benchmark="$1"; shift

run_torch=true
run_jax=true
data_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jax-only)   run_torch=false; shift ;;
    --torch-only) run_jax=false; shift ;;
    --data-dir)   data_dir="$2"; shift 2 ;;
    *)            extra_args+=("$1"); shift ;;
  esac
done

OUTDIR="$ROOT_DIR/examples/out-final"

case "$benchmark" in
  gpt2|gpt)
    run_torch=false  # GPT-2 is JAX only
    jax_script="$ROOT_DIR/examples/train_gpt2_pretrain.py"
    subdir="gpt2"
    : "${data_dir:=${DATA_DIR:-$HOME/data/fineweb10b-gpt2}}"
    optimizers=(adamw sgd lion)
    impls=(flash reference)
    skip_flash_adamw=true
    jax_args=(--data-dir "$data_dir" --batch-size 16 --steps 10000)
    torch_args=()
    torch_script=""
    torch_launcher=(python)
    ;;
  mnist)
    jax_script="$ROOT_DIR/examples/train_mnist.py"
    torch_script="$TORCH_ROOT_DIR/examples/train_mnist.py"
    subdir="mnist"
    : "${data_dir:=$ROOT_DIR/data/mnist}"
    optimizers=(adamw sgd lion)
    impls=(flash reference)
    skip_flash_adamw=false
    jax_args=(--data-dir "$data_dir" --steps 10000)
    torch_args=(--data-dir "$data_dir" --steps 10000)
    torch_launcher=(python)
    ;;
  imagenette|imagenet)
    jax_script="$ROOT_DIR/examples/train_imagenet.py"
    torch_script="$TORCH_ROOT_DIR/examples/train_imagenet.py"
    subdir="imagenet"
    : "${data_dir:=$ROOT_DIR/data/imagenet/imagenette2-320}"
    optimizers=(adamw sgd lion)
    impls=(flash reference)
    skip_flash_adamw=false
    jax_args=(--data-dir "$data_dir" --steps 5000 --warmup-steps 740 --batch-size 64 --seed 0 --log-interval 10)
    torch_args=(--data-dir "$data_dir" --steps 5000 --warmup-steps 740 --batch-size 64 --seed 0 --log-interval 10)
    torch_launcher=(python)
    ;;
  *)
    echo "Unknown benchmark: $benchmark" >&2
    usage
    ;;
esac

outdir="$OUTDIR/$subdir"
mkdir -p "$outdir"

total=$(( ${#optimizers[@]} * ${#impls[@]} ))
n=0

for optimizer in "${optimizers[@]}"; do
  for impl in "${impls[@]}"; do
    if [[ "$skip_flash_adamw" == true && "$optimizer" == "adamw" && "$impl" == "flash" ]]; then
      continue
    fi
    n=$((n + 1))

    if [[ "$run_jax" == true ]]; then
      out="$outdir/${optimizer}-${impl}-jax.jsonl"
      echo "=== [$n/$total] jax ${optimizer} ${impl} ==="
      python "$jax_script" \
        --impl "$impl" --optimizer "$optimizer" \
        --metrics-jsonl "$out" \
        "${jax_args[@]}" "${extra_args[@]}"
    fi

    if [[ "$run_torch" == true && -n "$torch_script" ]]; then
      out="$outdir/${optimizer}-${impl}-torch.jsonl"
      echo "=== [$n/$total] torch ${optimizer} ${impl} ==="
      "${torch_launcher[@]}" "$torch_script" \
        --impl "$impl" --optimizer "$optimizer" \
        --metrics-jsonl "$out" \
        "${torch_args[@]}" "${extra_args[@]}"
    fi
  done
done

echo ""
echo "=== Done. Output in $outdir ==="
ls -lh "$outdir"/*.jsonl
