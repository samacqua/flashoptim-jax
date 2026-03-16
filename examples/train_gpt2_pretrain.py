"""Pretrain GPT-2 (124M) with FlashOptim (JAX) or Optax reference.

This follows the paper's GPT-2 pretraining recipe:
  - GPT-2 124M architecture: 12 layers, 12 heads, hidden size 768, context 1024
  - FineWeb10B GPT-2-tokenized dataset (`kjj0/fineweb10B-gpt2`)
  - 20,000 optimization steps
  - warmup for first 700 steps, then cosine decay to 0
  - BF16 activations for both reference and FlashOptim runs
  - gradient clipping at global norm 1.0
  - weight decay 0.1 on 2D parameters only (matrices + embeddings)

Downloading data:
  1) Accept dataset terms and authenticate with HuggingFace.
  2) Run this script with `--download` to fetch required `.bin` shards into `--data-dir`.
  3) Or pre-download manually and point `--data-dir` to the folder containing:
       fineweb_train_000001.bin ... fineweb_train_000103.bin, fineweb_val_000000.bin

Usage:
  python examples/train_gpt2_pretrain.py --data-dir ~/data/fineweb10b-gpt2 --download
  NCCL_NET=Socket python examples/train_gpt2_pretrain.py --data-dir ~/data/fineweb10b-gpt2
  python examples/train_gpt2_pretrain.py --data-dir ~/data/fineweb10b-gpt2
  CUDA_VISIBLE_DEVICES=0 python examples/train_gpt2_pretrain.py --data-dir ~/data/fineweb10b-gpt2

The script uses all local JAX devices with data parallel `pmap`.
Global batch size (`--batch-size`) must be divisible by `jax.local_device_count()`.
"""

import argparse
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from huggingface_hub import snapshot_download

from benchmark_utils import OPTIMIZERS, get_jax_flash_opt
from example_utils import init_metrics_file, log_metrics, make_warmup_cosine_schedule, optax_with_master, tree_nbytes
from flash_attn_jax import flash_mha
from flashoptim_jax import FlashOptimizer

# =====
# opt
# =====

def _wrap_flash_with_clip(tx: FlashOptimizer, grad_clip: float) -> FlashOptimizer:
    """Wrap a FlashOptimizer with global-norm gradient clipping."""
    clip_tx = optax.clip_by_global_norm(grad_clip)

    def init_fn(params):
        """Initialize clip and optimizer state."""
        return {"clip": clip_tx.init(params), "opt": tx.init(params)}

    def step_fn(params, state, grads):
        """Clip grads, apply one optimizer step, and return new state."""
        grads, new_clip_state = clip_tx.update(grads, state["clip"], params)
        params, new_opt_state = tx.step(params, state["opt"], grads)
        return params, {"clip": new_clip_state, "opt": new_opt_state}

    return FlashOptimizer(init=init_fn, step=step_fn)

def _is_norm_or_bias(path) -> bool:
    """True for layer-norm and bias leaves (which should skip weight decay)."""
    path_str = "/".join(str(p) for p in path)
    return "ln" in path_str or path_str.endswith("bias")


def make_decay_mask(params: Any) -> Any:
    """Weight-decay mask: True for weight matrices, False for norms/biases."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _leaf: not _is_norm_or_bias(path), params
    )

def create_optimizer(args: argparse.Namespace, schedule, params: Any):
    """Build the configured FlashOptim or Optax optimizer."""
    if args.impl == "flash":
        no_decay_matcher = lambda path, _leaf: _is_norm_or_bias(path)
        flash_kwargs = {
            "weight_decay": args.weight_decay,
            "master_weight_bits": args.master_weight_bits,
            "param_groups": [{"params": no_decay_matcher, "weight_decay": 0.0}],
        }
        if args.optimizer == "sgd":
            flash_kwargs["momentum"] = args.momentum
        else:
            flash_kwargs["b1"] = args.beta1
            flash_kwargs["b2"] = args.beta2
        base = get_jax_flash_opt(args.optimizer, schedule, "bf16", quantize=True, **flash_kwargs)
        return _wrap_flash_with_clip(base, args.grad_clip)

    decay_mask = make_decay_mask(params)
    if args.optimizer == "sgd":
        base = optax.chain(
            optax.add_decayed_weights(args.weight_decay, mask=decay_mask),
            optax.sgd(learning_rate=schedule, momentum=args.momentum, nesterov=False),
        )
    elif args.optimizer == "adamw":
        base = optax.adamw(
            learning_rate=schedule,
            b1=args.beta1,
            b2=args.beta2,
            weight_decay=args.weight_decay,
            mask=decay_mask,
        )
    else:
        base = optax.lion(
            learning_rate=schedule,
            b1=args.beta1,
            b2=args.beta2,
            weight_decay=args.weight_decay,
            mask=decay_mask,
        )
    return optax_with_master(
        optax.chain(optax.clip_by_global_norm(args.grad_clip), base)
    )


# ======
# data
# ======

class TokenShardSampler:
    """Random batch sampler over FineWeb binary token shards."""

    def __init__(self, files: list[Path], seed: int, batches_per_shard: int = 256) -> None:
        """Load one shard at a time to keep random reads page-cache friendly."""
        self.files = files
        self.rng = np.random.default_rng(seed)
        self.batches_per_shard = batches_per_shard
        self.current_shard_idx = -1
        self.current_shard = np.empty((0,), dtype=np.uint16)
        self.remaining_batches = 0

    def sample(self, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample a batch of next-token prediction sequences."""
        if self.remaining_batches == 0:
            shard_idx = int(self.rng.integers(0, len(self.files)))
            if shard_idx != self.current_shard_idx:
                self.current_shard = np.fromfile(self.files[shard_idx], dtype=np.uint16)
                self.current_shard_idx = shard_idx
            self.remaining_batches = self.batches_per_shard
        self.remaining_batches -= 1
        shard = self.current_shard
        starts = self.rng.integers(0, shard.size - seq_len - 1, size=batch_size)
        idx = starts[:, None] + np.arange(seq_len + 1)[None, :]
        chunks = np.asarray(shard[idx], dtype=np.int32)
        return chunks[:, :-1], chunks[:, 1:]


def ensure_data(
    data_dir: Path,
    num_train_shards: int,
    download: bool,
    repo_id: str,
) -> tuple[list[Path], Path]:
    """Ensure FineWeb shards exist locally and download if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_names = [f"fineweb_train_{i:06d}.bin" for i in range(1, num_train_shards + 1)]
    val_name = "fineweb_val_000000.bin"
    needed = train_names + [val_name]
    missing = [name for name in needed if not (data_dir / name).exists()]
    if missing:
        print(
            f"Local GPT-2 data missing in {data_dir}; downloading {len(missing)} shard(s)..."
        )
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=needed + ["README.md"],
            local_dir=str(data_dir),
        )
    missing = [name for name in needed if not (data_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} dataset shard(s) in {data_dir} after download."
        )
    train_files = [data_dir / name for name in train_names]
    return train_files, data_dir / val_name

def shard_batch(
    x_np: np.ndarray,
    y_np: np.ndarray,
    num_devices: int,
    local_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a batch into per-device shards for `pmap`."""
    x_sharded = x_np.reshape(num_devices, local_batch_size, x_np.shape[1])
    y_sharded = y_np.reshape(num_devices, local_batch_size, y_np.shape[1])
    return x_sharded, y_sharded


# =====
# model
# =====


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Apply layer norm in fp32 and cast back to input dtype."""
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.var(x_f32, axis=-1, keepdims=True)
    out = scale * (x_f32 - mean) / jnp.sqrt(var + eps) + bias
    return out.astype(x.dtype)


def init_params(
    key: jax.Array,
    vocab_size: int,
    seq_len: int,
    n_layer: int,
    n_embd: int,
) -> dict[str, Any]:
    """Initialize GPT-2 parameters with the paper's scaling rules."""
    ff_dim = 4 * n_embd
    keys = jax.random.split(key, 4 + n_layer * 8)
    idx = 0
    wte = jax.random.normal(keys[idx], (vocab_size, n_embd), dtype=jnp.float32) * 0.02
    idx += 1
    wpe = jax.random.normal(keys[idx], (seq_len, n_embd), dtype=jnp.float32) * 0.02
    idx += 1
    ln_f_scale = jnp.ones((n_embd,), dtype=jnp.float32)
    ln_f_bias = jnp.zeros((n_embd,), dtype=jnp.float32)
    idx += 2

    proj_std = 0.02 / math.sqrt(2.0 * n_layer)
    layer_list = []
    for _ in range(n_layer):
        layer_keys = keys[idx : idx + 8]
        idx += 8
        layer_list.append(
            {
                "ln1_scale": jnp.ones((n_embd,), dtype=jnp.float32),
                "ln1_bias": jnp.zeros((n_embd,), dtype=jnp.float32),
                "wq": jax.random.normal(layer_keys[0], (n_embd, n_embd), dtype=jnp.float32)
                * 0.02,
                "wk": jax.random.normal(layer_keys[1], (n_embd, n_embd), dtype=jnp.float32)
                * 0.02,
                "wv": jax.random.normal(layer_keys[2], (n_embd, n_embd), dtype=jnp.float32)
                * 0.02,
                "wo": jax.random.normal(layer_keys[3], (n_embd, n_embd), dtype=jnp.float32)
                * proj_std,
                "ln2_scale": jnp.ones((n_embd,), dtype=jnp.float32),
                "ln2_bias": jnp.zeros((n_embd,), dtype=jnp.float32),
                "w1": jax.random.normal(layer_keys[4], (n_embd, ff_dim), dtype=jnp.float32)
                * 0.02,
                "w2": jax.random.normal(layer_keys[5], (ff_dim, n_embd), dtype=jnp.float32)
                * proj_std,
            }
        )

    stacked_layers = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *layer_list)
    return {
        "wte": wte,
        "wpe": wpe,
        "layers": stacked_layers,
        "ln_f_scale": ln_f_scale,
        "ln_f_bias": ln_f_bias,
    }


def cast_params_for_mixed_precision(params: dict[str, Any]) -> dict[str, Any]:
    """Keep norm and bias params in fp32, cast the rest to bf16."""
    def cast_leaf(path, value):
        """Cast one parameter leaf based on its path."""
        path_str = "/".join(str(p) for p in path)
        if "ln" in path_str or path_str.endswith("bias"):
            return value.astype(jnp.float32)
        return value.astype(jnp.bfloat16)

    return jax.tree_util.tree_map_with_path(cast_leaf, params)


def forward(params: dict[str, Any], tokens: jax.Array, n_head: int, n_layer: int) -> jax.Array:
    """Run a GPT-2 forward pass with FlashAttention."""
    B, T = tokens.shape
    D = params["wte"].shape[1]
    head_dim = D // n_head
    x = params["wte"][tokens] + params["wpe"][:T]
    x = x.astype(jnp.bfloat16)

    resid_scale = 1.0 / math.sqrt(2.0 * n_layer)
    layers = params["layers"]
    for i in range(n_layer):
        layer = jax.tree.map(lambda a: a[i], layers)
        h = layer_norm(x, layer["ln1_scale"], layer["ln1_bias"])
        h_qkv = h.astype(layer["wq"].dtype)
        q = (h_qkv @ layer["wq"]).reshape(B, T, n_head, head_dim)
        k = (h_qkv @ layer["wk"]).reshape(B, T, n_head, head_dim)
        v = (h_qkv @ layer["wv"]).reshape(B, T, n_head, head_dim)
        out = flash_mha(q, k, v, is_causal=True).reshape(B, T, D)
        x = x + ((out @ layer["wo"]) * resid_scale).astype(x.dtype)

        h2 = layer_norm(x, layer["ln2_scale"], layer["ln2_bias"])
        h2_f = h2.astype(layer["w1"].dtype)
        ffn = jax.nn.gelu(h2_f @ layer["w1"], approximate=True) @ layer["w2"]
        x = x + (ffn * resid_scale).astype(x.dtype)

    x = layer_norm(x, params["ln_f_scale"], params["ln_f_bias"])
    logits = x.astype(params["wte"].dtype) @ params["wte"].T
    return logits.astype(jnp.float32)


def host_scalar(x: Any) -> float:
    """Fetch a replicated scalar from device 0 to the host."""
    return float(np.asarray(jax.device_get(x[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain GPT-2 (124M) with FlashOptim (JAX) or Optax."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default="kjj0/fineweb10B-gpt2")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-train-shards", type=int, default=103)
    parser.add_argument("--impl", choices=["flash", "reference"], default="flash")
    parser.add_argument("--optimizer", choices=list(OPTIMIZERS), default="adamw")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=700)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--master-weight-bits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches-per-shard", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("metrics_train_gpt2_pretrain_jax.jsonl"),
        help="Path to write train/eval metrics JSONL.",
    )
    args = parser.parse_args()
    if args.lr is None:
        if args.optimizer == "sgd":
            args.lr = 1e-3
        elif args.optimizer == "adamw":
            args.lr = 6e-4
        else:
            args.lr = 2e-4
    if args.impl == "reference":
        args.master_weight_bits = None
    elif args.master_weight_bits == 0:
        args.master_weight_bits = None
    return args


def main() -> None:
    """Run GPT-2 pretraining and periodic evaluation."""
    args = parse_args()
    num_devices = jax.local_device_count()
    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by local device count {num_devices}."
        )
    local_batch_size = args.batch_size // num_devices
    metrics_path = init_metrics_file(args.metrics_jsonl)

    train_files, val_file = ensure_data(
        data_dir=args.data_dir,
        num_train_shards=args.num_train_shards,
        download=args.download,
        repo_id=args.repo_id,
    )
    train_sampler = TokenShardSampler(
        train_files, seed=args.seed, batches_per_shard=args.batches_per_shard
    )
    val_sampler = TokenShardSampler([val_file], seed=12345, batches_per_shard=args.batches_per_shard)
    rng = jax.random.PRNGKey(args.seed)
    params = init_params(
        rng,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
    )
    if args.impl in ("flash", "reference"):
        params = cast_params_for_mixed_precision(params)

    lr_steps = 100_000  # args.steps
    schedule = make_warmup_cosine_schedule(lr_steps, args.warmup_steps, args.lr)
    tx = create_optimizer(args, schedule, params)
    opt_state = tx.init(params)
    param_bytes = tree_nbytes(params)
    state_bytes = tree_nbytes(opt_state)
    nparams = sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params))

    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(np.array(devices), "batch")
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    def _replicate(tree):
        """Replicate a pytree across local devices."""
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding),
            tree,
        )
    params = _replicate(params)
    opt_state = _replicate(opt_state)

    def train_step(params, opt_state, x, y, step):
        """Compute one distributed training step."""
        def loss_fn(p):
            """Return mean next-token cross-entropy loss."""
            logits = forward(p, x, args.n_head, args.n_layer)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_params, new_opt_state = tx.step(params, opt_state, grads)
        loss = jax.lax.pmean(loss, axis_name="batch")
        lr = schedule(step)
        return new_params, new_opt_state, loss, lr

    def eval_step(params, x, y):
        """Compute replicated validation loss for one batch."""
        logits = forward(params, x, args.n_head, args.n_layer)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return jax.lax.pmean(loss, axis_name="batch")

    p_train_step = jax.pmap(train_step, axis_name="batch", in_axes=(0, 0, 0, 0, None), donate_argnums=(0, 1, 2, 3))
    p_eval_step = jax.pmap(eval_step, axis_name="batch", in_axes=(0, 0, 0))
    print(f"backend={jax.default_backend()} devices={jax.device_count()} local_devices={num_devices}")
    print(f"impl={args.impl} optimizer={args.optimizer}")
    print(f"steps={args.steps} warmup_steps={args.warmup_steps}")
    print(
        f"batch_size={args.batch_size} local_batch_size={local_batch_size} seq_len={args.seq_len}"
    )
    if args.optimizer == "sgd":
        print(f"lr={args.lr:.2e} wd={args.weight_decay} momentum={args.momentum}")
    else:
        print(f"lr={args.lr:.2e} wd={args.weight_decay} betas=({args.beta1},{args.beta2})")
    if args.impl == "flash":
        print(f"master_weight_bits={args.master_weight_bits}")
    print(f"parameters={nparams:,}")
    print(
        f"params={param_bytes / 2**20:.2f} MiB  "
        f"state={state_bytes / 2**20:.2f} MiB  "
        f"total={(param_bytes + state_bytes) / 2**20:.2f} MiB"
    )
    print()
    log_metrics(
        metrics_path,
        {
            "event": "config",
            "backend": jax.default_backend(),
            "devices": jax.device_count(),
            "local_devices": num_devices,
            "impl": args.impl,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "eval_interval": args.eval_interval,
            "eval_batches": args.eval_batches,
            "batch_size": args.batch_size,
            "local_batch_size": local_batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "grad_clip": args.grad_clip,
            "master_weight_bits": args.master_weight_bits,
            "num_params": nparams,
            "param_bytes": param_bytes,
            "state_bytes": state_bytes,
        },
    )

    start = time.perf_counter()
    prev_log_time = start
    prev_log_step = 0
    loss_accum = []
    for step in range(1, args.steps + 1):
        x_np, y_np = train_sampler.sample(args.batch_size, args.seq_len)
        x_np, y_np = shard_batch(
            x_np,
            y_np,
            num_devices=num_devices,
            local_batch_size=local_batch_size,
        )
        x = jnp.asarray(x_np, dtype=jnp.int32)
        y = jnp.asarray(y_np, dtype=jnp.int32)
        step_arr = jnp.asarray(step - 1, dtype=jnp.int32)
        params, opt_state, loss, lr = p_train_step(params, opt_state, x, y, step_arr)
        loss_accum.append(loss[0])

        if step % args.log_interval == 0 or step == 1:
            avg_loss = float(jnp.mean(jnp.stack(loss_accum)))
            loss_accum = []
            now = time.perf_counter()
            elapsed = now - start
            toks = step * args.batch_size * args.seq_len
            toks_per_s = toks / elapsed
            interval_toks = (step - prev_log_step) * args.batch_size * args.seq_len
            interval_time = now - prev_log_time
            interval_toks_per_s = interval_toks / interval_time if interval_time > 0 else 0
            prev_log_time = now
            prev_log_step = step
            print(
                f"step={step:05d}/{args.steps:05d} "
                f"loss={avg_loss:.4f} lr={host_scalar(lr):.3e} "
                f"toks/s={toks_per_s:,.0f} interval={interval_toks_per_s:,.0f}"
            )
            log_metrics(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "loss": avg_loss,
                    "lr": host_scalar(lr),
                    "tokens_per_s": toks_per_s,
                    "interval_tokens_per_s": interval_toks_per_s,
                    "elapsed_s": elapsed,
                },
            )

        if step % args.eval_interval == 0 or step == args.steps:
            losses = []
            for _ in range(args.eval_batches):
                x_np, y_np = val_sampler.sample(args.batch_size, args.seq_len)
                x_np, y_np = shard_batch(
                    x_np,
                    y_np,
                    num_devices=num_devices,
                    local_batch_size=local_batch_size,
                )
                losses.append(
                    host_scalar(
                        p_eval_step(
                            params,
                            jnp.asarray(x_np, dtype=jnp.int32),
                            jnp.asarray(y_np, dtype=jnp.int32),
                        )
                    )
                )
            val_loss = float(np.mean(losses))
            ppl = math.exp(min(val_loss, 20.0))
            print(f"  eval step={step:05d} val_loss={val_loss:.4f} val_ppl={ppl:.2f}")
            log_metrics(
                metrics_path,
                {
                    "event": "eval",
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": ppl,
                },
            )

    log_metrics(metrics_path, {"event": "training_complete", "steps": args.steps})


if __name__ == "__main__":
    main()
