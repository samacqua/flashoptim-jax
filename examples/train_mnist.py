"""Train a CNN on MNIST using FlashOptim (JAX).

Mirrors the PyTorch example in `flashoptim_torch/examples/train_mnist.py`
with a single configurable optimizer/implementation per run.
"""
import argparse
import gzip
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from example_utils import (
    StepOptimizer,
    init_metrics_file,
    log_metrics,
    optax_with_master,
    tree_nbytes,
    wrap_optax,
)
from flashoptim_jax import (
    flash_adam,
    flash_adamw,
    flash_lion,
    flash_sgd,
)


OPTIMIZERS = ("sgd", "adamw", "lion")

def _download(filename: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    if not path.exists():
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}", path)
    return path


def load_mnist(data_dir: Path) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    missing = [filename for filename in files.values() if not (data_dir / filename).exists()]
    if missing:
        print(f"Local MNIST data missing in {data_dir}; downloading...")
    paths = {k: _download(v, data_dir) for k, v in files.items()}

    def read_images(p: Path) -> jnp.ndarray:
        with gzip.open(p, "rb") as f:
            _, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = f.read()
        return jnp.array(bytearray(data), dtype=jnp.float32).reshape(n, 1, rows, cols) / 255.0

    def read_labels(p: Path) -> jnp.ndarray:
        with gzip.open(p, "rb") as f:
            f.read(8)
            data = f.read()
        return jnp.array(bytearray(data), dtype=jnp.int32)

    return (
        read_images(paths["train_images"]),
        read_labels(paths["train_labels"]),
        read_images(paths["test_images"]),
        read_labels(paths["test_labels"]),
    )


# -- Model: same architecture as PyTorch MNISTNet --
# Conv2d(1,32,3,pad=1) -> BN -> ReLU -> MaxPool(2)
# Conv2d(32,64,3,pad=1) -> BN -> ReLU -> MaxPool(2)
# Linear(64*7*7, 128) -> ReLU -> Linear(128, 10)
#
# BatchNorm state kept in fp32 (running mean/var), params in bf16.


def _pytorch_linear_or_conv_init(
    key_w: jax.Array,
    key_b: jax.Array,
    weight_shape: tuple[int, ...],
    fan_in: int,
) -> tuple[jax.Array, jax.Array]:
    # PyTorch Conv/Linear defaults use kaiming_uniform_(a=sqrt(5)), which reduces to
    # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for both weights and biases.
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))
    weight = jax.random.uniform(
        key_w,
        weight_shape,
        dtype=jnp.float32,
        minval=-bound,
        maxval=bound,
    )
    bias = jax.random.uniform(
        key_b,
        (weight_shape[0],),
        dtype=jnp.float32,
        minval=-bound,
        maxval=bound,
    )
    return weight, bias

def init_params(key: jax.Array) -> dict:
    k1w, k1b, k2w, k2b, k3w, k3b, k4w, k4b = jax.random.split(key, 8)
    conv1_w, conv1_b = _pytorch_linear_or_conv_init(k1w, k1b, (32, 1, 3, 3), 1 * 3 * 3)
    conv2_w, conv2_b = _pytorch_linear_or_conv_init(k2w, k2b, (64, 32, 3, 3), 32 * 3 * 3)
    fc1_w, fc1_b = _pytorch_linear_or_conv_init(k3w, k3b, (128, 64 * 7 * 7), 64 * 7 * 7)
    fc2_w, fc2_b = _pytorch_linear_or_conv_init(k4w, k4b, (10, 128), 128)
    return {
        "conv1": {"w": conv1_w, "b": conv1_b},
        "bn1": { "scale": jnp.ones(32), "bias": jnp.zeros(32), },
        "conv2": { "w": conv2_w, "b": conv2_b, },
        "bn2": { "scale": jnp.ones(64), "bias": jnp.zeros(64), },
        "fc1": { "w": fc1_w.T, "b": fc1_b, },
        "fc2": { "w": fc2_w.T, "b": fc2_b, },
    }


def init_bn_state() -> dict:
    return {
        "bn1": {"mean": jnp.zeros(32), "var": jnp.ones(32)},
        "bn2": {"mean": jnp.zeros(64), "var": jnp.ones(64)},
    }


def batch_norm(x, scale, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    # x: (N, C, H, W) -- compute stats in f32 for numerical stability
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    if training:
        mean = jnp.mean(x, axis=(0, 2, 3))
        var = jnp.var(x, axis=(0, 2, 3))
        new_running_mean = (1 - momentum) * running_mean + momentum * mean
        new_running_var = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean
        var = running_var
        new_running_mean = running_mean
        new_running_var = running_var
    x_norm = (x - mean[None, :, None, None]) / jnp.sqrt(var[None, :, None, None] + eps)
    out = scale[None, :, None, None] * x_norm + bias[None, :, None, None]
    return out.astype(orig_dtype), new_running_mean, new_running_var


def conv2d(x, w, b, padding=1):
    # x: (N,Cin,H,W), w: (Cout,Cin,kH,kW)
    x = x.astype(w.dtype)
    out = jax.lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return out + b[None, :, None, None]


def max_pool2d(x, size=2):
    return -jax.lax.reduce_window(
        -x, jnp.inf, jax.lax.min, (1, 1, size, size), (1, 1, size, size), "VALID",
    )


def forward(params, bn_state, x, training):
    x = conv2d(x, params["conv1"]["w"], params["conv1"]["b"])
    x, bn1_mean, bn1_var = batch_norm(
        x, params["bn1"]["scale"], params["bn1"]["bias"],
        bn_state["bn1"]["mean"], bn_state["bn1"]["var"], training,
    )
    x = jax.nn.relu(x)
    x = max_pool2d(x)

    x = conv2d(x, params["conv2"]["w"], params["conv2"]["b"])
    x, bn2_mean, bn2_var = batch_norm(
        x, params["bn2"]["scale"], params["bn2"]["bias"],
        bn_state["bn2"]["mean"], bn_state["bn2"]["var"], training,
    )
    x = jax.nn.relu(x)
    x = max_pool2d(x)

    x = x.reshape(x.shape[0], -1)  # flatten
    x = x.astype(params["fc1"]["w"].dtype)
    x = jax.nn.relu(x @ params["fc1"]["w"] + params["fc1"]["b"])
    logits = x @ params["fc2"]["w"] + params["fc2"]["b"]

    new_bn_state = {
        "bn1": {"mean": bn1_mean, "var": bn1_var},
        "bn2": {"mean": bn2_mean, "var": bn2_var},
    }
    return logits, new_bn_state


def loss_fn(params, bn_state, images, labels):
    logits, new_bn_state = forward(params, bn_state, images, training=True)
    logits_f32 = logits.astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, labels).mean()
    return loss, new_bn_state


@jax.jit
def eval_batch(params, bn_state, images):
    logits, _ = forward(params, bn_state, images, training=False)
    return jnp.argmax(logits, axis=1)


def evaluate(params, bn_state, images, labels, batch_size):
    correct = 0
    n = images.shape[0]
    for i in range(0, n, batch_size):
        preds = eval_batch(params, bn_state, images[i:i + batch_size])
        correct += int(jnp.sum(preds == labels[i:i + batch_size]))
    return correct / n


def host_scalar(x: jax.Array) -> float:
    return float(jax.device_get(x))


def make_step_cosine_schedule(
    base_lr: float,
    total_steps: int,
    one_based_count: bool,
):
    base_lr = jnp.asarray(base_lr, dtype=jnp.float32)
    denom = jnp.asarray(max(1, total_steps - 1), dtype=jnp.float32)

    def schedule(count: jax.Array) -> jax.Array:
        count = jnp.asarray(count, dtype=jnp.int32)
        if one_based_count:
            count = count - jnp.asarray(1, dtype=jnp.int32)
        progress = jnp.clip(count.astype(jnp.float32), 0.0, denom) / denom
        return base_lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

    return schedule


def cast_params_dtype(params: dict, dtype: jnp.dtype) -> dict:
    out = {}
    for k, v in params.items():
        if k.startswith("bn"):
            out[k] = v
        else:
            out[k] = jax.tree_util.tree_map(lambda x: x.astype(dtype), v)
    return out


def make_optimizer(
    optimizer_name: str,
    lr: float,
    total_steps: int,
    flash: bool,
    fused: bool,
    param_dtype: str,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    master_weight_bits: int | None = 24,
    quantize: bool = True,
) -> tuple[StepOptimizer, bool]:
    use_low_precision_params = param_dtype != "fp32"
    schedule = make_step_cosine_schedule(lr, total_steps, one_based_count=flash)

    def _ref(tx):
        if use_low_precision_params:
            return optax_with_master(tx), use_low_precision_params
        return wrap_optax(tx), use_low_precision_params

    if optimizer_name == "sgd":
        if flash:
            optimizer = flash_sgd(
                learning_rate=schedule,
                momentum=momentum,
                weight_decay=weight_decay,
                quantize=quantize,
                master_weight_bits=master_weight_bits,
                fused=fused,
            )
            return StepOptimizer(init=optimizer.init, step=optimizer.step, donate=True), use_low_precision_params
        return _ref(optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=schedule, momentum=momentum),
        ))
    if optimizer_name == "adam":
        if flash:
            optimizer = flash_adam(
                learning_rate=schedule,
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay,
                quantize=quantize,
                master_weight_bits=master_weight_bits,
                fused=fused,
            )
            return StepOptimizer(init=optimizer.init, step=optimizer.step, donate=True), use_low_precision_params
        return _ref(optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.adam(learning_rate=schedule, b1=beta1, b2=beta2),
        ))
    if optimizer_name == "adamw":
        if flash:
            optimizer = flash_adamw(
                learning_rate=schedule,
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay,
                quantize=quantize,
                master_weight_bits=master_weight_bits,
                fused=fused,
            )
            return StepOptimizer(init=optimizer.init, step=optimizer.step, donate=True), use_low_precision_params
        return _ref(optax.adamw(
            learning_rate=schedule,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
        ))
    if optimizer_name == "lion":
        if flash:
            optimizer = flash_lion(
                learning_rate=schedule,
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay,
                quantize=quantize,
                master_weight_bits=master_weight_bits,
                fused=fused,
            )
            return StepOptimizer(init=optimizer.init, step=optimizer.step, donate=True), use_low_precision_params
        return _ref(optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.lion(learning_rate=schedule, b1=beta1, b2=beta2),
        ))
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train(
    optimizer_name: str,
    steps: int,
    batch_size: int,
    lr: float,
    data_dir: Path,
    flash: bool,
    fused: bool = True,
    seed: int = 0,
    param_dtype: str = "bf16",
    quantize: bool = True,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    master_weight_bits: int | None = 24,
    log_interval: int = 20,
    metrics_path: Path | None = None,
) -> dict:
    train_x, train_y, test_x, test_y = load_mnist(data_dir)
    key = jax.random.PRNGKey(seed)
    steps_per_epoch = (train_x.shape[0] + batch_size - 1) // batch_size

    params = init_params(key)
    bn_state = init_bn_state()

    dtype_map = {
        "bf16": jnp.bfloat16,
        "fp16": jnp.float16,
        "fp32": jnp.float32,
    }
    if param_dtype not in dtype_map:
        raise ValueError(f"Unknown param_dtype: {param_dtype}")
    params_dtype = dtype_map[param_dtype]

    num_params = sum(l.size for l in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {num_params:,}")

    optimizer, use_bf16_params = make_optimizer(
        optimizer_name,
        lr,
        steps,
        flash,
        fused,
        param_dtype,
        weight_decay,
        momentum,
        beta1,
        beta2,
        master_weight_bits,
        quantize,
    )
    lr_schedule = make_step_cosine_schedule(lr, steps, one_based_count=flash)
    if use_bf16_params:
        params = cast_params_dtype(params, params_dtype)
    opt_state = optimizer.init(params)

    param_bytes = tree_nbytes(params)
    state_bytes = tree_nbytes(opt_state)
    print(f"  Params: {param_bytes / 1024:.1f} KB  State: {state_bytes / 1024:.1f} KB  "
          f"Total: {(param_bytes + state_bytes) / 1024:.1f} KB")

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def train_step_impl(params, opt_state, bn_state, images, labels):
        (loss, new_bn_state), grads = grad_fn(params, bn_state, images, labels)
        new_params, new_opt_state = optimizer.step(params, opt_state, grads)
        return new_params, new_opt_state, new_bn_state, loss

    donate_argnums = (0, 1, 2) if optimizer.donate and quantize else ()
    train_step = jax.jit(train_step_impl, donate_argnums=donate_argnums)

    def _jax_bytes_in_use() -> int:
        stats = jax.local_devices()[0].memory_stats()
        return stats["bytes_in_use"] if stats else 0

    results = []
    total_time = 0.0
    best_acc = 0.0
    total_start = time.perf_counter()
    global_step = 0

    # Warm up the compiled train/eval paths before timing to avoid XLA timer noise
    # and compilation overhead in the first measured epoch.
    warmup_x = train_x[:batch_size]
    warmup_y = train_y[:batch_size]
    params, opt_state, bn_state, _ = train_step(
        params, opt_state, bn_state, warmup_x, warmup_y,
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), opt_state)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), bn_state)
    eval_batch(params, bn_state, warmup_x).block_until_ready()

    warmup_peak_bytes = jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0)
    steady_peak_bytes = 0

    epoch = 0
    while global_step < steps:
        epoch += 1
        perm_key, key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, train_x.shape[0])
        train_x_shuf = train_x[perm]
        train_y_shuf = train_y[perm]

        train_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        train_count = 0
        interval_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        interval_count = 0
        t0 = time.perf_counter()

        for i in range(0, train_x.shape[0], batch_size):
            if global_step >= steps:
                break
            bx = train_x_shuf[i:i + batch_size]
            by = train_y_shuf[i:i + batch_size]
            params, opt_state, bn_state, loss_val = train_step(params, opt_state, bn_state, bx, by)
            batch_count = bx.shape[0]
            batch_weight = jnp.asarray(batch_count, dtype=jnp.float32)
            train_loss_sum = train_loss_sum + loss_val * batch_weight
            train_count += batch_count
            interval_loss_sum = interval_loss_sum + loss_val * batch_weight
            interval_count += batch_count
            global_step += 1

            if metrics_path is not None and (
                global_step % log_interval == 0
                or global_step == 1
                or global_step == steps
            ):
                avg_loss = host_scalar(interval_loss_sum / interval_count)
                current_lr = host_scalar(
                    lr_schedule(
                        jnp.asarray(global_step if flash else global_step - 1, dtype=jnp.int32)
                    )
                )
                elapsed = time.perf_counter() - total_start
                print(
                    f"  step={global_step:05d}/{steps:05d} epoch={epoch:02d} "
                    f"loss={avg_loss:.4f} lr={current_lr:.3e}"
                )
                log_metrics(
                    metrics_path,
                    {
                        "event": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "max_steps": steps,
                        "loss": avg_loss,
                        "lr": current_lr,
                        "elapsed_s": elapsed,
                    },
                )
                interval_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                interval_count = 0

        # Block for accurate timing
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
        epoch_time = time.perf_counter() - t0

        if epoch > 1:
            total_time += epoch_time
            steady_peak_bytes = max(steady_peak_bytes, _jax_bytes_in_use())

        train_loss = host_scalar(train_loss_sum / train_count)
        acc = evaluate(params, bn_state, test_x, test_y, batch_size)
        best_acc = max(best_acc, acc)
        elapsed = time.perf_counter() - total_start
        print(
            f"  epoch={epoch:02d} step={global_step:05d}/{steps:05d} "
            f"loss={train_loss:.4f} acc={acc:.4f} time={epoch_time:.2f}s"
        )
        results.append({"epoch": epoch, "acc": acc, "loss": train_loss, "time": epoch_time})
        if metrics_path is not None:
            log_metrics(
                metrics_path,
                {
                    "event": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "max_steps": steps,
                    "train_loss": train_loss,
                    "epoch_time_s": epoch_time,
                    "elapsed_s": elapsed,
                },
            )
            log_metrics(
                metrics_path,
                {
                    "event": "eval",
                    "epoch": epoch,
                    "step": global_step,
                    "max_steps": steps,
                    "eval_acc": acc,
                    "best_eval_acc": best_acc,
                    "elapsed_s": elapsed,
                },
            )

    if steady_peak_bytes == 0:
        steady_peak_bytes = _jax_bytes_in_use()

    final_acc = evaluate(params, bn_state, test_x, test_y, batch_size)
    print(f"  Final accuracy: {final_acc:.2%}  Total time (epochs 2+): {total_time:.2f}s")

    return {
        "optimizer": optimizer_name,
        "impl": "flash" if flash else "reference",
        "flash": flash,
        "quantize": quantize,
        "final_acc": final_acc,
        "total_time": total_time,
        "param_bytes": param_bytes,
        "state_bytes": state_bytes,
        "warmup_peak_bytes": warmup_peak_bytes,
        "steady_peak_bytes": steady_peak_bytes,
        "epochs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST CNN with FlashOptim (JAX) or Optax."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--impl", choices=["flash", "reference"], default="flash")
    parser.add_argument("--optimizer", choices=list(OPTIMIZERS), default="adamw")
    parser.add_argument("--steps", type=int, default=2_350)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--master-weight-bits", type=int, default=24)
    parser.add_argument("--no-fused", action="store_true", help="Disable fused Pallas kernels (use pure JAX).")
    parser.add_argument("--no-quantize", action="store_true", help="Disable state quantization.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("metrics_train_mnist_jax.jsonl"),
        help="Path to write train/eval metrics JSONL.",
    )
    args = parser.parse_args()
    if args.lr is None:
        if args.optimizer == "sgd":
            args.lr = 1e-2
        elif args.optimizer == "lion":
            args.lr = 1e-4
        else:
            args.lr = 1e-3
    if args.weight_decay is None:
        args.weight_decay = 0.01 if args.optimizer == "adamw" else 0.0
    if args.impl == "reference":
        args.master_weight_bits = None
    elif args.master_weight_bits == 0:
        args.master_weight_bits = None
    return args


def main() -> None:
    args = parse_args()
    metrics_path = init_metrics_file(args.metrics_jsonl)
    flash = args.impl == "flash"
    param_dtype = "bf16"

    print(f"backend={jax.default_backend()} devices={jax.device_count()}")
    print(f"impl={args.impl} optimizer={args.optimizer}")
    if args.optimizer == "sgd":
        print(f"lr={args.lr:.2e} wd={args.weight_decay} momentum={args.momentum}")
    else:
        print(f"lr={args.lr:.2e} wd={args.weight_decay} betas=({args.beta1},{args.beta2})")
    if args.impl == "flash":
        print(f"master_weight_bits={args.master_weight_bits}")
    print()
    log_metrics(
        metrics_path,
        {
            "event": "config",
            "backend": jax.default_backend(),
            "devices": jax.device_count(),
            "impl": args.impl,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "master_weight_bits": args.master_weight_bits,
            "data_dir": str(args.data_dir),
            "seed": args.seed,
            "log_interval": args.log_interval,
        },
    )
    result = train(
        args.optimizer,
        args.steps,
        args.batch_size,
        args.lr,
        args.data_dir,
        flash=flash,
        fused=not args.no_fused,
        seed=args.seed,
        param_dtype=param_dtype,
        quantize=not args.no_quantize,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        master_weight_bits=args.master_weight_bits,
        log_interval=args.log_interval,
        metrics_path=metrics_path,
    )
    log_metrics(
        metrics_path,
        {
            "event": "training_complete",
            "impl": args.impl,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "final_acc": float(result["final_acc"]),
            "total_time_s": float(result["total_time"]),
            "param_bytes": int(result["param_bytes"]),
            "state_bytes": int(result["state_bytes"]),
        },
    )


if __name__ == "__main__":
    main()
