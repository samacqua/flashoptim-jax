"""Shared helpers for example and benchmark scripts."""

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import matplotlib.pyplot as plt

from flashoptim_jax import flash_adamw, flash_lion, flash_sgd, FlashOptimizer
from flashoptim import FlashAdamW as TorchFlashAdamW
from flashoptim import FlashLion as TorchFlashLion
from flashoptim import FlashSGD as TorchFlashSGD


# ── Optimizer wrappers ──────────────────────────────────────────────────────


class StepOptimizer(NamedTuple):
    """Minimal optimizer interface matching FlashOptimizer's (init, step)."""
    init: Callable[[Any], Any]
    step: Callable[[Any, Any, Any], tuple[Any, Any]]
    donate: bool = False


def wrap_optax(tx: optax.GradientTransformation) -> StepOptimizer:
    """Wrap a plain optax transform into a StepOptimizer."""
    def step_fn(params, state, grads):
        updates, new_state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), new_state

    return StepOptimizer(init=tx.init, step=step_fn)


def optax_with_master(tx: optax.GradientTransformation) -> StepOptimizer:
    """Wrap an optax transform with fp32 master params for low-precision training."""
    def init_fn(params):
        master = jax.tree_util.tree_map(
            lambda x: jnp.array(x, dtype=jnp.float32, copy=True), params,
        )
        return {"master": master, "opt": tx.init(master)}

    def step_fn(params, state, grads):
        grads_f32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), grads)
        updates, new_opt = tx.update(grads_f32, state["opt"], state["master"])
        new_master = optax.apply_updates(state["master"], updates)
        new_params = jax.tree_util.tree_map(
            lambda m, p: m.astype(p.dtype), new_master, params,
        )
        return new_params, {"master": new_master, "opt": new_opt}

    return StepOptimizer(init=init_fn, step=step_fn, donate=True)


def optax_with_master_params(tx: optax.GradientTransformation):
    """Wrap an optax transform with fp32 master params. Returns (init_fn, step_fn).

    Prefer ``optax_with_master`` for new code — it returns a ``StepOptimizer``
    namedtuple directly. This wrapper exists for backward compatibility with
    callers that destructure into two functions.
    """
    opt = optax_with_master(tx)
    return opt.init, opt.step


# ── Metrics / IO ────────────────────────────────────────────────────────────


def tree_nbytes(tree: Any) -> int:
    """Return the total size of all leaves in bytes."""
    return sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(tree))


def init_metrics_file(path: Path) -> Path:
    """Create or clear a JSONL metrics file."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def log_metrics(path: Path, record: dict) -> None:
    """Append one metrics record to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── LR schedules ────────────────────────────────────────────────────────────


def make_warmup_cosine_schedule(total_steps: int, warmup_steps: int, peak_lr: float):
    """Linear warmup for *warmup_steps*, then cosine decay to 0."""
    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup_lr = peak_lr * (step + 1.0) / warmup_steps if warmup_steps > 0 else peak_lr
        progress = jnp.clip((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0)
        cosine_lr = peak_lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)

    return schedule


# ── Benchmark constants ─────────────────────────────────────────────────────


OPTIMIZERS = ("sgd", "adamw", "lion")
WEIGHT_DTYPES = ("bf16", "fp16", "fp32")

FIVE_CONDITIONS = [
    ("torch_baseline", "PyTorch baseline", "#98df8a"),
    ("torch_flash", "PyTorch flash", "#ffbb78"),
    ("jax_baseline", "JAX baseline", "#1f77b4"),
    ("jax_flash", "JAX flash", "#d62728"),
]


def condition_label_map() -> dict[str, str]:
    return {condition: label for condition, label, _ in FIVE_CONDITIONS}


# ── Torch reference / mixed-precision optimizers ────────────────────────────


class AdamWMixedPrecision(torch.optim.Optimizer):
    """AdamW baseline that keeps fp32 master weights for low-precision params."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("closure is not supported")
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().to(torch.float32)
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["master_param"] = param.detach().to(torch.float32).clone()
                    state["exp_avg"] = torch.zeros_like(state["master_param"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master_param"])

                state["step"] += 1
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                master.mul_(1.0 - lr * weight_decay)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
                master.addcdiv_(exp_avg, denom, value=-step_size)
                param.copy_(master.to(param.dtype))


class SGDMixedPrecision(torch.optim.Optimizer):
    """SGD baseline that keeps fp32 master weights for low-precision params."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("closure is not supported")
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().to(torch.float32)
                state = self.state[param]
                if len(state) == 0:
                    state["master_param"] = param.detach().to(torch.float32).clone()
                    if momentum > 0.0:
                        state["momentum_buffer"] = torch.zeros_like(state["master_param"])

                master = state["master_param"]
                if weight_decay != 0.0:
                    grad = grad.add(master, alpha=weight_decay)
                if momentum > 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    grad = buf
                master.add_(grad, alpha=-lr)
                param.copy_(master.to(param.dtype))


class ReferenceLion(torch.optim.Optimizer):
    """Reference Lion optimizer implementation with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("closure is not supported")
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                grad_f32 = grad.detach().to(torch.float32)
                param_f32 = param.detach().to(torch.float32)
                if weight_decay != 0.0:
                    param_f32.mul_(1.0 - lr * weight_decay)
                update = exp_avg.lerp(grad_f32, 1.0 - beta1).sign_()
                param_f32.add_(update, alpha=-lr)
                exp_avg.lerp_(grad_f32, 1.0 - beta2)
                param.copy_(param_f32.to(param.dtype))


class LionMixedPrecision(torch.optim.Optimizer):
    """Lion baseline that keeps fp32 master weights for low-precision params."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("closure is not supported")
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    state["master_param"] = param.detach().to(torch.float32).clone()
                    state["exp_avg"] = torch.zeros_like(state["master_param"])
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                grad_f32 = grad.detach().to(torch.float32)
                if weight_decay != 0.0:
                    master.mul_(1.0 - lr * weight_decay)
                update = exp_avg.lerp(grad_f32, 1.0 - beta1).sign_()
                master.add_(update, alpha=-lr)
                exp_avg.lerp_(grad_f32, 1.0 - beta2)
                param.copy_(master.to(param.dtype))


# ── Benchmark helpers ───────────────────────────────────────────────────────


def is_low_precision(weights: str) -> bool:
    return weights != "fp32"


def jax_dtype_from_name(weights: str):
    dtype_map = {
        "bf16": jnp.bfloat16,
        "fp16": jnp.float16,
        "fp32": jnp.float32,
    }
    return dtype_map[weights]


def torch_dtype_from_name(weights: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[weights]


def parse_shape(text: str) -> tuple[int, ...]:
    value = text.strip().lower()
    if "x" in value:
        return tuple(int(part) for part in value.split("x"))
    return (int(value),)


def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {text}")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return sorted(values)[index]


def fmt_bytes(nbytes: int) -> str:
    if nbytes >= 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    return f"{nbytes / 1024:.1f} KB"


def estimate_state_nbytes(obj: Any) -> int:
    """Estimate tensor-like optimizer state size in bytes."""
    if isinstance(obj, torch.Tensor):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(estimate_state_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(estimate_state_nbytes(v) for v in obj)
    total = 0
    for name in ("data", "quantized", "scales"):
        value = getattr(obj, name, None)
        if isinstance(value, torch.Tensor):
            total += value.nbytes
    if total:
        return total
    if hasattr(obj, "nbytes"):
        return obj.nbytes
    return 0


def query_gpu_memory_mib(pid: int) -> int:
    """Read `nvidia-smi` memory usage for one process."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    total = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and int(parts[0]) == pid:
            total += int(parts[1])
    return total


# ── Plotting ────────────────────────────────────────────────────────────────


def save_memory_plot(
    results: list[dict],
    path: str,
    title: str,
    ordered_conditions: list[tuple[str, str, str]] | None = None,
) -> None:
    """Save a simple GPU-memory timeline plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    if ordered_conditions is None:
        for row in results:
            xs = [sample["elapsed_s"] for sample in row["memory_samples"]]
            ys = [sample["gpu_memory_mib"] for sample in row["memory_samples"]]
            ax.plot(xs, ys, label=row.get("label", row["condition"]), linewidth=1.5)
    else:
        for condition, label, color in ordered_conditions:
            row = next(item for item in results if item["condition"] == condition)
            xs = [sample["elapsed_s"] for sample in row["memory_samples"]]
            ys = [sample["gpu_memory_mib"] for sample in row["memory_samples"]]
            ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Plot saved to {path}")


# ── Optimizer factories ─────────────────────────────────────────────────────


def _master_weight_bits(weights: str) -> int | None:
    return None if weights == "fp32" else 24


def get_jax_flash_opt(
    optimizer: str,
    lr: float,
    weights: str,
    quantize: bool,
    fused: bool = True,
    **kwargs,
) -> FlashOptimizer:
    kwargs.setdefault("fused", fused)
    kwargs.setdefault("master_weight_bits", _master_weight_bits(weights))
    if optimizer == "sgd":
        kwargs.setdefault("momentum", 0.9)
    name2optim = {"adamw": flash_adamw, "sgd": flash_sgd, "lion": flash_lion}
    optim = name2optim[optimizer]
    return optim(lr, quantize=quantize, **kwargs)


def get_torch_flash_opt(
    optimizer: str,
    params: list,
    lr: float,
    weights: str,
    quantize: bool,
    fused: bool = True,
):
    mw_bits = _master_weight_bits(weights)
    name2optim = {"adamw": TorchFlashAdamW, "sgd": TorchFlashSGD, "lion": TorchFlashLion}
    optim = name2optim[optimizer]
    extra = {"momentum": 0.9} if optimizer == "sgd" else {}
    return optim(params, lr=lr, quantize=quantize, master_weight_bits=mw_bits, fused=fused, **extra)


def get_jax_baseline_opt(optimizer: str, lr: float, weights: str):
    if optimizer == "adamw":
        tx = optax.adamw(lr)
    elif optimizer == "sgd":
        tx = optax.sgd(lr, momentum=0.9)
    elif optimizer == "lion":
        tx = optax.lion(lr)
    else:
        raise ValueError(optimizer)
    if is_low_precision(weights):
        opt = optax_with_master(tx)
    else:
        opt = wrap_optax(tx)
    return opt.init, opt.step


def get_torch_baseline_opt(optimizer: str, params: list[torch.nn.Parameter], lr: float, weights: str):
    if optimizer == "adamw":
        if is_low_precision(weights):
            return AdamWMixedPrecision(params, lr=lr)
        return torch.optim.AdamW(params, lr=lr)
    if optimizer == "sgd":
        if is_low_precision(weights):
            return SGDMixedPrecision(params, lr=lr, momentum=0.9)
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if optimizer == "lion":
        if is_low_precision(weights):
            return LionMixedPrecision(params, lr=lr)
        return ReferenceLion(params, lr=lr)
    raise ValueError(optimizer)
