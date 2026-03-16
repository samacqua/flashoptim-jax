"""Shared helpers for example training scripts."""

import json
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


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
