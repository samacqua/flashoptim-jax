"""Shared helpers for PyTorch example training scripts."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn


# ── Metrics / IO ────────────────────────────────────────────────────────────


def init_metrics_file(path: Path) -> Path:
    """Create or clear a JSONL metrics file."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def log_metrics(path: Path | None, record: dict) -> None:
    """Append one metrics record to a JSONL file."""
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Optimizer helpers ───────────────────────────────────────────────────────


class ReferenceLion(torch.optim.Optimizer):
    """Simple reference Lion optimizer with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                grad_f32 = grad.detach().to(torch.float32)
                p_f32 = p.data.to(torch.float32)

                if wd != 0:
                    p_f32.mul_(1.0 - lr * wd)

                update = exp_avg.mul(b1).add(grad_f32, alpha=1.0 - b1).sign_()
                p_f32.add_(update, alpha=-lr)
                exp_avg.mul_(b2).add_(grad_f32, alpha=1.0 - b2)
                p.data.copy_(p_f32.to(p.dtype))
        return loss


class MasterWeightOptimizer:
    """Wrap any torch.optim.Optimizer with fp32 master weights for bf16 params.

    Mirrors the JAX ``optax_with_master`` wrapper: gradients are upcast to fp32,
    the base optimizer updates persistent fp32 master weights, and the bf16 model
    params are refreshed from the masters after each step.
    """

    def __init__(self, base_opt: torch.optim.Optimizer):
        self.base_opt = base_opt
        self._pairs: list[tuple[nn.Parameter, torch.Tensor]] = []
        for group in base_opt.param_groups:
            new_params = []
            for p in group["params"]:
                if p.dtype in (torch.bfloat16, torch.float16):
                    master = p.data.float().clone().detach().requires_grad_(True)
                    self._pairs.append((p, master))
                    new_params.append(master)
                else:
                    new_params.append(p)
            group["params"] = new_params

    @property
    def param_groups(self):
        return self.base_opt.param_groups

    def step(self):
        for bf16_p, master in self._pairs:
            if bf16_p.grad is not None:
                master.grad = bf16_p.grad.float()
        self.base_opt.step()
        for bf16_p, master in self._pairs:
            bf16_p.data.copy_(master.data.to(bf16_p.dtype))

    def zero_grad(self, set_to_none: bool = False):
        self.base_opt.zero_grad(set_to_none=set_to_none)
        for bf16_p, _ in self._pairs:
            bf16_p.grad = None


# ── LR schedule ─────────────────────────────────────────────────────────────


def warmup_cosine_lambda(total_steps: int, warmup_steps: int):
    """Return a LambdaLR-compatible function: linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ── Distributed helpers ─────────────────────────────────────────────────────


def setup_distributed() -> tuple[bool, int, int, int]:
    """Return (distributed, rank, local_rank, world_size)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    os.environ.setdefault("NCCL_NET", "Socket")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def seed_everything(seed: int) -> None:
    """Seed Python, PyTorch, and CUDA RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
