"""Pretrain GPT-2 (124M) with FlashOptim or reference optimizers.

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
  python examples/torch/train_gpt2_pretrain.py --data-dir ~/data/fineweb10b-gpt2 --download
  NCCL_NET=Socket torchrun --nproc_per_node=8 examples/torch/train_gpt2_pretrain.py --data-dir ~/data/fineweb10b-gpt2
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flashoptim import FlashAdamW, FlashLion, FlashSGD, cast_model
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flashoptim import FlashAdamW, FlashLion, FlashSGD, cast_model

from example_utils import (
    ReferenceLion,
    cleanup_distributed,
    init_metrics_file,
    is_main_process,
    log_metrics,
    setup_distributed,
    warmup_cosine_lambda,
)


OPTIMIZERS = ("adamw", "sgd", "lion")
DEFAULT_LRS = {"adamw": 6e-4, "sgd": 1e-3, "lion": 2e-4}


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_layer: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj._is_residual_proj = True
        self.resid_scale = 1.0 / math.sqrt(2.0 * n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y) * self.resid_scale


class MLP(nn.Module):
    def __init__(self, n_embd: int, n_layer: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.c_proj._is_residual_proj = True
        self.resid_scale = 1.0 / math.sqrt(2.0 * n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")) * self.resid_scale


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_layer: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_layer)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        seq_len: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, n_layer) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if getattr(module, "_is_residual_proj", False):
                std = std / math.sqrt(2.0 * len(self.blocks))
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        if T > self.seq_len:
            raise ValueError(f"Sequence length {T} exceeds model context {self.seq_len}")
        pos = torch.arange(0, T, device=tokens.device, dtype=torch.long)
        x = self.wte(tokens) + self.wpe(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class TokenShardSampler:
    """Random batch sampler over FineWeb binary token shards."""

    def __init__(self, files: list[Path], seed: int, batches_per_shard: int = 256) -> None:
        self.files = files
        self.rng = np.random.default_rng(seed)
        self.batches_per_shard = batches_per_shard
        self.current_shard_idx = -1
        self.current_shard = np.empty((0,), dtype=np.uint16)
        self.remaining_batches = 0

    def sample(self, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
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
        chunks = np.asarray(shard[idx], dtype=np.int64)
        return chunks[:, :-1], chunks[:, 1:]


def ensure_data(
    data_dir: Path,
    num_train_shards: int,
    download: bool,
    repo_id: str,
) -> tuple[list[Path], Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_names = [f"fineweb_train_{i:06d}.bin" for i in range(1, num_train_shards + 1)]
    val_name = "fineweb_val_000000.bin"
    needed = train_names + [val_name]
    missing = [name for name in needed if not (data_dir / name).exists()]
    if missing and download:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=needed + ["README.md"],
            local_dir=str(data_dir),
        )
    missing = [name for name in needed if not (data_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} dataset shard(s) in {data_dir}. "
            "Use --download to fetch them."
        )
    train_files = [data_dir / name for name in train_names]
    return train_files, data_dir / val_name


def create_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay_params = []
    nodecay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    param_groups = create_param_groups(model, args.weight_decay)

    if args.optimizer == "sgd":
        if args.impl == "flash":
            return FlashSGD(
                param_groups,
                lr=args.lr,
                momentum=args.momentum,
                master_weight_bits=args.master_weight_bits,
            )
        return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

    if args.optimizer == "adamw":
        if args.impl == "flash":
            return FlashAdamW(
                param_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                master_weight_bits=args.master_weight_bits,
            )
        return torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(args.beta1, args.beta2)
        )

    if args.impl == "flash":
        return FlashLion(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            master_weight_bits=args.master_weight_bits,
        )
    return ReferenceLion(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())




def evaluate_loss(
    model: nn.Module,
    sampler: TokenShardSampler,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x_np, y_np = sampler.sample(batch_size, seq_len)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.long, non_blocking=True)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain GPT-2 (124M) with FlashOptim or reference optimizers."
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
    parser.add_argument("--batch-size", type=int, default=512, help="Global batch size.")
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
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("metrics_train_gpt2_pretrain_torch.jsonl"),
        help="Path to write train/eval metrics JSONL.",
    )
    args = parser.parse_args()

    if args.lr is None:
        args.lr = DEFAULT_LRS[args.optimizer]
    if args.impl == "reference":
        args.master_weight_bits = None
    elif args.master_weight_bits == 0:
        args.master_weight_bits = None
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank if distributed else 0)
    metrics_path = init_metrics_file(args.metrics_jsonl) if is_main_process() else None
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by world size {world_size}."
        )
    local_batch = args.batch_size // world_size

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_files, val_file = ensure_data(
        data_dir=args.data_dir,
        num_train_shards=args.num_train_shards,
        download=args.download,
        repo_id=args.repo_id,
    )
    train_sampler = TokenShardSampler(
        train_files, seed=args.seed + rank, batches_per_shard=args.batches_per_shard
    )
    val_sampler = TokenShardSampler(
        [val_file], seed=12345 + rank, batches_per_shard=args.batches_per_shard
    )

    model = GPT2Model(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    ).to(device)
    if args.impl == "flash":
        cast_model(
            model,
            dtype=torch.bfloat16,
            full_precision_keywords=["ln_1", "ln_2", "ln_f"],
        )
    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = create_optimizer(model, args)

    lr_steps = 100_000  # args.steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(lr_steps, args.warmup_steps)
    )

    if is_main_process():
        nparams = sum(p.numel() for p in model.parameters())
        print(f"impl={args.impl} optimizer={args.optimizer}")
        print(f"world_size={world_size} global_batch={args.batch_size} local_batch={local_batch}")
        print(f"steps={args.steps} warmup_steps={args.warmup_steps}")
        print(f"model: n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd}")
        if args.optimizer == "sgd":
            print(f"lr={args.lr:.2e} wd={args.weight_decay} momentum={args.momentum}")
        else:
            print(f"lr={args.lr:.2e} wd={args.weight_decay} betas=({args.beta1},{args.beta2})")
        if args.impl == "flash":
            print(f"master_weight_bits={args.master_weight_bits}")
        print(f"parameters={nparams:,}")
        print()
        log_metrics(
            metrics_path,
            {
                "event": "config",
                "impl": args.impl,
                "optimizer": args.optimizer,
                "world_size": world_size,
                "global_batch": args.batch_size,
                "local_batch": local_batch,
                "steps": args.steps,
                "warmup_steps": args.warmup_steps,
                "eval_interval": args.eval_interval,
                "eval_batches": args.eval_batches,
                "seq_len": args.seq_len,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "grad_clip": args.grad_clip,
                "master_weight_bits": args.master_weight_bits,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "vocab_size": args.vocab_size,
                "parameters": nparams,
            },
        )

    model.train()
    start = time.perf_counter()
    prev_log_time = start
    prev_log_step = 0
    running_loss = 0.0
    running_loss_steps = 0

    for step in range(1, args.steps + 1):
        x_np, y_np = train_sampler.sample(local_batch, args.seq_len)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.long, non_blocking=True)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = all_reduce_mean(float(loss.item()), device=device)
        running_loss += loss_val
        running_loss_steps += 1

        if is_main_process() and (step % args.log_interval == 0 or step == 1):
            now = time.perf_counter()
            elapsed = now - start
            toks = step * args.batch_size * args.seq_len
            toks_per_s = toks / elapsed
            interval_toks = (step - prev_log_step) * args.batch_size * args.seq_len
            interval_time = now - prev_log_time
            interval_toks_per_s = interval_toks / interval_time if interval_time > 0 else 0
            prev_log_time = now
            prev_log_step = step
            avg_loss = running_loss / running_loss_steps
            running_loss = 0.0
            running_loss_steps = 0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"step={step:05d}/{args.steps:05d} "
                f"loss={avg_loss:.4f} lr={cur_lr:.3e} "
                f"toks/s={toks_per_s:,.0f} interval={interval_toks_per_s:,.0f}"
            )
            log_metrics(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "loss": avg_loss,
                    "lr": cur_lr,
                    "tokens_per_s": toks_per_s,
                    "interval_tokens_per_s": interval_toks_per_s,
                    "elapsed_s": elapsed,
                },
            )

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = evaluate_loss(
                model=model,
                sampler=val_sampler,
                batch_size=local_batch,
                seq_len=args.seq_len,
                eval_batches=args.eval_batches,
                device=device,
            )
            val_loss = all_reduce_mean(val_loss, device=device)
            if is_main_process():
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

    cleanup_distributed(distributed)
    log_metrics(metrics_path, {"event": "training_complete", "steps": args.steps})


if __name__ == "__main__":
    main()
