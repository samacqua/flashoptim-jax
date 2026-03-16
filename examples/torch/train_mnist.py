"""Train a CNN on MNIST with FlashOptim or reference optimizers."""

from __future__ import annotations

import argparse
import gzip
import struct
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flashoptim import FlashAdamW, FlashLion, FlashSGD, cast_model, enable_gradient_release
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flashoptim import FlashAdamW, FlashLion, FlashSGD, cast_model, enable_gradient_release

from example_utils import (
    MasterWeightOptimizer,
    ReferenceLion,
    init_metrics_file,
    log_metrics,
)


_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
OPTIMIZERS = ("sgd", "adamw", "lion")
DEFAULT_LRS = {"sgd": 1e-2, "adamw": 1e-3, "lion": 1e-4}


def _download(filename: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    if not path.exists():
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(f"{_MIRROR}/{filename}", path)
    return path


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, 1, rows, cols)


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        f.read(8)
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def load_mnist(
    data_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download MNIST if needed and return normalized tensors on device."""
    missing = [filename for filename in _FILES.values() if not (data_dir / filename).exists()]
    if missing:
        print(f"Local MNIST data missing in {data_dir}; downloading...")
    paths = {k: _download(v, data_dir) for k, v in _FILES.items()}
    train_x = _read_images(paths["train_images"]).to(device, dtype) / 255.0
    train_y = _read_labels(paths["train_labels"]).to(device)
    test_x = _read_images(paths["test_images"]).to(device, dtype) / 255.0
    test_y = _read_labels(paths["test_labels"]).to(device)
    return train_x, train_y, test_x, test_y


class MNISTNet(nn.Module):
    """Small CNN: two conv blocks followed by two linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    params_by_dtype: dict[torch.dtype, list[torch.nn.Parameter]] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params_by_dtype.setdefault(param.dtype, []).append(param)
    return [
        {"params": params, "weight_decay": weight_decay}
        for params in params_by_dtype.values()
    ]


def create_optimizer(model: nn.Module, args: argparse.Namespace):
    param_groups = create_param_groups(model, args.weight_decay)
    if args.optimizer == "sgd":
        if args.impl == "flash":
            return FlashSGD(
                param_groups,
                lr=args.lr,
                momentum=args.momentum,
                master_weight_bits=args.master_weight_bits,
            )
        return MasterWeightOptimizer(
            torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)
        )
    if args.optimizer == "adamw":
        if args.impl == "flash":
            return FlashAdamW(
                param_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                master_weight_bits=args.master_weight_bits,
            )
        return MasterWeightOptimizer(
            torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
        )
    if args.impl == "flash":
        return FlashLion(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            master_weight_bits=args.master_weight_bits,
        )
    return MasterWeightOptimizer(
        ReferenceLion(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    )


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    epoch: int,
    total_start: float,
    log_interval: int,
    metrics_path: Path | None,
) -> float:
    model.train()
    n = images.size(0)
    perm = torch.randperm(n, device=images.device)
    images, labels = images[perm], labels[perm]

    total_loss_sum = torch.zeros((), device=images.device, dtype=torch.float32)
    total_count = 0
    interval_loss_sum = torch.zeros((), device=images.device, dtype=torch.float32)
    interval_count = 0
    steps_per_epoch = (n + batch_size - 1) // batch_size
    current_lr = optimizer.param_groups[0]["lr"]
    for step_in_epoch, i in enumerate(range(0, n, batch_size), start=1):
        logits = model(images[i : i + batch_size])
        loss = F.cross_entropy(logits, labels[i : i + batch_size])
        loss.backward()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.step()
            optimizer.zero_grad()
        batch_count = labels[i : i + batch_size].size(0)
        loss_f32 = loss.detach().to(torch.float32)
        total_loss_sum += loss_f32 * batch_count
        total_count += batch_count
        interval_loss_sum += loss_f32 * batch_count
        interval_count += batch_count

        global_step = (epoch - 1) * steps_per_epoch + step_in_epoch
        if metrics_path is not None and (
            global_step % log_interval == 0
            or step_in_epoch == steps_per_epoch
            or global_step == 1
        ):
            torch.cuda.synchronize()
            avg_loss = (interval_loss_sum / interval_count).item()
            elapsed = time.perf_counter() - total_start
            print(
                f"  step={global_step:05d} epoch={epoch:02d} "
                f"step_in_epoch={step_in_epoch:03d}/{steps_per_epoch:03d} "
                f"loss={avg_loss:.4f} lr={current_lr:.3e}"
            )
            log_metrics(
                metrics_path,
                {
                    "event": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "step_in_epoch": step_in_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "loss": avg_loss,
                    "lr": current_lr,
                    "elapsed_s": elapsed,
                },
            )
            interval_loss_sum.zero_()
            interval_count = 0

    return (total_loss_sum / total_count).item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    n = images.size(0)
    for i in range(0, n, batch_size):
        preds = model(images[i : i + batch_size]).argmax(1)
        correct += (preds == labels[i : i + batch_size]).sum().item()
    return correct / n


class TrainResult(NamedTuple):
    epoch_results: list[dict[str, float]]
    total_time: float
    param_bytes: int
    warmup_peak_bytes: int
    steady_peak_bytes: int
    final_acc: float


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    steps: int,
    batch_size: int,
    metrics_path: Path | None,
    log_interval: int,
    print_lr: bool = False,
) -> TrainResult:
    sched_opt = getattr(optimizer, "base_opt", optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sched_opt, T_max=max(1, steps - 1)
    )
    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.nbytes for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Params: {param_bytes / 1024:.1f} KB")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    epoch_results = []
    total_time = 0.0
    total_start = time.perf_counter()
    best_acc = 0.0
    global_step = 0
    epoch = 0
    while global_step < steps:
        epoch += 1
        t0 = time.perf_counter()
        model.train()
        n = train_x.size(0)
        perm = torch.randperm(n, device=train_x.device)
        images = train_x[perm]
        labels = train_y[perm]
        total_loss_sum = torch.zeros((), device=train_x.device, dtype=torch.float32)
        total_count = 0
        interval_loss_sum = torch.zeros((), device=train_x.device, dtype=torch.float32)
        interval_count = 0
        for i in range(0, n, batch_size):
            if global_step >= steps:
                break
            current_lr = optimizer.param_groups[0]["lr"]
            logits = model(images[i : i + batch_size])
            loss = F.cross_entropy(logits, labels[i : i + batch_size])
            loss.backward()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            batch_count = labels[i : i + batch_size].size(0)
            loss_f32 = loss.detach().to(torch.float32)
            total_loss_sum += loss_f32 * batch_count
            total_count += batch_count
            interval_loss_sum += loss_f32 * batch_count
            interval_count += batch_count
            global_step += 1

            if metrics_path is not None and (
                global_step % log_interval == 0
                or global_step == 1
                or global_step == steps
            ):
                torch.cuda.synchronize()
                avg_interval_loss = (interval_loss_sum / interval_count).item()
                elapsed = time.perf_counter() - total_start
                print(
                    f"  step={global_step:05d}/{steps:05d} epoch={epoch:02d} "
                    f"loss={avg_interval_loss:.4f} lr={current_lr:.3e}"
                )
                log_metrics(
                    metrics_path,
                    {
                        "event": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "max_steps": steps,
                        "loss": avg_interval_loss,
                        "lr": current_lr,
                        "elapsed_s": elapsed,
                    },
                )
                interval_loss_sum.zero_()
                interval_count = 0

        avg_loss = (total_loss_sum / total_count).item()
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        if epoch > 1:
            total_time += epoch_time
        if epoch == 1:
            warmup_peak_bytes = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        acc = evaluate(model, test_x, test_y, batch_size)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - total_start
        best_acc = max(best_acc, acc)
        if print_lr:
            print(
                f"  epoch={epoch:02d} step={global_step:05d}/{steps:05d} loss={avg_loss:.4f}  "
                f"acc={acc:.4f}  lr={lr:.2e}  time={epoch_time:.2f}s"
            )
        else:
            print(
                f"  epoch={epoch:02d} step={global_step:05d}/{steps:05d} loss={avg_loss:.4f}  "
                f"acc={acc:.4f}  time={epoch_time:.2f}s"
            )
        epoch_results.append(
            {
                "epoch": epoch,
                "acc": float(acc),
                "loss": avg_loss,
                "lr": lr,
                "time": epoch_time,
            }
        )
        if metrics_path is not None:
            log_metrics(
                metrics_path,
                {
                    "event": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "max_steps": steps,
                    "train_loss": float(avg_loss),
                    "lr": float(lr),
                    "epoch_time_s": float(epoch_time),
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
                    "eval_acc": float(acc),
                    "best_eval_acc": float(best_acc),
                    "elapsed_s": elapsed,
                },
            )

    steady_peak_bytes = torch.cuda.memory_allocated()
    if epoch <= 1:
        steady_peak_bytes = warmup_peak_bytes
    final_acc = evaluate(model, test_x, test_y, batch_size)
    return TrainResult(
        epoch_results=epoch_results,
        total_time=total_time,
        param_bytes=param_bytes,
        warmup_peak_bytes=int(warmup_peak_bytes),
        steady_peak_bytes=int(steady_peak_bytes),
        final_acc=float(final_acc),
    )


def verify_checkpoint_roundtrip(
    model: nn.Module,
    optimizer: FlashAdamW,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
) -> None:
    """Save a full-precision checkpoint, reload it, and verify correctness."""
    acc_before = evaluate(model, test_x, test_y, batch_size)
    fp32_sd = optimizer.get_fp32_model_state_dict(model)

    dtypes = {v.dtype for v in fp32_sd.values() if isinstance(v, torch.Tensor)}
    non_fp32 = dtypes - {torch.float32}
    assert not non_fp32, f"Expected all fp32, got {non_fp32}"
    print(f"  Exported {len(fp32_sd)} tensors, all fp32.")

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(fp32_sd, f.name)
        loaded_sd = torch.load(f.name, weights_only=True)

    optimizer.set_fp32_model_state_dict(model, loaded_sd)

    acc_after = evaluate(model, test_x, test_y, batch_size)
    print(f"  Accuracy before save: {acc_before:.2%}")
    print(f"  Accuracy after load:  {acc_after:.2%}")
    assert acc_before == acc_after, (
        f"Accuracy changed after checkpoint reload: {acc_before} -> {acc_after}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST CNN with FlashOptim or reference optimizers."
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--gradient-release", action="store_true")
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("metrics_train_mnist_torch.jsonl"),
        help="Path to write train/eval metrics JSONL.",
    )
    args = parser.parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LRS[args.optimizer]
    if args.weight_decay is None:
        args.weight_decay = 0.01 if args.optimizer == "adamw" else 0.0
    if args.impl == "reference":
        args.master_weight_bits = None
    elif args.master_weight_bits == 0:
        args.master_weight_bits = None
    if args.gradient_release and args.impl != "flash":
        raise SystemExit("--gradient-release requires --impl flash.")
    if args.verify_checkpoint and not (args.impl == "flash" and args.optimizer == "adamw"):
        raise SystemExit("--verify-checkpoint requires --impl flash --optimizer adamw.")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    dtype = torch.bfloat16
    metrics_path = init_metrics_file(args.metrics_jsonl)

    print("Loading MNIST...")
    train_x, train_y, test_x, test_y = load_mnist(args.data_dir, device, dtype)

    model = MNISTNet().to(device)
    cast_model(model, dtype=dtype)
    optimizer = create_optimizer(model, args)

    print("backend=torch devices=1")
    print(f"impl={args.impl} optimizer={args.optimizer}")
    if args.optimizer == "sgd":
        print(f"lr={args.lr:.2e} wd={args.weight_decay} momentum={args.momentum}")
    else:
        print(f"lr={args.lr:.2e} wd={args.weight_decay} betas=({args.beta1},{args.beta2})")
    if args.impl == "flash":
        print(f"master_weight_bits={args.master_weight_bits}")
    if args.gradient_release:
        print("gradient_release=true")
    print()

    log_metrics(
        metrics_path,
        {
            "event": "config",
            "backend": "torch",
            "devices": 1,
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
            "gradient_release": args.gradient_release,
            "log_interval": args.log_interval,
            "num_params": sum(p.numel() for p in model.parameters()),
        },
    )

    gr_handle = None
    if args.gradient_release:
        gr_handle = enable_gradient_release(model, optimizer)

    result = run_training(
        model,
        optimizer,
        train_x,
        train_y,
        test_x,
        test_y,
        args.steps,
        args.batch_size,
        metrics_path,
        args.log_interval,
        print_lr=True,
    )

    print(
        f"\nFinal accuracy: {result.final_acc:.2%}  "
        f"Total time (epochs 2+): {result.total_time:.2f}s"
    )

    if args.verify_checkpoint:
        print("\nVerifying checkpoint round-trip...")
        if gr_handle is not None and gr_handle.active:
            gr_handle.remove()
        verify_checkpoint_roundtrip(model, optimizer, test_x, test_y, args.batch_size)

    if gr_handle is not None and gr_handle.active:
        gr_handle.remove()

    log_metrics(
        metrics_path,
        {
            "event": "training_complete",
            "impl": args.impl,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "final_acc": result.final_acc,
            "total_time_s": result.total_time,
            "param_bytes": result.param_bytes,
            "warmup_peak_bytes": result.warmup_peak_bytes,
            "steady_peak_bytes": result.steady_peak_bytes,
        },
    )


if __name__ == "__main__":
    main()
