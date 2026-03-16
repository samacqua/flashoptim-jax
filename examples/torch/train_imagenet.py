"""Train ResNet-50 on ImageNet with the paper's recipe.

This example mirrors the ImageNet setup described in the FlashOptim paper:
  - `timm` ResNet-50 with zero-init residuals
  - 90 epochs
  - global batch size 1024
  - 5-epoch linear warmup followed by cosine decay
  - label smoothing 0.1
  - no weight decay on biases or BatchNorm parameters
  - bf16 activations for both reference and FlashOptim runs

The script supports both the reference optimizers and FlashOptim:
  - SGD: `torch.optim.SGD` vs `FlashSGD`
  - AdamW: `torch.optim.AdamW` vs `FlashAdamW`

Dataset input:

    Preferred: load directly from Hugging Face `datasets`::

        hf auth login
        python examples/train_imagenet.py --hf-cache-dir ~/data/hf-cache

    Optional: use an extracted ImageNet folder layout::

        python examples/train_imagenet.py --data-dir /path/to/imagenet

Expected folder layout:

    /path/to/imagenet/
      train/<class_name>/*.JPEG
      val/<class_name>/*.JPEG

Usage:

    python examples/train_imagenet.py --hf-cache-dir ~/data/hf-cache
    python examples/train_imagenet.py --data-dir /path/to/imagenet --impl reference
    NCCL_NET=Socket torchrun --nproc_per_node=8 examples/train_imagenet.py --hf-cache-dir ~/data/hf-cache

For the paper's exact global batch size, keep `--batch-size 1024`.
With 8 GPUs this corresponds to 128 samples per GPU.
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from flashoptim import FlashAdamW, FlashLion, FlashSGD, cast_model

from example_utils import (
    MasterWeightOptimizer,
    ReferenceLion,
    cleanup_distributed,
    init_metrics_file,
    is_main_process,
    log_metrics,
    seed_everything,
    setup_distributed,
    warmup_cosine_lambda,
)

from datasets import load_dataset
import timm
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENETTE_SIZES = {
    "imagenette2": "full",
    "imagenette2-160": "160px",
    "imagenette2-320": "320px",
}


class RandomShortSideResize:
    """Resize so the shorter image side is sampled uniformly from a range."""

    def __init__(
        self,
        min_size: int = 256,
        max_size: int = 480,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        size = random.randint(self.min_size, self.max_size)
        return TF.resize(img, size=size, interpolation=self.interpolation)


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            RandomShortSideResize(256, 480, InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HuggingFaceImageNet(Dataset):
    def __init__(self, split, transform: transforms.Compose) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        example = self.split[index]
        image = example["image"].convert("RGB")
        label = int(example["label"])
        return self.transform(image), label


def partition_params(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Iterable[torch.nn.Parameter] | float]]:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_model(device: torch.device, impl: str) -> nn.Module:
    model = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=1000,
        zero_init_last=True,
    ).to(device)
    cast_model(model, dtype=torch.bfloat16)
    return model


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    impl: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    betas: tuple[float, float],
    master_weight_bits: int | None,
) -> torch.optim.Optimizer:
    param_groups = partition_params(model, weight_decay)

    if optimizer_name == "sgd":
        if impl == "flash":
            return FlashSGD(
                param_groups, lr=lr, momentum=momentum,
                master_weight_bits=master_weight_bits,
            )
        return MasterWeightOptimizer(
            torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        )

    if optimizer_name == "lion":
        if impl == "flash":
            return FlashLion(
                param_groups, lr=lr, betas=betas,
                master_weight_bits=master_weight_bits,
            )
        return MasterWeightOptimizer(
            ReferenceLion(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
        )

    if impl == "flash":
        return FlashAdamW(
            param_groups, lr=lr, betas=betas,
            master_weight_bits=master_weight_bits,
        )
    return MasterWeightOptimizer(
        torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(total_steps, warmup_steps),
    )


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...]) -> list[float]:
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.unsqueeze(0))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    max_steps: int,
    print_freq: int,
    log_interval: int,
    metrics_path: Path | None,
) -> tuple[float, float, int]:
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_correct = torch.zeros((), device=device, dtype=torch.float32)
    total_seen = torch.zeros((), device=device, dtype=torch.float32)
    interval_loss = torch.zeros((), device=device, dtype=torch.float32)
    interval_correct = torch.zeros((), device=device, dtype=torch.float32)
    interval_seen = torch.zeros((), device=device, dtype=torch.float32)
    start_time = time.time()
    steps_per_epoch = len(loader)

    for step_in_epoch, (images, target) in enumerate(loader, start=1):
        if global_step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1

        batch_size = target.size(0)
        batch_size_t = torch.tensor(float(batch_size), device=device)
        loss_f32 = loss.detach().to(torch.float32)
        correct_f32 = (output.argmax(dim=1) == target).sum().to(torch.float32)
        total_loss += loss_f32 * batch_size_t
        total_correct += correct_f32
        total_seen += batch_size_t
        interval_loss += loss_f32 * batch_size_t
        interval_correct += correct_f32
        interval_seen += batch_size_t

        should_log = (
            global_step % log_interval == 0
            or global_step == 1
            or global_step == max_steps
        )
        if should_log:
            log_stats = reduce_tensor(torch.stack([interval_loss, interval_correct, interval_seen]))
            if is_main_process():
                torch.cuda.synchronize()
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                avg_loss = (log_stats[0] / log_stats[2]).item()
                avg_acc1 = (log_stats[1] / log_stats[2]).item()
                if (
                    global_step % print_freq == 0
                    or global_step == 1
                    or global_step == max_steps
                ):
                    print(
                        f"step={global_step:05d}/{max_steps:05d} epoch={epoch:03d} "
                        f"loss={avg_loss:.4f} lr={lr:.4e} time={elapsed:.1f}s"
                    )
                log_metrics(
                    metrics_path,
                    {
                        "event": "train",
                        "epoch": epoch,
                        "global_step": global_step,
                        "step": global_step,
                        "step_in_epoch": step_in_epoch,
                        "max_steps": max_steps,
                        "steps_per_epoch": steps_per_epoch,
                        "loss": avg_loss,
                        "train_acc1": avg_acc1,
                        "lr": lr,
                        "elapsed_s": elapsed,
                    },
                )
            interval_loss.zero_()
            interval_correct.zero_()
            interval_seen.zero_()

    stats = reduce_tensor(torch.stack([total_loss, total_correct, total_seen]))
    return (stats[0] / stats[2]).item(), (stats[1] / stats[2]).item(), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_seen = 0.0

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = loss_fn(output, target)

        batch_size = target.size(0)
        top1, top5 = accuracy(output, target, topk=(1, 5))

        total_loss += loss.item() * batch_size
        total_top1 += top1
        total_top5 += top5
        total_seen += batch_size

    stats = torch.tensor(
        [total_loss, total_top1, total_top5, total_seen],
        device=device,
    )
    stats = reduce_tensor(stats)
    return (
        (stats[0] / stats[3]).item(),
        (stats[1] / stats[3]).item(),
        (stats[2] / stats[3]).item(),
    )


def create_datasets(args: argparse.Namespace):
    if args.data_dir is not None:
        train_dir = args.data_dir / "train"
        val_dir = args.data_dir / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            imagenette_size = IMAGENETTE_SIZES.get(args.data_dir.name)
            if imagenette_size is None:
                raise SystemExit(
                    f"Expected ImageNet layout under {args.data_dir} with `train/` and `val/`."
                )
            print(
                f"Local image data missing in {args.data_dir}; downloading Imagenette ({imagenette_size})..."
            )
            datasets.Imagenette(
                root=args.data_dir.parent,
                split="train",
                size=imagenette_size,
                download=True,
            )
            datasets.Imagenette(
                root=args.data_dir.parent,
                split="val",
                size=imagenette_size,
                download=True,
            )
        return (
            datasets.ImageFolder(train_dir, transform=build_train_transform()),
            datasets.ImageFolder(val_dir, transform=build_eval_transform()),
        )

    dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=args.hf_cache_dir)
    return (
        HuggingFaceImageNet(dataset["train"], transform=build_train_transform()),
        HuggingFaceImageNet(dataset["validation"], transform=build_eval_transform()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on ImageNet with the FlashOptim paper recipe."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional extracted ImageNet root with `train/` and `val/`.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Cache directory for Hugging Face datasets when using `load_dataset`.",
    )
    parser.add_argument(
        "--impl",
        choices=["flash", "reference"],
        default="flash",
        help="Use FlashOptim or the reference PyTorch optimizer.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw", "lion"],
        default="sgd",
        help="Optimizer family.",
    )
    parser.add_argument("--steps", type=int, default=140_760)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20_700,
        help="Linear warmup steps before cosine decay.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Global batch size across all ranks.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Defaults to 1.024 for SGD and 3e-3 for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Defaults to 3e-5 for SGD and 3e-4 for AdamW.",
    )
    parser.add_argument(
        "--master-weight-bits",
        type=int,
        default=24,
        help="FlashOptim master-weight precision: 24, 32, or 0 for None.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("metrics_train_imagenet_torch.jsonl"),
        help="Path to write train/eval metrics JSONL.",
    )
    args = parser.parse_args()
    if args.lr is None:
        if args.optimizer == "sgd":
            args.lr = 1.024
        elif args.optimizer == "adamw":
            args.lr = 3e-3
        else:
            args.lr = 1e-4
    if args.weight_decay is None:
        if args.optimizer == "sgd":
            args.weight_decay = 3e-5
        elif args.optimizer == "adamw":
            args.weight_decay = 3e-4
        else:
            args.weight_decay = 0.0
    if args.impl == "reference":
        args.master_weight_bits = None
    elif args.master_weight_bits == 0:
        args.master_weight_bits = None
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This example requires CUDA.")

    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank if distributed else 0)
    metrics_path = init_metrics_file(args.metrics_jsonl) if is_main_process() else None

    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by world size {world_size}."
        )

    per_device_batch_size = args.batch_size // world_size
    seed_everything(args.seed + rank)

    # Disable TF32 so matmul precision matches JAX's default (full fp32 mantissa).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train_dataset, val_dataset = create_datasets(args)

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = create_model(device, args.impl)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = create_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        impl=args.impl,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        betas=(args.beta1, args.beta2),
        master_weight_bits=args.master_weight_bits,
    )
    sched_opt = optimizer.base_opt if isinstance(optimizer, MasterWeightOptimizer) else optimizer
    scheduler = build_scheduler(
        optimizer=sched_opt,
        total_steps=args.steps,
        warmup_steps=min(args.steps, args.warmup_steps),
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    if is_main_process():
        print(f"impl={args.impl} optimizer={args.optimizer}")
        print(f"world_size={world_size} global_batch_size={args.batch_size}")
        print(f"per_device_batch_size={per_device_batch_size}")
        print(f"steps={args.steps} warmup_steps={args.warmup_steps}")
        print(f"lr={args.lr} weight_decay={args.weight_decay}")
        if args.impl == "flash":
            print(f"master_weight_bits={args.master_weight_bits}")
        print()
    log_metrics(
        metrics_path,
        {
            "event": "config",
            "impl": args.impl,
            "optimizer": args.optimizer,
            "world_size": world_size,
            "global_batch_size": args.batch_size,
            "per_device_batch_size": per_device_batch_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "label_smoothing": args.label_smoothing,
            "master_weight_bits": args.master_weight_bits,
            "log_interval": args.log_interval,
        },
    )

    best_top1 = 0.0
    global_step = 0
    epoch = 0
    while global_step < args.steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            global_step=global_step,
            max_steps=args.steps,
            print_freq=args.print_freq,
            log_interval=args.log_interval,
            metrics_path=metrics_path,
        )
        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        epoch_time = time.time() - epoch_start
        best_top1 = max(best_top1, val_top1)

        if is_main_process():
            print(
                f"epoch={epoch:03d} step={global_step:05d}/{args.steps:05d} "
                f"train_loss={train_loss:.4f} train_acc1={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc1={val_top1:.4f} val_acc5={val_top5:.4f} "
                f"best_acc1={best_top1:.4f}"
            )
        log_metrics(
            metrics_path,
            {
                "event": "eval",
                "epoch": epoch,
                "step": global_step,
                "max_steps": args.steps,
                "train_loss": train_loss,
                "train_acc1": train_acc,
                "val_loss": val_loss,
                "val_acc1": val_top1,
                "val_acc5": val_top5,
                "best_acc1": best_top1,
                "epoch_time_s": epoch_time,
            },
        )

    cleanup_distributed(distributed)
    log_metrics(metrics_path, {"event": "training_complete", "steps": args.steps})


if __name__ == "__main__":
    main()
