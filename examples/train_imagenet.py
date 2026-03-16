"""Train ResNet-50 on ImageNet with FlashOptim (JAX).

This mirrors the ImageNet setup described in the FlashOptim paper with one
optimizer/implementation per run:
  - ResNet-50 with zero-init residuals
  - 90 epochs
  - global batch size 1024
  - 5-epoch linear warmup followed by cosine decay
  - label smoothing 0.1
  - no weight decay on biases or BatchNorm parameters
  - BF16 compute for both reference and FlashOptim runs

The script supports:
  - Reference JAX/Optax training
  - FlashOptim JAX training (`flash_sgd` / `flash_adamw`)

Dataset input:

    Preferred: load directly from Hugging Face `datasets`::

        hf auth login
        python examples/train_imagenet.py --hf-cache-dir ~/data/hf-cache

    Optional: use an extracted ImageNet folder layout::

        python examples/train_imagenet.py --data-dir /path/to/imagenet

Folder layout:

    /path/to/imagenet/
      train/<class_name>/*.JPEG
      val/<class_name>/*.JPEG

Usage:

    python examples/train_imagenet.py --data-dir /path/to/imagenet
    NCCL_NET=Socket python examples/train_imagenet.py --data-dir /path/to/imagenet
    python examples/train_imagenet.py --data-dir /path/to/imagenet --impl reference
    python examples/train_imagenet.py --data-dir /path/to/imagenet --optimizer adamw
    CUDA_VISIBLE_DEVICES=0 python examples/train_imagenet.py --data-dir /path/to/imagenet

The script uses all local JAX devices through `jax.pmap`. The global batch size
must be divisible by `jax.local_device_count()`.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import linen as nn
from flax import struct
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from example_utils import (
    StepOptimizer,
    init_metrics_file,
    log_metrics,
    make_warmup_cosine_schedule,
    optax_with_master,
    tree_nbytes,
    wrap_optax,
)
from flashoptim_jax import flash_adamw, flash_lion, flash_sgd


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000
IMAGENETTE_SIZES = {
    "imagenette2": "full",
    "imagenette2-160": "160px",
    "imagenette2-320": "320px",
}


def he_kernel_init():
    return nn.initializers.variance_scaling(
        2.0, mode="fan_out", distribution="truncated_normal"
    )


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


class BottleneckBlock(nn.Module):
    features: int
    strides: tuple[int, int] = (1, 1)
    axis_name: str = "batch"
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32
    zero_init_residual: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        residual = x

        y = nn.Conv(
            self.features,
            (1, 1),
            use_bias=False,
            kernel_init=he_kernel_init(),
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )(x)
        y = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis_name=self.axis_name,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="bn1",
        )(y)
        y = nn.relu(y)

        y = nn.Conv(
            self.features,
            (3, 3),
            strides=self.strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=he_kernel_init(),
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="conv2",
        )(y)
        y = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis_name=self.axis_name,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="bn2",
        )(y)
        y = nn.relu(y)

        y = nn.Conv(
            self.features * 4,
            (1, 1),
            use_bias=False,
            kernel_init=he_kernel_init(),
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="conv3",
        )(y)
        y = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis_name=self.axis_name,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            scale_init=(
                nn.initializers.zeros_init()
                if self.zero_init_residual
                else nn.initializers.ones_init()
            ),
            name="bn3",
        )(y)

        if residual.shape != y.shape:
            residual = nn.Conv(
                self.features * 4,
                (1, 1),
                strides=self.strides,
                use_bias=False,
                kernel_init=he_kernel_init(),
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name="proj_conv",
            )(residual)
            residual = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                axis_name=self.axis_name,
                dtype=self.compute_dtype,
                param_dtype=jnp.float32,
                name="proj_bn",
            )(residual)

        return nn.relu(y + residual)


class ResNet50(nn.Module):
    num_classes: int = NUM_CLASSES
    axis_name: str = "batch"
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32
    zero_init_residual: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = nn.Conv(
            64,
            (7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            kernel_init=he_kernel_init(),
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="stem_conv",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis_name=self.axis_name,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="stem_bn",
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        stage_blocks = (3, 4, 6, 3)
        stage_features = (64, 128, 256, 512)
        for stage_idx, (features, blocks) in enumerate(
            zip(stage_features, stage_blocks, strict=True),
            start=1,
        ):
            for block_idx in range(blocks):
                strides = (2, 2) if stage_idx > 1 and block_idx == 0 else (1, 1)
                x = BottleneckBlock(
                    features=features,
                    strides=strides,
                    axis_name=self.axis_name,
                    compute_dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                    zero_init_residual=self.zero_init_residual,
                    name=f"layer{stage_idx}_block{block_idx}",
                )(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(
            self.num_classes,
            kernel_init=he_kernel_init(),
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="fc",
        )(x)
        return x.astype(jnp.float32)


@struct.dataclass
class TrainState:
    params: Any
    batch_stats: Any
    opt_state: Any
def make_weight_decay_matcher():
    def is_no_decay(path: tuple[Any, ...], _leaf: Any) -> bool:
        path_str = [str(entry) for entry in path]
        return path_str[-1] == "bias" or any("bn" in segment for segment in path_str)

    return is_no_decay


def make_decay_mask(params: Any) -> Any:
    matcher = make_weight_decay_matcher()
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    mask_leaves = [not matcher(tuple(path), leaf) for path, leaf in path_leaves]
    return treedef.unflatten(mask_leaves)


def create_optimizer(
    args: argparse.Namespace,
    schedule,
    params: Any,
):
    no_decay_matcher = make_weight_decay_matcher()
    decay_mask = make_decay_mask(params)
    use_bf16 = any(
        jnp.asarray(l).dtype == jnp.bfloat16
        for l in jax.tree_util.tree_leaves(params)
    )

    def _ref(tx):
        return optax_with_master(tx) if use_bf16 else wrap_optax(tx)

    if args.optimizer == "sgd":
        if args.impl == "flash":
            return flash_sgd(
                learning_rate=schedule,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                quantize=True,
                master_weight_bits=args.master_weight_bits,
                param_groups=[{"params": no_decay_matcher, "weight_decay": 0.0}],
            )
        return _ref(optax.chain(
            optax.add_decayed_weights(args.weight_decay, mask=decay_mask),
            optax.sgd(learning_rate=schedule, momentum=args.momentum, nesterov=False),
        ))

    if args.optimizer == "adamw":
        if args.impl == "flash":
            return flash_adamw(
                learning_rate=schedule,
                b1=args.beta1,
                b2=args.beta2,
                weight_decay=args.weight_decay,
                quantize=True,
                fused=True,
                master_weight_bits=args.master_weight_bits,
                param_groups=[{"params": no_decay_matcher, "weight_decay": 0.0}],
            )
        return _ref(optax.adamw(
            learning_rate=schedule,
            b1=args.beta1,
            b2=args.beta2,
            weight_decay=args.weight_decay,
            mask=decay_mask,
        ))

    if args.impl == "flash":
        return flash_lion(
            learning_rate=schedule,
            b1=args.beta1,
            b2=args.beta2,
            weight_decay=args.weight_decay,
            quantize=True,
            fused=True,
            master_weight_bits=args.master_weight_bits,
            param_groups=[{"params": no_decay_matcher, "weight_decay": 0.0}],
        )
    return _ref(optax.chain(
        optax.add_decayed_weights(args.weight_decay, mask=decay_mask),
        optax.lion(learning_rate=schedule, b1=args.beta1, b2=args.beta2),
    ))


def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    label_smoothing: float,
) -> jax.Array:
    labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1], dtype=jnp.float32)
    smoothed = labels_one_hot * (1.0 - label_smoothing) + label_smoothing / logits.shape[-1]
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(smoothed * log_probs, axis=-1)


def topk_correct(logits: jax.Array, labels: jax.Array, k: int) -> jax.Array:
    k = min(k, logits.shape[-1])
    topk = jax.lax.top_k(logits, k)[1]
    return jnp.any(topk == labels[..., None], axis=-1)


def prepare_batch(
    images,
    labels,
    global_batch_size: int,
    num_devices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images_np = np.asarray(images, dtype=np.float32).transpose(0, 2, 3, 1)
    labels_np = np.asarray(labels, dtype=np.int32)
    valid = np.ones(labels_np.shape[0], dtype=np.float32)

    if labels_np.shape[0] < global_batch_size:
        pad = global_batch_size - labels_np.shape[0]
        images_np = np.pad(images_np, ((0, pad), (0, 0), (0, 0), (0, 0)))
        labels_np = np.pad(labels_np, ((0, pad),), constant_values=0)
        valid = np.pad(valid, ((0, pad),))

    per_device_batch = global_batch_size // num_devices
    return (
        images_np.reshape(num_devices, per_device_batch, 224, 224, 3),
        labels_np.reshape(num_devices, per_device_batch),
        valid.reshape(num_devices, per_device_batch),
    )


def create_train_step(model: ResNet50, tx, label_smoothing: float):
    def train_step(state: TrainState, images: jax.Array, labels: jax.Array):
        def loss_with_aux(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            logits, updates = model.apply(
                variables,
                images,
                train=True,
                mutable=["batch_stats"],
            )
            per_example_loss = cross_entropy_loss(logits, labels, label_smoothing)
            loss = per_example_loss.mean()
            correct1 = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
            return loss, (updates["batch_stats"], correct1, labels.shape[0])

        grad_fn = jax.value_and_grad(loss_with_aux, has_aux=True)
        (loss, (new_batch_stats, correct1, count)), grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")

        new_params, new_opt_state = tx.step(state.params, state.opt_state, grads)

        metrics = {
            "loss_sum": jax.lax.psum(loss * count, axis_name="batch"),
            "correct1": jax.lax.psum(correct1, axis_name="batch"),
            "count": jax.lax.psum(count, axis_name="batch"),
        }
        return (
            TrainState(
                params=new_params,
                batch_stats=new_batch_stats,
                opt_state=new_opt_state,
            ),
            metrics,
        )

    return jax.pmap(train_step, axis_name="batch")


def create_eval_step(model: ResNet50, label_smoothing: float):
    def eval_step(state: TrainState, images: jax.Array, labels: jax.Array, valid: jax.Array):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        logits = model.apply(variables, images, train=False, mutable=False)
        per_example_loss = cross_entropy_loss(logits, labels, label_smoothing)
        top1 = topk_correct(logits, labels, 1).astype(jnp.float32)
        top5 = topk_correct(logits, labels, 5).astype(jnp.float32)
        valid = valid.astype(jnp.float32)

        metrics = {
            "loss_sum": jax.lax.psum(jnp.sum(per_example_loss * valid), axis_name="batch"),
            "correct1": jax.lax.psum(jnp.sum(top1 * valid), axis_name="batch"),
            "correct5": jax.lax.psum(jnp.sum(top5 * valid), axis_name="batch"),
            "count": jax.lax.psum(jnp.sum(valid), axis_name="batch"),
        }
        return metrics

    return jax.pmap(eval_step, axis_name="batch")


def host_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}


def accumulate_metrics(
    accum: dict[str, Any] | None, metrics: dict[str, Any]
) -> dict[str, Any]:
    if accum is None:
        return metrics
    return {k: accum[k] + metrics[k] for k in metrics}


def unreplicate(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], tree)


def create_model(args: argparse.Namespace) -> ResNet50:
    return ResNet50(
        num_classes=NUM_CLASSES,
        axis_name="batch",
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        zero_init_residual=True,
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


def train(args: argparse.Namespace) -> None:
    num_devices = jax.local_device_count()
    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by local device count {num_devices}."
        )
    metrics_path = init_metrics_file(args.metrics_jsonl)

    train_dataset, val_dataset = create_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    steps_per_epoch = len(train_loader)
    model = create_model(args)
    rng = jax.random.PRNGKey(args.seed)
    init_rng, rng = jax.random.split(rng)
    dummy = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    variables = model.init(init_rng, dummy, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    total_steps = args.steps
    warmup_steps = min(total_steps, args.warmup_steps)
    schedule = make_warmup_cosine_schedule(total_steps, warmup_steps, args.lr)
    tx = create_optimizer(args, schedule, params)
    opt_state = tx.init(params)

    state = TrainState(params=params, batch_stats=batch_stats, opt_state=opt_state)
    param_bytes = tree_nbytes(params)
    state_bytes = tree_nbytes(opt_state)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    devices = jax.local_devices()
    state = jax.device_put_replicated(state, devices)
    train_step = create_train_step(model, tx, args.label_smoothing)
    eval_step = create_eval_step(model, args.label_smoothing)

    print(f"backend={jax.default_backend()} devices={jax.device_count()} local_devices={num_devices}")
    print(f"impl={args.impl} optimizer={args.optimizer}")
    print(f"global_batch_size={args.batch_size} per_device_batch={args.batch_size // num_devices}")
    print(f"steps={args.steps} warmup_steps={args.warmup_steps}")
    print(f"lr={args.lr} weight_decay={args.weight_decay}")
    if args.impl == "flash":
        print(f"master_weight_bits={args.master_weight_bits}")
    print(f"model_parameters={num_params:,}")
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
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "momentum": args.momentum,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "master_weight_bits": args.master_weight_bits,
            "log_interval": args.log_interval,
            "num_params": num_params,
            "param_bytes": param_bytes,
            "state_bytes": state_bytes,
        },
    )

    best_top1 = 0.0
    global_step = 0
    epoch = 0
    while global_step < args.steps:
        epoch += 1
        epoch_start = time.perf_counter()
        epoch_metrics = None
        interval_metrics = None

        for step_in_epoch, (images, labels) in enumerate(train_loader, start=1):
            if global_step >= args.steps:
                break
            batch_images, batch_labels, _ = prepare_batch(
                images,
                labels,
                global_batch_size=args.batch_size,
                num_devices=num_devices,
            )
            state, metrics = train_step(state, batch_images, batch_labels)
            epoch_metrics = accumulate_metrics(epoch_metrics, metrics)
            interval_metrics = accumulate_metrics(interval_metrics, metrics)
            global_step += 1

            should_log = (
                global_step % args.log_interval == 0
                or global_step == 1
                or global_step == args.steps
            )
            if should_log:
                metrics_host = host_metrics(interval_metrics)
                current_lr = float(schedule(global_step - 1))
                elapsed = time.perf_counter() - epoch_start
                if (
                    global_step % args.print_freq == 0
                    or global_step == 1
                    or global_step == args.steps
                ):
                    print(
                        f"step={global_step:05d}/{args.steps:05d} epoch={epoch:03d} "
                        f"loss={metrics_host['loss_sum'] / metrics_host['count']:.4f} "
                        f"lr={current_lr:.4e}"
                    )
                log_metrics(
                    metrics_path,
                    {
                        "event": "train",
                        "epoch": epoch,
                        "global_step": global_step,
                        "step": global_step,
                        "step_in_epoch": step_in_epoch,
                        "max_steps": args.steps,
                        "steps_per_epoch": steps_per_epoch,
                        "loss": metrics_host["loss_sum"] / metrics_host["count"],
                        "train_acc1": metrics_host["correct1"] / metrics_host["count"],
                        "lr": current_lr,
                        "elapsed_s": elapsed,
                    },
                )
                interval_metrics = None

        jax.block_until_ready(state)
        epoch_time = time.perf_counter() - epoch_start

        val_loss_sum = 0.0
        val_correct1 = 0.0
        val_correct5 = 0.0
        val_count = 0.0
        for images, labels in val_loader:
            batch_images, batch_labels, valid = prepare_batch(
                images,
                labels,
                global_batch_size=args.batch_size,
                num_devices=num_devices,
            )
            metrics = eval_step(state, batch_images, batch_labels, valid)
            metrics = host_metrics(metrics)
            val_loss_sum += metrics["loss_sum"]
            val_correct1 += metrics["correct1"]
            val_correct5 += metrics["correct5"]
            val_count += metrics["count"]

        train_metrics = host_metrics(epoch_metrics)
        train_loss = train_metrics["loss_sum"] / train_metrics["count"]
        train_top1 = train_metrics["correct1"] / train_metrics["count"]
        val_loss = val_loss_sum / val_count
        val_top1 = val_correct1 / val_count
        val_top5 = val_correct5 / val_count
        best_top1 = max(best_top1, val_top1)

        print(
            f"epoch={epoch:03d} step={global_step:05d}/{args.steps:05d} "
            f"train_loss={train_loss:.4f} train_acc1={train_top1:.4f} "
            f"val_loss={val_loss:.4f} val_acc1={val_top1:.4f} val_acc5={val_top5:.4f} "
            f"best_acc1={best_top1:.4f} time={epoch_time:.1f}s"
        )
        log_metrics(
            metrics_path,
            {
                "event": "eval",
                "epoch": epoch,
                "step": global_step,
                "max_steps": args.steps,
                "train_loss": train_loss,
                "train_acc1": train_top1,
                "val_loss": val_loss,
                "val_acc1": val_top1,
                "val_acc5": val_top5,
                "best_acc1": best_top1,
                "epoch_time_s": epoch_time,
            },
        )

    _ = unreplicate(state)
    log_metrics(metrics_path, {"event": "training_complete", "steps": args.steps})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on ImageNet with FlashOptim (JAX)."
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
        help="Use FlashOptim or Optax reference training.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw", "lion"],
        default="sgd",
        help="Optimizer family.",
    )
    parser.add_argument("--steps", type=int, default=140_760)
    parser.add_argument("--warmup-steps", type=int, default=20_700)
    parser.add_argument("--batch-size", type=int, default=1024)
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
    parser.add_argument("--master-weight-bits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("metrics_train_imagenet_jax.jsonl"),
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
    train(parse_args())


if __name__ == "__main__":
    main()
