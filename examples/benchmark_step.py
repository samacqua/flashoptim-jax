"""Benchmark optimizer-step memory and speed with shared 5 conditions."""

import argparse
import os
import statistics
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import torch

from example_utils import (
    OPTIMIZERS,
    WEIGHT_DTYPES,
    estimate_state_nbytes,
    fmt_bytes,
    jax_dtype_from_name,
    parse_bool,
    parse_shape,
    percentile,
    torch_dtype_from_name,
    get_jax_baseline_opt,
    get_jax_flash_opt,
    get_torch_baseline_opt,
    get_torch_flash_opt,
)


def print_speed_row(name: str, median_ms: float, p95_ms: float, n_params: int) -> None:
    params_per_sec = n_params / (median_ms / 1000.0)
    print(
        f"  {name:24s}  median={median_ms:7.3f}ms  "
        f"p95={p95_ms:7.3f}ms  params/s={params_per_sec:13,.0f}"
    )


def _jax_tree_nbytes(tree) -> int:
    return sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(tree))

def _torch_nbytes(params: list[torch.nn.Parameter], opt: torch.optim.Optimizer) -> int:
    param_bytes = sum(p.numel() * p.element_size() for p in params)
    state_bytes = 0
    for state in opt.state.values():
        for value in state.values():
            state_bytes += estimate_state_nbytes(value)
    return param_bytes + state_bytes

def _jax_make_params(shape: tuple[int, ...], dtype) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    return {
        "w": jax.random.normal(k1, shape, dtype=jnp.float32).astype(dtype),
        "b": jax.random.normal(k2, (shape[-1],), dtype=jnp.float32).astype(dtype),
    }

def _torch_make_params(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> list[torch.nn.Parameter]:
    g = torch.Generator(device=device)
    g.manual_seed(42)
    return [
        torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype, generator=g)),
        torch.nn.Parameter(torch.randn((shape[-1],), device=device, dtype=dtype, generator=g)),
    ]


def _jax_make_grads(shape: tuple[int, ...]) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(jax.random.PRNGKey(123))
    return {
        "w": jax.random.normal(k1, shape, dtype=jnp.float32) * 0.01,
        "b": jax.random.normal(k2, (shape[-1],), dtype=jnp.float32) * 0.01,
    }

def _torch_make_grads(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads = []
    g = torch.Generator(device=params[0].device)
    g.manual_seed(123)
    for p in params:
        grads.append(torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=g) * 0.01)
    return grads


def _torch_init_state(opt: torch.optim.Optimizer, params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = g
    opt.step()
    opt.zero_grad(set_to_none=True)


def _jax_block(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()

def run_memory(
    optimizer: str,
    weights: str,
    quantize: bool,
    fused: bool,
    shapes: list[tuple[int, ...]],
    device: torch.device,
) -> None:
    print("=" * 80)
    print(f"MEMORY COMPARISON  ({optimizer}, weights={weights})  params + optimizer state")
    print("=" * 80)
    print(
        f"{'shape':>14s}  {'PT base':>10s}  {'PT flash':>10s}  "
        f"{'JAX base':>10s}  {'JAX flash':>10s}"
    )
    print("-" * 80)

    jax_dtype = jax_dtype_from_name(weights)
    torch_dtype = torch_dtype_from_name(weights)

    for shape in shapes:
        p = _jax_make_params(shape, jax_dtype)
        jax_init, _ = get_jax_baseline_opt(optimizer, 1e-3, weights)
        s = jax_init(p)
        jax_base_bytes = _jax_tree_nbytes(p) + _jax_tree_nbytes(s)

        p_flash = _jax_make_params(shape, jax_dtype)
        tx_flash = get_jax_flash_opt(optimizer, 1e-3, weights, quantize, fused=fused)
        s_flash = tx_flash.init(p_flash)
        jax_flash_bytes = _jax_tree_nbytes(p_flash) + _jax_tree_nbytes(s_flash)

        pt = _torch_make_params(shape, torch_dtype, device)
        gt = _torch_make_grads(pt)
        opt = get_torch_baseline_opt(optimizer, pt, 1e-3, weights)
        _torch_init_state(opt, pt, gt)
        torch_base_bytes = _torch_nbytes(pt, opt)

        pt = _torch_make_params(shape, torch_dtype, device)
        gt = _torch_make_grads(pt)
        opt = get_torch_flash_opt(optimizer, pt, 1e-3, weights, quantize, fused=fused)
        _torch_init_state(opt, pt, gt)
        torch_flash_bytes = _torch_nbytes(pt, opt)

        shape_str = "x".join(str(d) for d in shape)
        print(
            f"{shape_str:>14s}  {fmt_bytes(torch_base_bytes):>10s}  {fmt_bytes(torch_flash_bytes):>10s}  "
            f"{fmt_bytes(jax_base_bytes):>10s}  {fmt_bytes(jax_flash_bytes):>10s}"
        )
    print()


def _bench_jax(name: str, step_fn, params, state, grads, warmup: int, steps: int, n_params: int, steps_per_sync: int) -> None:
    for _ in range(warmup):
        for _ in range(steps_per_sync):
            params, state = step_fn(params, state, grads)
        _jax_block((params, state))

    samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(steps_per_sync):
            params, state = step_fn(params, state, grads)
        _jax_block((params, state))
        samples.append((time.perf_counter() - t0) * 1000 / steps_per_sync)
    print_speed_row(name, statistics.median(samples), percentile(samples, 0.95), n_params)


def _bench_torch(name: str, step_fn, warmup: int, steps: int, n_params: int, steps_per_sync: int) -> None:
    for _ in range(warmup):
        for _ in range(steps_per_sync):
            step_fn()
        torch.cuda.synchronize()

    samples = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps_per_sync):
            step_fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000 / steps_per_sync)
    print_speed_row(name, statistics.median(samples), percentile(samples, 0.95), n_params)


def run_speed(
    optimizer: str,
    weights: str,
    quantize: bool,
    fused: bool,
    shapes: list[tuple[int, ...]],
    warmup: int,
    steps: int,
    device: torch.device,
    steps_per_sync: int = 80,
) -> None:
    print("=" * 80)
    print(f"SPEED COMPARISON  ({optimizer}, weights={weights}) pipelined throughput ({steps_per_sync} steps/sync)")
    print("=" * 80)

    jax_dtype = jax_dtype_from_name(weights)
    torch_dtype = torch_dtype_from_name(weights)

    for shape in shapes:
        grads_jax = _jax_make_grads(shape)
        n_params = 1
        for dim in shape:
            n_params *= dim
        n_params += shape[-1]
        print(f"\nshape={'x'.join(str(d) for d in shape)}  params={n_params:,}")

        # Benchmark JAX baseline.
        base_init, base_step = get_jax_baseline_opt(optimizer, 1e-3, weights)
        @jax.jit
        def jax_base(pv, sv, gv, _step=base_step):
            return _step(pv, sv, gv)
        p = _jax_make_params(shape, jax_dtype)
        s = base_init(p)
        _bench_jax("jax baseline", jax_base, p, s, grads_jax, warmup, steps, n_params, steps_per_sync)

        # Benchmark JAX flash.
        tx = get_jax_flash_opt(optimizer, 1e-3, weights, quantize, fused=fused)
        p = _jax_make_params(shape, jax_dtype)
        s = tx.init(p)
        jax_flash = jax.jit(
            lambda pv, sv, gv, _tx=tx: _tx.step(pv, sv, gv),
            donate_argnums=(0, 1) if quantize else (),
        )
        _bench_jax("jax flash", jax_flash, p, s, grads_jax, warmup, steps, n_params, steps_per_sync)

        # Benchmark Torch baseline.
        def make_torch_step(params, grads, opt):
            def step():
                for p, g in zip(params, grads, strict=True):
                    p.grad = g
                opt.step()
                opt.zero_grad(set_to_none=True)

            return step

        params = _torch_make_params(shape, torch_dtype, device)
        grads = _torch_make_grads(params)
        opt = get_torch_baseline_opt(optimizer, params, 1e-3, weights)
        _bench_torch("torch baseline", torch.compile(make_torch_step(params, grads, opt)), warmup, steps, n_params, steps_per_sync)

        # Benchmark Torch flash.
        params = _torch_make_params(shape, torch_dtype, device)
        grads = _torch_make_grads(params)
        opt = get_torch_flash_opt(optimizer, params, 1e-3, weights, quantize, fused=fused)
        _bench_torch("torch flash", make_torch_step(params, grads, opt), warmup, steps, n_params, steps_per_sync)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark optimizer-step memory and speed.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=list(OPTIMIZERS))
    parser.add_argument("--weights", type=str, default="bf16", choices=list(WEIGHT_DTYPES))
    parser.add_argument("--quantize", type=parse_bool, default=True)
    parser.add_argument("--fused", type=parse_bool, default=True)
    parser.add_argument("--shapes", type=str, default="4096x4096,8192x4096,2048x2048,1024x1024")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--steps-per-sync", type=int, default=20, help="Steps between GPU syncs.")
    parser.add_argument("--skip-memory", type=parse_bool, default=False)
    parser.add_argument("--skip-speed", type=parse_bool, default=False)
    args = parser.parse_args()

    shapes = [parse_shape(part) for part in args.shapes.split(",")]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(0)

    print(f"JAX backend={jax.default_backend()}  devices={jax.device_count()}")
    print(f"Torch device={torch.cuda.get_device_name(device)}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Weights:   {args.weights}")
    print(f"Quantize:  {args.quantize}")
    print(f"Fused:     {args.fused}")
    print()

    if not args.skip_memory:
        run_memory(args.optimizer, args.weights, args.quantize, args.fused, shapes, device)
    if not args.skip_speed:
        run_speed(
            args.optimizer,
            args.weights,
            args.quantize,
            args.fused,
            shapes,
            args.warmup,
            args.steps,
            device,
            steps_per_sync=args.steps_per_sync,
        )


if __name__ == "__main__":
    main()
