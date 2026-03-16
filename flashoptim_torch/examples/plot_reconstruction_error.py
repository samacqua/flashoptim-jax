#!/usr/bin/env python3
"""Reproduce the FP32 reconstruction-error figure from flashoptim.tex.

This script sweeps FP32 values by exponent bucket and computes mean relative
reconstruction error for four schemes:
  1) No correction (narrow dtype only)
  2) Narrow + narrow residual (e.g. BF16 + BF16)
  3) Narrow + ULP-normalized int8 correction
  4) Narrow + ULP-normalized int16 correction

By default it reproduces the paper ranges:
  - BF16 panel: exponents [-127, 127]
  - FP16 panel: exponents [-30, 15]
with all 2^23 mantissas per exponent and both signs.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from flashoptim import reconstruct_fp32_param

TOTAL_FP32_MANTISSAS = 1 << 23
INT8_MAX = 127.0
INT16_MAX = 32767.0

COLOR_BASELINE = "0.55"
COLOR_FLOAT = "#c77b16"
COLOR_ULP8 = "#70bcd8"
COLOR_ULP16 = "#1f77b4"


@dataclass(frozen=True)
class DtypeConfig:
    name: str
    dtype: torch.dtype
    mantissa_bits: int
    exponent_bias: int
    exponent_min: int
    exponent_max: int
    x_limits: tuple[float, float]
    x_ticks: tuple[int, ...]
    y_limits: tuple[float, float]
    symlog_linthresh: float | None = None


BF16_CONFIG = DtypeConfig(
    name="BF16",
    dtype=torch.bfloat16,
    mantissa_bits=7,
    exponent_bias=127,
    exponent_min=-127,
    exponent_max=127,
    x_limits=(-128.0, 128.0),
    x_ticks=(-128, -64, 0, 64, 128),
    y_limits=(1e-10, 3e-2),
)

FP16_CONFIG = DtypeConfig(
    name="FP16",
    dtype=torch.float16,
    mantissa_bits=10,
    exponent_bias=15,
    exponent_min=-30,
    exponent_max=15,
    x_limits=(-30.0, 15.0),
    x_ticks=(-24, -16, -8, 0, 8),
    y_limits=(1e-10, 1.0),
    symlog_linthresh=1e-10,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FP32 reconstruction error across exponent buckets."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compression_comparison_torch.png",
        help="Output path for the rendered figure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--mantissa-count",
        type=int,
        default=TOTAL_FP32_MANTISSAS,
        help="Number of mantissas to evaluate per exponent (default: full 2^23).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=8,
        help="Print progress every N exponents.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for raster outputs.",
    )
    return parser.parse_args()


def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _stable_pow2_scale(x: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Compute x * 2**log_scale with stable exponent splitting.

    This mirrors the decomposition in flashoptim Triton kernels, avoiding
    overflow/underflow from materializing very large/small powers of two.
    """
    log_scale_f32 = log_scale.to(torch.float32)
    h = torch.floor(log_scale_f32 / 2.0)
    return x * torch.exp2(h) * torch.exp2(log_scale_f32 - h)


def _make_mantissa_fraction(
    count: int, *, device: torch.device
) -> torch.Tensor:
    if count == TOTAL_FP32_MANTISSAS:
        mantissa = torch.arange(count, device=device, dtype=torch.int32)
    else:
        mantissa = (
            torch.linspace(
                0,
                TOTAL_FP32_MANTISSAS - 1,
                steps=count,
                device=device,
            )
            .round()
            .to(torch.int32)
        )
    return mantissa.to(torch.float32) / float(TOTAL_FP32_MANTISSAS)


def _values_for_exponent(
    mantissa_fraction: torch.Tensor, exponent: int, sign: float
) -> torch.Tensor:
    if exponent == -127:
        base = mantissa_fraction * math.ldexp(1.0, -126)
    else:
        base = (1.0 + mantissa_fraction) * math.ldexp(1.0, exponent)
    return base if sign > 0 else -base


def _log_scale_from_narrow(
    x_narrow: torch.Tensor, cfg: DtypeConfig
) -> torch.Tensor:
    bits = x_narrow.abs().view(torch.uint16).to(torch.int32)
    exponent_bits = bits >> cfg.mantissa_bits
    unbiased_exponent = torch.where(
        exponent_bits == 0,
        torch.full_like(exponent_bits, 1 - cfg.exponent_bias),
        exponent_bits - cfg.exponent_bias,
    )
    return unbiased_exponent - cfg.mantissa_bits - 1


def _masked_relative_error_sum(
    x64: torch.Tensor,
    denom64: torch.Tensor,
    candidate: torch.Tensor,
    finite_mask: torch.Tensor,
) -> torch.Tensor:
    rel = (candidate.to(torch.float64) - x64).abs() / denom64
    rel = torch.where(finite_mask, rel, torch.zeros_like(rel))
    return rel.sum(dtype=torch.float64)


def _bucket_error_means(
    mantissa_fraction: torch.Tensor, exponent: int, cfg: DtypeConfig
) -> tuple[float, float, float, float]:
    sums = torch.zeros(4, dtype=torch.float64, device=mantissa_fraction.device)
    count = torch.tensor(0.0, dtype=torch.float64, device=mantissa_fraction.device)

    for sign in (1.0, -1.0):
        x = _values_for_exponent(mantissa_fraction, exponent, sign)
        x_narrow = x.to(cfg.dtype)
        finite_mask = torch.isfinite(x_narrow)
        finite_count = finite_mask.sum(dtype=torch.float64)
        if finite_count.item() == 0:
            continue

        x_recon = x_narrow.to(torch.float32)
        x64 = x.to(torch.float64)
        denom64 = torch.where(x64.abs() > 0.0, x64.abs(), torch.ones_like(x64))

        sums[0] += _masked_relative_error_sum(x64, denom64, x_recon, finite_mask)

        residual_float = (x - x_recon).to(cfg.dtype)
        recon_float = x_recon + residual_float.to(torch.float32)
        sums[1] += _masked_relative_error_sum(x64, denom64, recon_float, finite_mask)

        error = x - x_recon
        log_scale = _log_scale_from_narrow(x_narrow, cfg)
        # Numerical-stability decomposition matches kernel implementation.
        error_normalized = _stable_pow2_scale(error, -log_scale).clamp(-1.0, 1.0)

        ecc8 = _round_half_away_from_zero(error_normalized * INT8_MAX).to(torch.int8)
        recon_ulp8 = x_recon + _stable_pow2_scale(
            ecc8.to(torch.float32) / INT8_MAX,
            log_scale,
        )
        sums[2] += _masked_relative_error_sum(x64, denom64, recon_ulp8, finite_mask)

        ecc16 = _round_half_away_from_zero(error_normalized * INT16_MAX).to(torch.int16)
        # Use the package reconstruction kernel here so the figure matches the
        # exact arithmetic used by FlashOptim for the 32-bit master-weight path.
        recon_ulp16 = reconstruct_fp32_param(x_narrow, ecc16)
        sums[3] += _masked_relative_error_sum(x64, denom64, recon_ulp16, finite_mask)

        count += finite_count

    means = (sums / count).cpu().tolist()
    return tuple(float(v) for v in means)


def evaluate_dtype(
    cfg: DtypeConfig,
    mantissa_fraction: torch.Tensor,
    *,
    progress_every: int,
) -> tuple[list[int], dict[str, list[float]]]:
    exponents = list(range(cfg.exponent_min, cfg.exponent_max + 1))
    metrics = {
        "baseline": [],
        "float_buffer": [],
        "ulp_int8": [],
        "ulp_int16": [],
    }

    start = time.perf_counter()
    for idx, exponent in enumerate(exponents, start=1):
        baseline, float_buffer, ulp_int8, ulp_int16 = _bucket_error_means(
            mantissa_fraction,
            exponent,
            cfg,
        )
        metrics["baseline"].append(baseline)
        metrics["float_buffer"].append(float_buffer)
        metrics["ulp_int8"].append(ulp_int8)
        metrics["ulp_int16"].append(ulp_int16)

        if (
            idx == 1
            or idx == len(exponents)
            or (progress_every > 0 and idx % progress_every == 0)
        ):
            elapsed = time.perf_counter() - start
            print(
                f"{cfg.name}: {idx:3d}/{len(exponents)} exponents "
                f"(elapsed {elapsed:6.1f}s)"
            )

    return exponents, metrics


def _denormal_boundaries(cfg: DtypeConfig) -> tuple[int, int]:
    min_subnormal_exponent = 1 - cfg.exponent_bias - cfg.mantissa_bits
    max_subnormal_exponent = (1 - cfg.exponent_bias) - 1
    return min_subnormal_exponent, max_subnormal_exponent


def _plot_x_values(cfg: DtypeConfig, exponents: list[int]) -> list[float]:
    if cfg.name == "BF16" and exponents and exponents[0] == -127:
        return [-128.0, *[float(x) for x in exponents[1:]]]
    return [float(x) for x in exponents]


def _configure_y_axis(ax, cfg: DtypeConfig) -> None:
    if cfg.symlog_linthresh is None:
        ax.set_yscale("log")
        ax.set_ylim(cfg.y_limits[0], cfg.y_limits[1])
        return

    ax.set_yscale(
        "symlog",
        linthresh=cfg.symlog_linthresh,
        linscale=1.0,
        base=10,
    )
    ax.set_ylim(0.0, cfg.y_limits[1])
    ticks = [
        0.0,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ]
    labels = [
        "0",
        r"$10^{-10}$",
        r"$10^{-9}$",
        r"$10^{-8}$",
        r"$10^{-7}$",
        r"$10^{-6}$",
        r"$10^{-5}$",
        r"$10^{-4}$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
    ]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def plot_results(
    *,
    output_path: Path,
    dpi: int,
    bf16_exponents: list[int],
    bf16_metrics: dict[str, list[float]],
    fp16_exponents: list[int],
    fp16_metrics: dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 9.0), constrained_layout=True)

    panel_data = [
        (axes[0], BF16_CONFIG, bf16_exponents, bf16_metrics),
        (axes[1], FP16_CONFIG, fp16_exponents, fp16_metrics),
    ]

    for ax, cfg, exponents, metrics in panel_data:
        plot_x = _plot_x_values(cfg, exponents)
        ax.plot(
            plot_x,
            metrics["baseline"],
            linestyle="--",
            color=COLOR_BASELINE,
            linewidth=2.2,
            label=cfg.name,
            zorder=5,
        )
        ax.plot(
            plot_x,
            metrics["float_buffer"],
            color=COLOR_FLOAT,
            linewidth=2.4,
            label=f"{cfg.name} + {cfg.name}",
            zorder=3,
        )
        ax.plot(
            plot_x,
            metrics["ulp_int8"],
            color=COLOR_ULP8,
            linewidth=2.4,
            label=f"{cfg.name} + ULP int8",
            zorder=2,
        )
        ax.plot(
            plot_x,
            metrics["ulp_int16"],
            color=COLOR_ULP16,
            linewidth=2.4,
            label=f"{cfg.name} + ULP int16",
            zorder=4,
        )

        denorm_left, denorm_right = _denormal_boundaries(cfg)
        for x in (denorm_left, denorm_right):
            if exponents[0] <= x <= exponents[-1]:
                ax.axvline(
                    x=x,
                    linestyle=":",
                    color="0.70",
                    linewidth=1.2,
                    zorder=0,
                )

        ax.set_xlim(cfg.x_limits[0], cfg.x_limits[1])
        _configure_y_axis(ax, cfg)
        ax.set_xticks(list(cfg.x_ticks))
        ax.set_ylabel("Mean Relative Error", fontsize=15)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", fontsize=13, framealpha=0.95)
        ax.tick_params(axis="both", labelsize=13)

    axes[1].set_xlabel("Exponent", fontsize=15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mantissa_count <= 0:
        raise ValueError("mantissa-count must be positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use --device cuda to run the full exponent sweep efficiently.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    output_path = Path(args.output)

    print(f"Using device: {device}")
    print(f"Mantissas per exponent: {args.mantissa_count:,}")
    mantissa_fraction = _make_mantissa_fraction(args.mantissa_count, device=device)

    with torch.inference_mode():
        bf16_exponents, bf16_metrics = evaluate_dtype(
            BF16_CONFIG,
            mantissa_fraction,
            progress_every=args.progress_every,
        )
        fp16_exponents, fp16_metrics = evaluate_dtype(
            FP16_CONFIG,
            mantissa_fraction,
            progress_every=args.progress_every,
        )

    plot_results(
        output_path=output_path,
        dpi=args.dpi,
        bf16_exponents=bf16_exponents,
        bf16_metrics=bf16_metrics,
        fp16_exponents=fp16_exponents,
        fp16_metrics=fp16_metrics,
    )
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
