"""Reproduce the FP32 reconstruction-error figure (Figure 3) using flashoptim_jax."""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from flashoptim_jax import reconstruct_leaf, split_leaf
from flashoptim_jax.compression import _log_half_ulp

TOTAL_FP32_MANTISSAS = 1 << 23

COLOR_BASELINE = "0.55"
COLOR_FLOAT = "#c77b16"
COLOR_ULP8 = "#70bcd8"
COLOR_ULP16 = "#1f77b4"


@dataclass(frozen=True)
class DtypeConfig:
    name: str
    narrow_dtype: jnp.dtype
    exponent_bias: int
    mantissa_bits: int
    exponent_min: int
    exponent_max: int
    x_limits: tuple[float, float]
    x_ticks: tuple[int, ...]
    y_limits: tuple[float, float]
    symlog_linthresh: float | None = None


BF16_CONFIG = DtypeConfig(
    name="BF16",
    narrow_dtype=jnp.bfloat16,
    exponent_bias=127,
    mantissa_bits=7,
    exponent_min=-127,
    exponent_max=127,
    x_limits=(-128.0, 128.0),
    x_ticks=(-128, -64, 0, 64, 128),
    y_limits=(1e-10, 3e-2),
)

FP16_CONFIG = DtypeConfig(
    name="FP16",
    narrow_dtype=jnp.float16,
    exponent_bias=15,
    mantissa_bits=10,
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
        default="compression_comparison_jax.png",
        help="Output path for the rendered figure.",
    )
    parser.add_argument(
        "--mantissa-count",
        type=int,
        default=TOTAL_FP32_MANTISSAS,
        help="Number of mantissas to evaluate per exponent.",
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


def _make_mantissa_fraction(count: int) -> jax.Array:
    if count == TOTAL_FP32_MANTISSAS:
        mantissa = jnp.arange(count, dtype=jnp.int32)
    else:
        mantissa = jnp.rint(
            jnp.linspace(0, TOTAL_FP32_MANTISSAS - 1, num=count, dtype=jnp.float32)
        ).astype(jnp.int32)
    return mantissa.astype(jnp.float32) / float(TOTAL_FP32_MANTISSAS)


def _values_for_exponent(
    mantissa_fraction: jax.Array,
    exponent: jax.Array,
    sign: float,
) -> jax.Array:
    exponent_f32 = exponent.astype(jnp.float32)
    normal = (1.0 + mantissa_fraction) * jnp.exp2(exponent_f32)
    fp32_subnormal = mantissa_fraction * math.ldexp(1.0, -126)
    values = jnp.where(exponent == -127, fp32_subnormal, normal)
    return values if sign > 0 else -values


def _masked_relative_error_sum(
    target: jax.Array,
    candidate: jax.Array,
    finite_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    target64 = target.astype(jnp.float64)
    candidate64 = candidate.astype(jnp.float64)
    denom = jnp.where(jnp.abs(target64) > 0.0, jnp.abs(target64), 1.0)
    rel = jnp.where(finite_mask, jnp.abs(candidate64 - target64) / denom, 0.0)
    return jnp.sum(rel, dtype=jnp.float64), jnp.sum(finite_mask, dtype=jnp.float64)


def _reconstruct_leaf_exact(theta_lp: jax.Array, rho: jax.Array) -> jax.Array:
    # The generic JAX float32 reconstruction path is slightly noisier than the
    # intended arithmetic for BF16+int16. Use float64 intermediates here so the
    # figure reflects the exact split representation rather than extra rounding
    # in the diagnostic itself.
    if jnp.asarray(rho).dtype != jnp.int16:
        return reconstruct_leaf(theta_lp, rho)
    log_scale = _log_half_ulp(theta_lp)
    correction = rho.astype(jnp.float64) / jnp.float64(32767.0)
    error = jnp.ldexp(correction, log_scale)
    return (theta_lp.astype(jnp.float64) + error).astype(jnp.float32)


def _evaluate_bucket(
    cfg: DtypeConfig,
    mantissa_fraction: jax.Array,
    exponent: int,
) -> tuple[float, float, float, float]:
    # These errors are sign-symmetric, so sweeping positive values alone yields
    # the same mean relative error as averaging over both signs.
    theta = _values_for_exponent(mantissa_fraction, jnp.asarray(exponent), 1.0)
    theta_lp = theta.astype(cfg.narrow_dtype)
    finite_mask = jnp.isfinite(theta_lp)

    recon_plain = theta_lp.astype(jnp.float32)
    baseline_sum, count = _masked_relative_error_sum(theta, recon_plain, finite_mask)

    residual_lp = (theta - recon_plain).astype(cfg.narrow_dtype)
    recon_float = recon_plain + residual_lp.astype(jnp.float32)
    float_sum, _ = _masked_relative_error_sum(theta, recon_float, finite_mask)

    theta_lp_24, rho_24 = split_leaf(
        theta,
        narrow_dtype=cfg.narrow_dtype,
        master_weight_bits=24,
    )
    recon_24 = reconstruct_leaf(theta_lp_24, rho_24)
    ulp8_sum, _ = _masked_relative_error_sum(theta, recon_24, finite_mask)

    theta_lp_32, rho_32 = split_leaf(
        theta,
        narrow_dtype=cfg.narrow_dtype,
        master_weight_bits=32,
    )
    recon_32 = _reconstruct_leaf_exact(theta_lp_32, rho_32)
    ulp16_sum, _ = _masked_relative_error_sum(theta, recon_32, finite_mask)

    values = jax.device_get(
        jnp.asarray(
            [
                baseline_sum / count,
                float_sum / count,
                ulp8_sum / count,
                ulp16_sum / count,
            ],
            dtype=jnp.float64,
        )
    )
    return tuple(float(v) for v in values)


def evaluate_dtype(
    cfg: DtypeConfig,
    mantissa_fraction: jax.Array,
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
        baseline, float_buffer, ulp_int8, ulp_int16 = _evaluate_bucket(
            cfg,
            mantissa_fraction,
            exponent,
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
            if cfg.x_limits[0] <= x <= cfg.x_limits[1]:
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

    print(f"Devices: {[str(device) for device in jax.devices()]}")
    print(f"Mantissas per exponent: {args.mantissa_count:,}")

    mantissa_fraction = _make_mantissa_fraction(args.mantissa_count)
    output_path = Path(args.output)

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
