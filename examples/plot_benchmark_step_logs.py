"""Plot benchmark_step throughput and memory from benchmark logs."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


SHAPE_RE = re.compile(r"^shape=(\d+)x(\d+)\s+params=")
TIME_RE = re.compile(
    r"^\s+(jax baseline|jax flash|torch baseline|torch flash)\s+median=\s*([0-9.]+)ms"
)
MEMORY_RE = re.compile(
    r"^\s*(\d+)x(\d+)\s+([0-9.]+)\s+([KMG]B)\s+([0-9.]+)\s+([KMG]B)\s+([0-9.]+)\s+([KMG]B)\s+([0-9.]+)\s+([KMG]B)\s*$"
)
SERIES_STYLES = (
    ("jax baseline", "JAX baseline", "tab:blue", "--", "s"),
    ("jax flash", "JAX flash", "tab:blue", "-", "o"),
    ("torch baseline", "Torch baseline", "tab:orange", "--", "s"),
    ("torch flash", "Torch flash", "tab:orange", "-", "o"),
)
MEMORY_STYLES = (
    ("baseline", "Baseline", "tab:gray", "--", "s"),
    ("flash", "Flash", "tab:green", "-", "o"),
)
MEMORY_LABELS = (
    ("torch baseline", 2, 3),
    ("torch flash", 4, 5),
    ("jax baseline", 6, 7),
    ("jax flash", 8, 9),
)
UNIT_BYTES = {"KB": 1024.0, "MB": 1024.0**2, "GB": 1024.0**3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-x",
        "--x-range",
        nargs=2,
        type=int,
        metavar=("MIN_EXP", "MAX_EXP"),
        default=None,
        help="Only plot square sizes between 2^MIN_EXP and 2^MAX_EXP, inclusive.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("examples/out/benchmark_step_square"),
        help="Directory containing benchmark log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/out/benchmark_step_square/flash_timings.png"),
        help="Output path for the throughput plot image.",
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=Path("examples/out/benchmark_step_square/flash_memory.png"),
        help="Output path for the memory plot image.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--flash-only",
        action="store_true",
        help="For the throughput plot, only show JAX flash and Torch flash.",
    )
    return parser.parse_args()


def size_in_range(size: int, x_range: tuple[int, int] | None) -> bool:
    if x_range is None:
        return True
    min_exp, max_exp = x_range
    min_size = 2**min_exp
    max_size = 2**max_exp
    return min_size <= size <= max_size


def parse_size_bytes(value: str, unit: str) -> float:
    return float(value) * UNIT_BYTES[unit]


def parse_speed_log(path: Path, x_range: tuple[int, int] | None) -> dict[str, list[tuple[int, float]]]:
    data = {label: [] for label, _, _, _, _ in SERIES_STYLES}
    current_size: int | None = None

    for raw_line in path.read_text().splitlines():
        shape_match = SHAPE_RE.match(raw_line)
        if shape_match:
            m, n = map(int, shape_match.groups())
            current_size = m if m == n else None
            if current_size is not None and not size_in_range(current_size, x_range):
                current_size = None
            continue

        speed_match = TIME_RE.match(raw_line)
        if speed_match and current_size is not None:
            name, median_ms = speed_match.groups()
            data[name].append((current_size, 1000.0 / float(median_ms)))

    return data


def parse_memory_log(path: Path, x_range: tuple[int, int] | None) -> dict[str, list[tuple[int, float]]]:
    data = {label: [] for label, _, _, _, _ in MEMORY_STYLES}

    for raw_line in path.read_text().splitlines():
        memory_match = MEMORY_RE.match(raw_line)
        if not memory_match:
            continue
        m, n = map(int, memory_match.groups()[:2])
        if m != n or not size_in_range(m, x_range):
            continue
        groups = memory_match.groups()
        values = {
            label: parse_size_bytes(groups[value_idx], groups[unit_idx])
            for label, value_idx, unit_idx in MEMORY_LABELS
        }
        if values["jax baseline"] != values["torch baseline"]:
            raise AssertionError(f"baseline memory mismatch at size {m} in {path}")
        if values["jax flash"] != values["torch flash"]:
            raise AssertionError(f"flash memory mismatch at size {m} in {path}")
        data["baseline"].append((m, values["jax baseline"]))
        data["flash"].append((m, values["jax flash"]))

    return data


def plot_grid(
    all_series: dict[tuple[str, str], dict[str, list[tuple[int, float]]]],
    *,
    output: Path,
    title: str,
    ylabel: str,
    log_y: bool,
    styles: tuple[tuple[str, str, str, str, str], ...],
    annotate_last_ratio: bool = False,
) -> None:
    optimizers = ("adamw", "sgd", "lion")
    weights = ("bf16", "fp16", "fp32")
    output.parent.mkdir(parents=True, exist_ok=True)

    if not any(
        all_series[(optimizer, weight)][label]
        for optimizer in optimizers
        for weight in weights
        for label, _, _, _, _ in styles
    ):
        return

    fig, axes = plt.subplots(len(optimizers), len(weights), figsize=(12, 9), sharex=True, sharey="row")

    for row, optimizer in enumerate(optimizers):
        row_ys: list[float] = []
        for weight in weights:
            series = all_series[(optimizer, weight)]
            for label, _, _, _, _ in styles:
                row_ys.extend(steps_per_sec for _, steps_per_sec in series[label])

        if not row_ys:
            for col in range(len(weights)):
                axes[row][col].set_visible(False)
            continue

        row_ymin = min(row_ys) * 0.9
        row_ymax = max(row_ys) * 1.05

        for col, weight in enumerate(weights):
            ax = axes[row][col]
            series = all_series[(optimizer, weight)]

            for label, display_label, color, linestyle, marker in styles:
                points = series[label]
                xs = [size for size, _ in points]
                ys = [steps_per_sec for _, steps_per_sec in points]
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    markersize=6,
                    linewidth=2,
                    linestyle=linestyle,
                    color=color,
                    label=display_label,
                )

            if annotate_last_ratio:
                baseline_points = dict(series["baseline"])
                flash_points = dict(series["flash"])
                common_xs = sorted(set(baseline_points) & set(flash_points))
                if common_xs:
                    fill_xs = common_xs
                    fill_baseline = [baseline_points[x] for x in fill_xs]
                    fill_flash = [flash_points[x] for x in fill_xs]
                    ax.fill_between(
                        fill_xs, fill_flash, fill_baseline,
                        alpha=0.13, color="tab:red", zorder=0,
                    )
                    last_x = common_xs[-1]
                    ratio = baseline_points[last_x] / flash_points[last_x]
                    y_text = (baseline_points[last_x] * flash_points[last_x]) ** 0.5 if log_y else 0.5 * (baseline_points[last_x] + flash_points[last_x])
                    ax.text(
                        last_x, y_text, f"{ratio:.2f}x",
                        ha="right", va="center", fontsize=10, fontweight="bold",
                        color="tab:red",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
                    )

            ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            ax.set_ylim(row_ymin, row_ymax)
            ax.set_title(f"{optimizer} / {weight}")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.suptitle(title, y=0.99)
    fig.supxlabel("matrix size (2^n)")
    fig.supylabel(ylabel)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.90))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    optimizers = ("adamw", "sgd", "lion")
    weights = ("bf16", "fp16", "fp32")
    speed_styles = (
        tuple(style for style in SERIES_STYLES if style[0] in {"jax flash", "torch flash"})
        if args.flash_only
        else SERIES_STYLES
    )
    all_speed_series: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = {}
    all_memory_series: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = {}

    for optimizer in optimizers:
        for weight in weights:
            path = args.input_dir / f"{optimizer}-{weight}.log"
            all_speed_series[(optimizer, weight)] = parse_speed_log(path, args.x_range)
            all_memory_series[(optimizer, weight)] = parse_memory_log(path, args.x_range)

    plot_grid(
        all_speed_series,
        output=args.output,
        title=(
            "benchmark_step square matrices: JAX flash vs Torch flash"
            if args.flash_only
            else "benchmark_step square matrices: JAX vs Torch baselines and flash"
        ),
        ylabel="throughput (steps/s)",
        log_y=args.log,
        styles=speed_styles,
    )
    plot_grid(
        all_memory_series,
        output=args.memory_output,
        title="benchmark_step square matrices: optimizer memory (JAX = Torch)",
        ylabel="memory (bytes)",
        log_y=args.log,
        styles=MEMORY_STYLES,
        annotate_last_ratio=True,
    )


if __name__ == "__main__":
    main()
