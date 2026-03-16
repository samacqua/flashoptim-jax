"""3×3 grid: rows = GPT-2 / MNIST / ImageNette, columns = AdamW / Lion / SGD.

Each panel shows training loss for all available conditions
(JAX Flash, JAX Reference, Torch Flash, Torch Reference) with a zoom inset.

Usage:
    python examples/plot_all_benchmarks_3x3.py
    python examples/plot_all_benchmarks_3x3.py --jax-only
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxConnector,
    BboxPatch,
    _TransformedBboxWithCallback,
)

COLORS = {
    "JAX Flash": "#2563eb",
    "JAX Ref": "#f97316",
    "Torch Flash": "#16a34a",
    "Torch Ref": "#dc2626",
}
STYLES = {
    "JAX Flash": "-",
    "JAX Ref": "--",
    "Torch Flash": "-",
    "Torch Ref": "--",
}
ALPHAS = {
    "JAX Flash": 1.0,
    "JAX Ref": 0.9,
    "Torch Flash": 0.9,
    "Torch Ref": 0.8,
}

OUTPUT = Path("assets/all-benchmarks-3x3.png")

# ── data layout ──────────────────────────────────────────────────────────
DATA_ROOT = Path("examples/out-final")
GPT2_DIR = DATA_ROOT / "gpt2"
MNIST_DIR = DATA_ROOT / "mnist"
IMAGENET_DIR = DATA_ROOT / "imagenet"

JAX_ONLY = False
LOG_SCALE = False

OPTIMIZERS = ["adamw", "lion", "sgd"]
OPT_TITLES = {"adamw": "AdamW", "lion": "Lion", "sgd": "SGD"}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


GPT2_MAX_STEP = 10_000


def _is_jax(label: str) -> bool:
    return label.startswith("JAX")


def _load_gpt2(opt: str) -> list[tuple[str, list[float], list[float]]]:
    series = []
    for label, filename in [
        ("JAX Flash", f"{opt}-flash-jax.jsonl"),
        ("JAX Ref", f"{opt}-reference-jax.jsonl"),
    ]:
        p = GPT2_DIR / filename
        if not p.exists():
            continue
        xs, ys = [], []
        for r in _load_jsonl(p):
            if r.get("event") != "train":
                continue
            s = r.get("step", 0)
            if s < 10 or s > GPT2_MAX_STEP:
                continue
            xs.append(float(s))
            ys.append(float(r["loss"]))
        if xs:
            series.append((label, xs, ys))
    return series


def _load_mnist(opt: str) -> list[tuple[str, list[float], list[float]]]:
    specs = [
        ("JAX Flash", MNIST_DIR / f"{opt}-flash-jax.jsonl"),
        ("JAX Ref", MNIST_DIR / f"{opt}-reference-jax.jsonl"),
        ("Torch Flash", MNIST_DIR / f"{opt}-flash-torch.jsonl"),
        ("Torch Ref", MNIST_DIR / f"{opt}-reference-torch.jsonl"),
    ]
    series = []
    for label, p in specs:
        if JAX_ONLY and not _is_jax(label):
            continue
        if not p.exists():
            continue
        xs, ys = [], []
        spe = None
        for r in _load_jsonl(p):
            if r.get("event") == "config":
                continue
            if r.get("event") != "train" or "loss" not in r:
                continue
            if spe is None:
                spe = r.get("steps_per_epoch", 235)
            epoch = r.get("epoch", 1)
            step = r.get("step_in_epoch") or r.get("step", 1)
            gs = (epoch - 1) * spe + step
            if gs < 20 or gs > 10_000:
                continue
            xs.append(float(gs))
            ys.append(float(r["loss"]))
        if xs:
            series.append((label, xs, ys))
    return series


def _load_imagenet(opt: str) -> list[tuple[str, list[float], list[float]]]:
    specs = [
        ("JAX Flash", IMAGENET_DIR / f"{opt}-flash-jax.jsonl"),
        ("JAX Ref", IMAGENET_DIR / f"{opt}-reference-jax.jsonl"),
        ("Torch Flash", IMAGENET_DIR / f"{opt}-flash-torch.jsonl"),
        ("Torch Ref", IMAGENET_DIR / f"{opt}-reference-torch.jsonl"),
    ]
    series = []
    for label, p in specs:
        if JAX_ONLY and not _is_jax(label):
            continue
        if not p.exists():
            continue
        xs, ys = [], []
        for r in _load_jsonl(p):
            if r.get("event") != "train":
                continue
            gs = r.get("global_step", 0)
            if gs < 10:
                continue
            xs.append(float(gs))
            ys.append(float(r["loss"]))
        if xs:
            series.append((label, xs, ys))
    return series


# ── styling ──────────────────────────────────────────────────────────────

def _plain_fmt(x, _pos):
    if x == 0:
        return "0"
    if x == int(x) and abs(x) < 1e6:
        return f"{int(x):,}".replace(",", "\u2009")
    return f"{x:g}"


def _style_ax(ax, tick_size=11):
    ax.set_facecolor("white")
    ax.grid(True, color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=tick_size, length=0)
    for spine in ax.spines.values():
        spine.set_color("#c0c0c0")
        spine.set_linewidth(1.0)
    fmt = FuncFormatter(_plain_fmt)
    ax.xaxis.set_major_formatter(fmt)
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    else:
        ax.yaxis.set_major_formatter(fmt)


def _zoom_limits(series, x0, x1):
    ys_in = []
    for _, xs, ys in series:
        for x, y in zip(xs, ys):
            if x0 <= x <= x1:
                ys_in.append(y)
    if not ys_in:
        return 0, 1
    lo, hi = min(ys_in), max(ys_in)
    margin = (hi - lo) * 0.08 or 0.01
    return lo - margin, hi + margin


def _add_zoom(ax, series, zoom):
    x0, x1 = zoom
    y0, y1 = _zoom_limits(series, x0, x1)

    axins = ax.inset_axes([0.38, 0.42, 0.58, 0.54])
    for label, xs, ys in series:
        axins.plot(xs, ys, linewidth=1.2, color=COLORS[label],
                   linestyle=STYLES[label], alpha=ALPHAS[label])
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    _style_ax(axins, tick_size=8)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in axins.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(1.3)

    zoom_box = BboxPatch(
        _TransformedBboxWithCallback(
            axins.viewLim, ax.transData, callback=ax._unstale_viewLim
        ),
        fc="none", ec="0.6", linestyle="--", linewidth=1.0,
    )
    ax.add_patch(zoom_box)
    for loc1, loc2 in [(3, 2), (4, 1)]:
        conn = BboxConnector(
            axins.bbox, zoom_box.bbox,
            loc1=loc1, loc2=loc2, fc="none", ec="0.6",
            linestyle="--", linewidth=1.0,
        )
        axins.add_patch(conn)
        conn.set_clip_on(False)
        conn.set_zorder(10)
    zoom_box.set_zorder(10)


def _plot_panel(ax, series, zoom):
    for label, xs, ys in series:
        ax.plot(xs, ys, linewidth=1.8, label=label,
                color=COLORS[label], linestyle=STYLES[label],
                alpha=ALPHAS[label], zorder=3)
    if LOG_SCALE:
        ax.set_yscale("log")
    _style_ax(ax)
    if zoom and series:
        _add_zoom(ax, series, zoom)


# ── zoom ranges ──────────────────────────────────────────────────────────
GPT2_ZOOM = (8_000, 10_000)
MNIST_ZOOM = (8_000, 10_000)
IMAGENET_ZOOM = (4_000, 5_000)

ROW_LABELS = ["GPT-2 Pretrain", "MNIST", "ImageNette"]
LOADERS = [_load_gpt2, _load_mnist, _load_imagenet]
ZOOMS = [GPT2_ZOOM, MNIST_ZOOM, IMAGENET_ZOOM]


def main() -> None:
    global JAX_ONLY, LOG_SCALE

    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-only", action="store_true",
                        help="Only plot JAX Flash and JAX Ref series.")
    parser.add_argument("--log", action="store_true",
                        help="Use log scale on y-axis.")
    args = parser.parse_args()
    JAX_ONLY = args.jax_only
    LOG_SCALE = args.log

    plt.rcParams.update({
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "font.family": "sans-serif",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.patch.set_facecolor("white")

    title = "Training Loss: Flash vs Reference"
    if JAX_ONLY:
        title += " (JAX)"
    fig.suptitle(title, fontsize=24, fontweight="bold", y=1.02)

    all_handles, all_labels = {}, {}

    for row_idx, (loader, zoom, row_label) in enumerate(
        zip(LOADERS, ZOOMS, ROW_LABELS)
    ):
        for col_idx, opt in enumerate(OPTIMIZERS):
            ax = axes[row_idx, col_idx]
            series = loader(opt)
            _plot_panel(ax, series, zoom)

            if row_idx == 0:
                ax.set_title(OPT_TITLES[opt], fontweight="bold", fontsize=20, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nloss", fontsize=16)
            if row_idx == 2:
                ax.set_xlabel("step", fontsize=15)

            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in all_labels:
                    all_handles[l] = h
                    all_labels[l] = l

    legend_order = ["JAX Flash", "JAX Ref", "Torch Flash", "Torch Ref"]
    handles = [all_handles[k] for k in legend_order if k in all_handles]
    labels = [k for k in legend_order if k in all_handles]

    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(labels),
        framealpha=0.95, fancybox=True,
        facecolor="white", edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.99), fontsize=16,
    )

    fig.subplots_adjust(hspace=0.18, wspace=0.18, top=0.93, bottom=0.05,
                        left=0.06, right=0.98)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT}")


if __name__ == "__main__":
    main()
