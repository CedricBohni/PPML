"""Plot the median speed-up of each IPFE decryption strategy over the sequential baseline.

Writes ``docs/speedup-light.png`` and ``docs/speedup-dark.png`` (one per GitHub theme).

Usage::

    uv run python scripts/plot_results.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import RESULTS_DIR, VARIANTS, load_runs  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
BASELINE = "ipfe"
STRATEGIES = [
    "ipfe_kernel_patches_parallelization",
    "ipfe_kernel_parallelization",
    "ipfe_batch_parallelization",
    "ipfe_batch_kernels_parallelization",
]

THEMES = {
    "light": {"surface": "#ffffff", "text": "#191919", "muted": "#666666", "grid": "#e5e5e5", "bar": "#257de0"},
    "dark": {"surface": "#0d1117", "text": "#efefef", "muted": "#9c9c9c", "grid": "#30363d", "bar": "#338ced"},
}


def median_speedups(test_size: int = 64) -> tuple[list[float], int]:
    table = load_runs(RESULTS_DIR, test_size)
    speedups = {s: [] for s in STRATEGIES}
    models = set()
    for tag, runs in table.items():
        if BASELINE not in runs:
            continue
        for s in STRATEGIES:
            if s in runs:
                speedups[s].append(runs[BASELINE].eval_seconds / runs[s].eval_seconds)
                models.add(tag)
    return [statistics.median(speedups[s]) for s in STRATEGIES], len(models)


def plot(values: list[float], n_models: int, theme: str) -> Path:
    t = THEMES[theme]
    labels = [VARIANTS[s] for s in STRATEGIES]

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    bars = ax.barh(labels, values, height=0.55, color=t["bar"], zorder=2)
    for bar, v in zip(bars, values):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.2f}×", va="center", color=t["text"], fontsize=10)

    ax.axvline(1, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(1.03, len(values) - 0.45, "sequential baseline (1×)", color=t["muted"], fontsize=8.5, va="bottom")

    ax.set_xlim(0, max(values) * 1.18)
    ax.set_ylim(-0.5, len(values) - 0.1)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9.5)
    ax.tick_params(axis="y", colors=t["text"])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}×")

    fig.suptitle(
        "Encrypted inference: median speed-up over sequential IPFE decryption",
        x=0.02,
        ha="left",
        color=t["text"],
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_title(
        f"{n_models} CNN configurations, 64 MNIST test images each",
        loc="left",
        color=t["muted"],
        fontsize=9.5,
        pad=8,
    )
    fig.tight_layout()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / f"speedup-{theme}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def main() -> None:
    values, n_models = median_speedups()
    for label, v in zip(STRATEGIES, values):
        print(f"{VARIANTS[label]:<35} {v:.2f}x")
    for theme in THEMES:
        print("wrote", plot(values, n_models, theme))


if __name__ == "__main__":
    main()
