r"""Build the RQ1 score-distribution figure as two separately-referenceable PDFs.

The paper shows SPECTRA-Siam's learned latent spectrum above and the fixed AST
spectrum below, both without a trained downstream head. Each panel is its own
file so the paper can \subfloat them and cite utoref{fig:rq1_method} or
\autoref{fig:rq1_best_spectral} individually.

The two are still designed to be read as one stacked figure, so the shared
annotation lives on exactly one panel each: the top panel carries the legend
and no x-axis numbers, the bottom panel carries the x-axis label and numbers
and no legend. Both use the same x range and figure width so they line up when
typeset one above the other. Neither carries a title or metric annotation
because the paper supplies its own caption.

Scores come from the cache written by ``render_all_histograms.py``, so nothing
is retrained.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# (cache stem, output stem, carries the legend, carries the x-axis annotation).
# Top panel first: the paper stacks the learned spectrum above the fixed one.
PANELS = (
    ("latent_NoTrain", "latent_NoTrain", True, False),
    ("ast_NoTrain", "ast_NoTrain", False, True),
)
# Matches the stacked pair the single-file version drew, halved per panel.
PANEL_SIZE = (6.4, 2.8)


def panel(axis, labels: np.ndarray, scores: np.ndarray, show_xlabel: bool,
          show_legend: bool) -> None:
    for value, color, name in ((0, "#4C72B0", "Non-clone pairs"),
                               (1, "#C44E52", "Clone pairs")):
        subset = np.clip(scores[labels == value], 0.0, 1.0)
        if not len(subset):
            continue
        axis.hist(subset, bins=40, range=(0, 1), density=True, alpha=0.55,
                  color=color, label=name, edgecolor="white", linewidth=0.3)
        if subset.std() > 1e-6:
            kde = stats.gaussian_kde(subset)
            grid = np.linspace(0, 1, 400)
            axis.plot(grid, kde(grid), color=color, linestyle="--", linewidth=1.4)
    axis.set_xlim(0, 1)
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    # Both panels use the same two colours, so one legend covers the pair.
    if show_legend:
        axis.legend(fontsize=8.5, frameon=True, framealpha=0.9)
    if show_xlabel:
        axis.set_xlabel("Output score")
    else:
        # The panel below carries the axis: keep the ticks, drop the numbers.
        axis.tick_params(labelbottom=False)


def build(source: Path, destination: Path) -> list[Path]:
    cache = np.load(source / "scores.npz")
    missing = [stem for stem, *_ in PANELS if f"{stem}_scores" not in cache.files]
    if missing:
        raise SystemExit(f"{source/'scores.npz'} has no scores for {missing}")

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, name, show_legend, show_xlabel in PANELS:
        figure, axis = plt.subplots(figsize=PANEL_SIZE)
        panel(axis, cache[f"{stem}_labels"], cache[f"{stem}_scores"],
              show_xlabel=show_xlabel, show_legend=show_legend)
        figure.tight_layout()
        path = destination / f"{name}.pdf"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                       facecolor="white")
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="xglue", choices=("xglue", "atcoder"),
                        help="xglue is BigCloneBench, which the paper reports")
    parser.add_argument("--histograms-dir", type=Path,
                        default=_ROOT.parent / "outputs/kaggle/RQ1/histograms")
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "kaggle/latex/rq1/figures")
    args = parser.parse_args()

    source = args.histograms_dir / args.dataset
    if not (source / "scores.npz").is_file():
        raise SystemExit(f"no scores.npz under {source}")
    for path in build(source, args.output_dir):
        print(f"wrote {path}")
        print(f"      {path.with_suffix('.png')}")
    print(f"top: {PANELS[0][1]} (legend), bottom: {PANELS[1][1]} (x-axis), "
          f"from {args.dataset}")


if __name__ == "__main__":
    main()
