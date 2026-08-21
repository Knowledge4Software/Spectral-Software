"""Build the Section 6.1 hyperparameter-sensitivity deliverables.

The paper carries this discussion as a single four-panel figure, so the
generated output is:

  * the combined figure under the exact filename the paper includes
  * each panel as its own PDF, for picking or rearranging
  * the figure block copied from the paper template
  * a supplementary table of final-epoch values (not in the paper; it makes the
    numbers behind the curves citable)
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

D1_ROOT = _ROOT.parent / "outputs" / "kaggle" / "d1"
COMBINED_NAME = "rq_hyperparameter_sensitivity_loss_only_clean.pdf"

# csv stem -> (panel title, legend template, LaTeX symbol)
SWEEPS = (
    ("latent_graph_size", "Latent graph size $m$", "$m = {v}$", "$m$"),
    ("assignment_iterations",
     "Latent-assignment iterations $I_{\\mathrm{assign}}$",
     "$I_{{\\mathrm{{assign}}}} = {v}$", "$I_{\\mathrm{assign}}$"),
    ("chebyshev_order", "Chebyshev order $K_{\\mathrm{cheb}}$",
     "$K_{{\\mathrm{{cheb}}}} = {v}$", "$K_{\\mathrm{cheb}}$"),
    ("batch_size", "Batch size", "$B = {v}$", "$B$"),
)
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def load(work: Path) -> dict[str, pd.DataFrame]:
    work.mkdir(parents=True, exist_ok=True)
    for archive in sorted(D1_ROOT.glob("*.zip")):
        target = work / archive.stem
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
    frames: dict[str, pd.DataFrame] = {}
    for path in work.rglob("sensitivity_*.csv"):
        frames[path.stem.replace("sensitivity_", "")] = pd.read_csv(path)
    return frames


def draw(axis, frame: pd.DataFrame, title: str, legend: str, show_xlabel: bool) -> None:
    for index, value in enumerate(sorted(frame.Value.unique())):
        arm = frame[frame.Value.eq(value)].sort_values("Epoch")
        color = COLORS[index % len(COLORS)]
        axis.plot(arm.Epoch, arm.TrainBCE, marker="o", markersize=3.2,
                  color=color, linewidth=1.5, label=legend.format(v=value))
        axis.plot(arm.Epoch, arm.ValidBCE, linestyle="--", color=color, linewidth=1.5)
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_ylabel("BCE loss")
    axis.set_xticks(sorted(frame.Epoch.unique()))
    if show_xlabel:
        axis.set_xlabel("Epoch")
    else:
        # Stacked panels share one x-axis: keep the gridlines, drop the tick
        # numbers so only the bottom panel is annotated.
        axis.tick_params(labelbottom=False)
    axis.grid(alpha=0.3, linestyle=":")
    axis.legend(ncol=2, fontsize=7.5, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path,
                        default=Path.home() / "AppData/Local/Temp/d1_latex_extract")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "kaggle" / "latex" / "d1")
    parser.add_argument("--template", type=Path, default=_ROOT / "kaggle" / "lecture.txt")
    args = parser.parse_args()

    frames = load(args.work_dir)
    missing = [stem for stem, *_ in SWEEPS if stem not in frames]
    print(f"loaded {len(frames)} sweeps" + (f"; missing {missing}" if missing else ""))

    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Only the combined figure is emitted: the paper includes that one file,
    # and separate per-panel PDFs would each repeat the shared x-axis label.
    # Combined figure, stacked in the paper's order, sharing one legend style.
    available = [item for item in SWEEPS if item[0] in frames]
    figure, axes = plt.subplots(len(available), 1, sharex=True,
                                figsize=(6.4, 2.35 * len(available)))
    for index, (stem, title, legend, _symbol) in enumerate(available):
        draw(axes[index], frames[stem], title, legend, show_xlabel=index == len(available) - 1)
    handles = [
        plt.Line2D([], [], color="black", linestyle="-", label="Train"),
        plt.Line2D([], [], color="black", linestyle="--", label="Validation"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(figures / COMBINED_NAME, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # Figure block, copied from the paper.
    text = args.template.read_text(encoding="utf-8", errors="ignore")
    # Anchor on the label first, then walk back to that figure's own opening;
    # searching forward from the first \begin{figure} would swallow the whole
    # document up to this point.
    anchor = text.find("\\label{fig:hyperparameter_sensitivity}")
    if anchor == -1:
        block = "% figure block not found in template"
    else:
        start = text.rfind("\\begin{figure}", 0, anchor)
        end = text.find("\\end{figure}", anchor)
        block = text[start:end + len("\\end{figure}")]
    (args.output_dir / "figure_d1.tex").write_text(
        "% Copied from the paper template; the figure file is generated by\n"
        "% research/discussions/build_d1_latex.py into figures/.\n" + block + "\n", encoding="utf-8")

    # Supplementary table of the numbers behind the curves.
    lines = [
        "% Supplementary: final-epoch values behind the sensitivity figure.",
        "% Not part of the paper template.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Final-epoch training and validation BCE loss, and test accuracy, for "
        "each hyperparameter setting in \\autoref{fig:hyperparameter_sensitivity}. "
        "All runs use CodeNet with five epochs per arm.}",
        "\\label{tab:d1_sensitivity}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.2pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "\\textbf{Hyperparameter} & \\textbf{Value} & \\textbf{Train BCE} "
        "& \\textbf{Valid BCE} & \\textbf{Test Acc.} \\\\",
        "\\midrule",
    ]
    for position, (stem, _title, _legend, symbol) in enumerate(available):
        frame = frames[stem]
        final = frame.sort_values("Epoch").groupby("Value").last()
        best = final.ValidBCE.idxmin()
        for order, (value, row) in enumerate(final.iterrows()):
            name = f"\\multirow{{{len(final)}}}{{*}}{{{symbol}}}" if order == 0 else ""
            marker = "\\textbf{" if value == best else ""
            close = "}" if marker else ""
            lines.append(
                f"{name} & {marker}{value}{close} & {marker}{row.TrainBCE:.4f}{close} "
                f"& {marker}{row.ValidBCE:.4f}{close} & {marker}{row.TestAccuracy:.4f}{close} \\\\"
            )
        lines.append("\\midrule" if position < len(available) - 1 else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]
    (args.output_dir / "table_d1_sensitivity.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.output_dir/'figure_d1.tex'}")
    print(f"wrote {args.output_dir/'table_d1_sensitivity.tex'}")
    print(f"wrote {figures / COMBINED_NAME} ({len(available)} stacked panels)")


if __name__ == "__main__":
    main()
