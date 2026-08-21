"""Build the Section 6.2 epoch-study deliverables.

Reads the per-epoch run and writes:

  * the figure under the exact filename the paper includes
  * the figure block copied from the paper template
  * Table 12 (per-epoch test accuracy), matching the paper's template
  * the statistics the surrounding prose needs
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

D2_ROOT = _ROOT.parent / "outputs" / "kaggle" / "d2"
FIGURE_NAME = "d2_number_of_epochs_codenet_loss_only_clean.pdf"

# Clean-data bucket -> the label the paper's table header uses.
SUBSETS = (
    ("java", "Java--Java", True),
    ("python", "Python--Python", True),
    ("cpp", "C++--C++", True),
    ("csharp", "C\\#--C\\#", True),
    ("python_java", "Java--Python", False),
    ("java_cpp", "Java--C++", False),
    ("java_csharp", "Java--C\\#", False),
    ("python_cpp", "Python--C++", False),
    ("python_csharp", "Python--C\\#", False),
    ("cpp_csharp", "C++--C\\#", False),
)
PLAIN = {
    "Java--Java": "Java–Java", "Python--Python": "Python–Python",
    "C++--C++": "C++–C++", "C\\#--C\\#": "C#–C#",
    "Java--Python": "Java–Python", "Java--C++": "Java–C++",
    "Java--C\\#": "Java–C#", "Python--C++": "Python–C++",
    "Python--C\\#": "Python–C#", "C++--C\\#": "C++–C#",
}


def load(work: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Unpack the run and return the loss and accuracy frames."""
    work.mkdir(parents=True, exist_ok=True)
    for archive in sorted(D2_ROOT.glob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(work)
    loss_files = sorted(work.rglob("epoch_study_validation_bce.csv"))
    if not loss_files:
        raise SystemExit(f"no epoch_study_validation_bce.csv under {work}")
    loss = pd.read_csv(loss_files[0]).sort_values("Epoch").reset_index(drop=True)
    accuracy_files = sorted(work.rglob("epoch_study_test_accuracy.csv"))
    accuracy = (pd.read_csv(accuracy_files[0]).sort_values("Epoch").reset_index(drop=True)
                if accuracy_files else None)
    return loss, accuracy


def detect_plateau(total: np.ndarray, tolerance: float = 0.01) -> int:
    """First epoch after which under ``tolerance`` of the total gain remains."""
    improvement = total[0] - total.min()
    if improvement <= 0:
        return int(len(total))
    for index in range(len(total)):
        if (total[index] - total.min()) <= tolerance * improvement:
            return int(index + 1)
    return int(len(total))


def draw(frame: pd.DataFrame, plateau: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 3.6))
    palette = plt.cm.tab10.colors
    for index, (bucket, label, same_language) in enumerate(SUBSETS):
        if bucket not in frame.columns:
            continue
        axis.plot(frame.Epoch, frame[bucket], label=PLAIN[label],
                  color=palette[index % len(palette)],
                  linestyle="-" if same_language else "--", linewidth=1.3)
    axis.plot(frame.Epoch, frame.Total, label="Total", color="black", linewidth=2.4)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation BCE loss")
    step = 1 if len(frame) <= 20 else 2
    axis.set_xticks(list(frame.Epoch)[::step])
    axis.grid(alpha=0.25, linestyle=":")

    # Headroom above the curves for the legend, which sits in the top band.
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + (high - low) * 0.26)
    axis.legend(ncol=6, fontsize=7, frameon=False, loc="upper center")

    # Stop the plateau line below the legend band instead of running the full
    # height, so the dotted line never crosses the legend text.
    line_top = 0.74
    axis.axvline(plateau, ymin=0, ymax=line_top,
                 color="tab:blue", linestyle=":", linewidth=1.2)
    # Label it just above where the line stops. Near the right edge the text
    # would run off the canvas, so anchor it on the other side of the line.
    last_epoch = int(frame.Epoch.max())
    near_edge = plateau > last_epoch - 0.2 * last_epoch
    axis.annotate(
        f"Plateau (epoch {plateau})",
        xy=(plateau, low + (high - low) * line_top),
        xytext=(-6 if near_edge else 6, 4), textcoords="offset points",
        fontsize=8, color="tab:blue",
        ha="right" if near_edge else "left", va="bottom",
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def four(value: float) -> str:
    text = f"{float(value):.4f}"
    return text if text.startswith("1") else text.lstrip("0")


def accuracy_table(frame: pd.DataFrame) -> str:
    """Table 12, matching the paper's template."""
    best = int(frame.Total.idxmax())
    lines = [
        "% Generated by research/discussions/build_d2_latex.py -- structure matches the paper template.",
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\caption{Test accuracy of \\method{} on CodeNet after each epoch, reported on the "
        "full test set (\\emph{Total}) and separately for each language configuration.}",
        "\\label{tab:d2_epochs}",
        "",
        "\\tiny",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\renewcommand{\\arraystretch}{1.5}",
        "",
        "\\begin{tabular}{|c*{11}{c}|}",
        "\\hline",
        "\\textbf{Epoch}",
        "& \\textbf{Total}",
        *[f"& \\textbf{{{label}}}" for _bucket, label, _same in SUBSETS],
        "\\\\",
        "\\hline",
        "",
    ]
    for position, row in frame.iterrows():
        bold = position == best
        def render(value) -> str:
            if pd.isna(value):
                return ".xxxx"
            text = four(value)
            return f"\\textbf{{{text}}}" if bold else text
        cells = [render(row.Total)] + [
            render(row[bucket]) if bucket in frame.columns else ".xxxx"
            for bucket, _label, _same in SUBSETS
        ]
        epoch = int(row.Epoch)
        label = f"\\textbf{{{epoch}}}" if bold else str(epoch)
        lines.append(f"{label:<3s} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def statistics(loss: pd.DataFrame, accuracy: pd.DataFrame | None, plateau: int) -> str:
    total = loss.Total.to_numpy()
    same = [bucket for bucket, _l, is_same in SUBSETS if is_same and bucket in loss.columns]
    cross = [bucket for bucket, _l, is_same in SUBSETS if not is_same and bucket in loss.columns]
    final = loss.iloc[-1]
    names = {bucket: PLAIN[label] for bucket, label, _s in SUBSETS}
    lines = [
        "# Section 6.2 - computed values", "",
        "## Validation BCE loss", "",
        f"- epochs trained: **{len(loss)}**",
        f"- epoch 1: **{total[0]:.4f}**",
        f"- lowest: **{total.min():.4f}** at epoch **{int(loss.Total.idxmin()) + 1}**",
        f"- final epoch: **{total[-1]:.4f}**",
        f"- total improvement: **{total[0] - total.min():.4f}**",
        f"- plateau epoch (under 1% of the gain left): **{plateau}**",
        f"- gain remaining after the plateau: **{total[plateau - 1] - total.min():.4f}**",
        "",
        f"- same-language mean, final epoch: **{final[same].mean():.4f}**",
        f"- cross-language mean, final epoch: **{final[cross].mean():.4f}**",
        f"- cross minus same: **{final[cross].mean() - final[same].mean():+.4f}**",
        "",
    ]
    if accuracy is not None and not accuracy.empty:
        values = accuracy.Total.to_numpy()
        best_epoch = int(accuracy.Total.idxmax()) + 1
        acc_same = [b for b in same if b in accuracy.columns]
        acc_cross = [b for b in cross if b in accuracy.columns]
        final_accuracy = accuracy.iloc[-1]
        ordered = final_accuracy.drop(
            labels=[c for c in ("Epoch", "Total", "Threshold") if c in final_accuracy.index]
        ).sort_values(ascending=False)
        lines += [
            "## Test accuracy (Table 12)", "",
            f"- epoch 1: **{values[0]:.4f}**",
            f"- best: **{values.max():.4f}** at epoch **{best_epoch}**",
            f"- final epoch: **{values[-1]:.4f}**",
            f"- gain from epoch 1 to best: **{values.max() - values[0]:+.4f}**",
            f"- change from best to final: **{values[-1] - values.max():+.4f}**",
            "",
            f"- same-language mean, final epoch: **{final_accuracy[acc_same].mean():.4f}**",
            f"- cross-language mean, final epoch: **{final_accuracy[acc_cross].mean():.4f}**",
            "",
            f"- easiest configuration: **{names.get(ordered.index[0], ordered.index[0])}** "
            f"({ordered.iloc[0]:.4f})",
            f"- hardest configuration: **{names.get(ordered.index[-1], ordered.index[-1])}** "
            f"({ordered.iloc[-1]:.4f})",
            "",
        ]
    deltas = loss.Total.diff().dropna()
    rising = int((deltas > 1e-6).sum())
    lines.append(
        f"- the loss curve decreases on **{len(deltas) - rising}** of **{len(deltas)}** "
        f"epoch transitions"
        + ("; no rise anywhere" if rising == 0 else f", rising on {rising}")
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path,
                        default=Path.home() / "AppData/Local/Temp/d2_latex_extract")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "kaggle" / "latex" / "d2")
    parser.add_argument("--template", type=Path, default=_ROOT / "kaggle" / "lecture.txt")
    args = parser.parse_args()

    loss, accuracy = load(args.work_dir)
    plateau = detect_plateau(loss.Total.to_numpy())
    print(f"{len(loss)} epochs, plateau at epoch {plateau}, "
          f"accuracy rows: {0 if accuracy is None else len(accuracy)}")

    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    draw(loss, plateau, figures / FIGURE_NAME)

    text = args.template.read_text(encoding="utf-8", errors="ignore")
    anchor = text.find("\\label{fig:d2_epochs}")
    if anchor != -1:
        start = text.rfind("\\begin{figure*}", 0, anchor)
        end = text.find("\\end{figure*}", anchor)
        (args.output_dir / "figure_d2.tex").write_text(
            "% Copied from the paper template; the figure file is generated by\n"
            "% research/discussions/build_d2_latex.py into figures/.\n"
            + text[start:end + len("\\end{figure*}")] + "\n", encoding="utf-8")

    if accuracy is not None and not accuracy.empty:
        (args.output_dir / "table_d2_epochs.tex").write_text(
            accuracy_table(accuracy), encoding="utf-8")
        print(f"wrote {args.output_dir/'table_d2_epochs.tex'}")
    else:
        print("no test-accuracy CSV found; Table 12 not generated")

    (args.output_dir / "section_6_2_numbers.md").write_text(
        statistics(loss, accuracy, plateau), encoding="utf-8")
    print(f"wrote {args.output_dir/'figure_d2.tex'}")
    print(f"wrote {args.output_dir/'section_6_2_numbers.md'}")
    print(f"wrote {figures/FIGURE_NAME}")


if __name__ == "__main__":
    main()
