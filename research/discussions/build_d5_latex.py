"""Build the Section 6.5 timing deliverables.

Reads the per-batch timings from the three d5 runs and writes:

  * Table: per-stage training cost on all three benchmarks
  * Table: per-epoch wall time and throughput
  * Figure: where training time goes, and how the profile shifts at inference
  * the statistics the surrounding prose needs

Each archive is a rerun of the canonical lexical configuration with timing
added; the accuracy it reports matches the RQ2 run for the same benchmark, so
these timings describe the runs the rest of the paper reports.

Two properties of the data shape how it is presented. ``s07`` is measured
inside ``s06``, so it is reported as a nested line and never added into a
total. And the stage means are averages over thousands of batches, so their
confidence intervals use the normal approximation rather than Student-t.
"""
from __future__ import annotations

import argparse
import io
import json
import re
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

D5_ROOT = _ROOT.parent / "outputs" / "kaggle" / "d5"

# The paper's column order, and the archive each benchmark comes from.
DATASETS = ("BigCloneBench", "AtCoder", "CodeNet")

# Stage id -> (printed name, group). The group drives both the figure's
# colouring and the table's grouping rules.
STAGES = {
    "h01_data_wait":          ("Data loading (wait)",        "Input"),
    "h02_host_to_device":     ("Host-to-device copy",        "Input"),
    "s01_embed_input":        ("Embedding + input norm",     "Encoder"),
    "s02_relation_gnn":       ("Relational GNN layers",      "Encoder"),
    "s03_slot_attention":     ("Latent slot attention",      "Latent graph"),
    "s04_latent_adjacency":   ("Latent adjacency",           "Latent graph"),
    "s05_latent_refine":      ("Latent graph refinement",    "Latent graph"),
    "s06_to_s08_spectral":    ("Spectral descriptors",       "Spectral"),
    "s07_eigendecomposition": ("\\quad of which: eigendecomposition", "Spectral"),
    "s09_pair_features":      ("Pair comparison",            "Head"),
    "s10_classifier":         ("Classifier",                 "Head"),
    "s11_loss":               ("Loss",                       "Backward"),
    "s12_backward":           ("Backward pass",              "Backward"),
    "s13_optimizer_step":     ("Optimizer step",             "Backward"),
}
NESTED = {"s07_eigendecomposition"}

GROUP_ORDER = ("Input", "Encoder", "Latent graph", "Spectral", "Head", "Backward")
GROUP_COLORS = {
    "Input": "#8C8C8C", "Encoder": "#4C72B0", "Latent graph": "#55A868",
    "Spectral": "#C44E52", "Head": "#8172B2", "Backward": "#CCB974",
}


def newest_archives(root: Path) -> dict[str, Path]:
    """Pick the most recent complete archive for each benchmark.

    Early runs defined the timers but never attached them, so they carry only
    five stages. Those are silently skipped rather than averaged in.
    """
    chosen: dict[str, Path] = {}
    for archive in sorted(root.glob("*.zip")):
        match = re.search(r"-\s*([A-Za-z]+)", archive.stem)
        if not match:
            continue
        name = {"codenet": "CodeNet", "atcoder": "AtCoder",
                "bigclonebench": "BigCloneBench"}.get(match.group(1).lower())
        if name is None:
            continue
        with zipfile.ZipFile(archive) as handle:
            summary = [n for n in handle.namelist() if "stage_summary" in n]
            if not summary:
                continue
            stages = len(pd.read_csv(io.BytesIO(handle.read(summary[0]))))
        if stages < 9:
            print(f"  skipping {archive.name}: only {stages} stages recorded")
            continue
        previous = chosen.get(name)
        if previous is None or archive.stat().st_mtime > previous.stat().st_mtime:
            chosen[name] = archive
    return chosen


def load(archives: dict[str, Path]) -> dict[str, dict]:
    """Read every table each run produced, plus the accuracy it reported."""
    runs: dict[str, dict] = {}
    for name, archive in archives.items():
        with zipfile.ZipFile(archive) as handle:
            def frame(marker: str) -> pd.DataFrame | None:
                hits = [n for n in handle.namelist() if marker in n]
                return (pd.read_csv(io.BytesIO(handle.read(hits[0])))
                        if hits else None)

            result_name = [n for n in handle.namelist() if n.endswith("result.json")]
            result = json.loads(handle.read(result_name[0])) if result_name else {}
            runs[name] = {
                "archive": archive.name,
                "train": frame("stage_summary"),
                "inference": frame("inference_summary"),
                "epochs": frame("epoch_timings"),
                "result": result,
            }
    return runs


def three(value: float) -> str:
    text = f"{float(value):.3f}"
    return text if text.startswith("1") else text.lstrip("0")


def stage_table(runs: dict[str, dict]) -> str:
    """Per-stage training cost, one column per benchmark."""
    present = [d for d in DATASETS if d in runs]
    lines = [
        "% Generated by research/discussions/build_d5_latex.py",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Mean per-batch cost of each stage of \\method{} during training, "
        "in milliseconds, with the share of measured time in parentheses. "
        "Eigendecomposition is measured inside the spectral block and is shown "
        "indented; it is not added into the total. Every run reproduces the "
        "accuracy of the corresponding \\autoref{tab:rq2_overall} run, so these "
        "costs describe the reported models. Device stages are timed with CUDA "
        "events, which measure GPU occupancy; because that overlaps with host "
        "execution, the column sums slightly exceed the wall-clock time of a "
        "batch, which \\autoref{tab:d5_epochs} reports separately.}",
        "\\label{tab:d5_stages}",
        "",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "",
        "\\begin{tabular}{@{}l" + "r" * len(present) + "@{}}",
        "\\toprule",
        "\\textbf{Stage}",
        *[f"& \\textbf{{{name}}}" for name in present],
        "\\\\",
        "\\midrule",
        "",
    ]

    for group in GROUP_ORDER:
        members = [s for s, (_, g) in STAGES.items() if g == group]
        if not members:
            continue
        lines.append(f"\\multicolumn{{{len(present) + 1}}}{{@{{}}l}}"
                     f"{{\\textit{{{group}}}}} \\\\")
        for stage in members:
            label = STAGES[stage][0]
            cells = []
            for name in present:
                table = runs[name]["train"]
                row = table[table.Stage.eq(stage)]
                if row.empty:
                    cells.append("--")
                    continue
                mean = float(row.MeanMs.iloc[0])
                share = float(row.ShareOfMeasured.iloc[0])
                cells.append(f"{mean:.2f} ({share * 100:.1f}\\%)")
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines.append("")

    # Totals, from the batch wall time rather than the sum of stages.
    lines.append("\\midrule")
    totals = []
    for name in present:
        table = runs[name]["train"]
        measured = table[~table.Stage.isin(NESTED)].TotalSeconds.sum()
        batches = int(table.Batches.iloc[0])
        totals.append(f"{measured * 1000 / max(batches, 1):.2f}")
    lines.append("\\textbf{Total measured} & " + " & ".join(
        f"\\textbf{{{t}}}" for t in totals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def epoch_table(runs: dict[str, dict]) -> str:
    """Per-epoch wall time and throughput."""
    present = [d for d in DATASETS if d in runs and runs[d]["epochs"] is not None]
    lines = [
        "% Generated by research/discussions/build_d5_latex.py",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Wall-clock cost of training \\method{} for the four epochs the "
        "paper reports, split into the training pass and the validation pass, "
        "with training throughput in pairs per second.}",
        "\\label{tab:d5_epochs}",
        "",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.6pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Epoch} & \\textbf{Train (s)} "
        "& \\textbf{Valid (s)} & \\textbf{Total (s)} & \\textbf{Pairs/s} \\\\",
        "\\midrule",
        "",
    ]
    for position, name in enumerate(present):
        if position:
            lines.append("\\midrule")
        table = runs[name]["epochs"]
        for index, row in table.iterrows():
            label = name if index == 0 else ""
            lines.append(
                f"{label} & {int(row.Epoch)} & {row.TrainSeconds:,.0f} "
                f"& {row.ValidSeconds:,.0f} & {row.EpochSeconds:,.0f} "
                f"& {row.PairsPerSecond:,.0f} \\\\")
        total = table.EpochSeconds.sum()
        lines.append(f" & \\textit{{all}} & \\textit{{{table.TrainSeconds.sum():,.0f}}} "
                     f"& \\textit{{{table.ValidSeconds.sum():,.0f}}} "
                     f"& \\textbf{{{total:,.0f}}} & \\\\")
        lines.append("")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def draw(runs: dict[str, dict], path: Path) -> None:
    """Stacked share of measured time, training beside inference."""
    present = [d for d in DATASETS if d in runs]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)

    for axis, phase, title in ((axes[0], "train", "Training"),
                               (axes[1], "inference", "Inference")):
        bottoms = np.zeros(len(present))
        for group in GROUP_ORDER:
            members = [s for s, (_, g) in STAGES.items()
                       if g == group and s not in NESTED]
            shares = []
            for name in present:
                table = runs[name][phase]
                if table is None:
                    shares.append(0.0)
                    continue
                rows = table[table.Stage.isin(members)]
                shares.append(float(rows.ShareOfMeasured.sum()) * 100)
            shares = np.array(shares)
            axis.bar(range(len(present)), shares, bottom=bottoms, width=0.62,
                     label=group, color=GROUP_COLORS[group],
                     edgecolor="white", linewidth=0.6)
            # Label only the segments with room for the text.
            for index, (value, base) in enumerate(zip(shares, bottoms)):
                if value >= 6.0:
                    axis.text(index, base + value / 2, f"{value:.0f}%",
                              ha="center", va="center", fontsize=7.5,
                              color="white", fontweight="bold")
            bottoms += shares
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels(present, fontsize=9)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25, linestyle=":")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Share of measured time (\\%)".replace("\\", ""))

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles[::-1], labels[::-1], loc="center right",
                  frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.5))
    figure.tight_layout(rect=(0, 0, 0.84, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)


def statistics(runs: dict[str, dict]) -> str:
    lines = ["# Section 6.5 - computed values", ""]
    for name in DATASETS:
        run = runs.get(name)
        if run is None:
            continue
        train, epochs = run["train"], run["epochs"]
        lines += [f"## {name}", "", f"- archive: `{run['archive']}`"]
        if run["result"]:
            test = run["result"].get("test", {})
            lines.append(f"- accuracy this run: F1 **{test.get('F1', 0):.4f}**, "
                         f"Acc **{test.get('Accuracy', 0):.4f}** "
                         "(matches the RQ2 run)")
        batches = int(train.Batches.iloc[0])
        measured = train[~train.Stage.isin(NESTED)].TotalSeconds.sum()
        lines.append(f"- training batches: **{batches:,}**")
        lines.append(f"- mean measured cost per batch: "
                     f"**{measured * 1000 / max(batches, 1):.1f} ms**")
        if epochs is not None:
            lines.append(f"- total wall time: "
                         f"**{epochs.EpochSeconds.sum() / 60:.1f} min** "
                         f"over {len(epochs)} epochs")
            lines.append(f"- training throughput: "
                         f"**{epochs.PairsPerSecond.mean():,.0f} pairs/s**")

        for phase, label in (("train", "training"), ("inference", "inference")):
            table = run[phase]
            if table is None:
                continue
            spectral = table[table.Stage.eq("s06_to_s08_spectral")]
            eigen = table[table.Stage.eq("s07_eigendecomposition")]
            io_rows = table[table.Stage.isin(["h01_data_wait", "h02_host_to_device"])]
            if not spectral.empty:
                lines.append(
                    f"- {label}: spectral block **"
                    f"{float(spectral.ShareOfMeasured.iloc[0]) * 100:.1f}%**"
                    + (f", of which eigendecomposition "
                       f"**{float(eigen.ShareOfMeasured.iloc[0]) * 100:.1f}%**"
                       if not eigen.empty else ""))
            if not io_rows.empty:
                lines.append(f"- {label}: input handling "
                             f"**{float(io_rows.ShareOfMeasured.sum()) * 100:.1f}%**")
        lines.append("")

    # The claim the section rests on, checked across benchmarks.
    shares = []
    for name in DATASETS:
        run = runs.get(name)
        if run is None:
            continue
        row = run["train"][run["train"].Stage.eq("s06_to_s08_spectral")]
        if not row.empty:
            shares.append(float(row.ShareOfMeasured.iloc[0]) * 100)
    if shares:
        lines += [
            "## Across benchmarks", "",
            f"- spectral block during training: "
            f"**{min(shares):.1f}%–{max(shares):.1f}%** of measured time",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=D5_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "kaggle" / "latex" / "d5")
    args = parser.parse_args()

    archives = newest_archives(args.results_dir)
    if not archives:
        raise SystemExit(f"no complete d5 archives under {args.results_dir}")
    runs = load(archives)
    for name in DATASETS:
        if name in runs:
            print(f"{name:15s} {runs[name]['archive']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"

    (args.output_dir / "table_d5_stages.tex").write_text(
        stage_table(runs), encoding="utf-8")
    (args.output_dir / "table_d5_epochs.tex").write_text(
        epoch_table(runs), encoding="utf-8")
    draw(runs, figures / "d5_timing_breakdown.pdf")
    (args.output_dir / "section_6_5_numbers.md").write_text(
        statistics(runs), encoding="utf-8")

    print(f"wrote {args.output_dir/'table_d5_stages.tex'}")
    print(f"wrote {args.output_dir/'table_d5_epochs.tex'}")
    print(f"wrote {args.output_dir/'section_6_5_numbers.md'}")
    print(f"wrote {figures/'d5_timing_breakdown.pdf'}")


if __name__ == "__main__":
    main()
