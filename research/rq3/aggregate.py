"""Aggregate RQ3 notebook results into Tables 7-10 and the Section 5.3 text.

Reads every ``rq3_*.csv`` produced by the notebooks, then writes:

    table7.tex / table8.tex / table9.tex / table10.tex
    table7.png / table8.png / table9.png / table10.png
    rq3_all_results.csv
    section_5_3_text.md

Cells with no result yet are rendered as ``.xx`` (or ``xxxx``) rather than
being silently dropped, so a partially finished sweep still produces a
readable table and it is obvious what is missing. The Pretrained column is
emitted empty: these notebooks do not train pretrained models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.rq3.matrix import SYMBOLS

MODELS = ("SPECTRA-Siam", "ASTNN", "RtvNN", "DeepSim")
DISPLAY_MODELS = (*MODELS, "Pretrained")
LANGUAGE_NAMES = {"J": "Java", "P": "Python", "C": "C++", "S": "C#"}
TABLE9_VARIANTS = ("none", "X2-X2")
TABLE10_VARIANTS = ("none", "X2-X2", "X3-X3", "X2-X2+X3-X3")


def load_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.rglob("rq3_*.csv"))
    if not files:
        raise FileNotFoundError(f"No rq3_*.csv files under {results_dir}")
    frames = [pd.read_csv(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    # A rerun notebook appends, so the same configuration can appear twice;
    # the last write is the authoritative one.
    frame = frame.drop_duplicates(subset=["Model", "Table", "Configuration"], keep="last")
    frame["AccuracyPct"] = frame.Accuracy.astype(float) * 100.0
    print(f"loaded {len(frame)} results from {len(files)} files")
    return frame.reset_index(drop=True)


def lookup(frame: pd.DataFrame, model: str, table: str, configuration: str) -> float | None:
    hit = frame[
        frame.Model.eq(model) & frame.Table.eq(table) & frame.Configuration.eq(configuration)
    ]
    return float(hit.AccuracyPct.iloc[-1]) if len(hit) else None


def _cell(value: float | None, width: int = 2) -> str:
    """Format one accuracy, or a placeholder when that run has not finished."""
    if value is None:
        return "xxxx" if width < 2 else ".xx"
    return f"{value:.{width}f}"


# --------------------------------------------------------------------------
# Table builders. Each returns a list of display rows shared by LaTeX and PNG.
# --------------------------------------------------------------------------

def table7_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for symbol in SYMBOLS:
        rows.append({
            "section": "Within-language reference",
            "left": symbol,
            "right": f"{symbol}-{symbol}",
            "values": [lookup(frame, model, "table7_within", f"{symbol}-{symbol}") for model in MODELS],
        })
    for index, left in enumerate(SYMBOLS):
        for right in SYMBOLS[index + 1:]:
            name = f"{left}-{right}"
            rows.append({
                "section": "Cross-language generalization without cross-language supervision",
                "left": f"{left},{right}",
                "right": name,
                "values": [lookup(frame, model, "table7_cross", name) for model in MODELS],
            })
    return rows


def table8_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for target in SYMBOLS:
        for source in SYMBOLS:
            if source == target:
                continue
            name = f"{source}->{target}"
            rows.append({
                "section": f"Target {LANGUAGE_NAMES[target]} ({target})",
                "path": f"{source} \\rightarrow {target}",
                "values": [lookup(frame, model, "table8", name) for model in MODELS],
            })
    return rows


def _bridge_rows(frame: pd.DataFrame, table: str, variants: tuple[str, ...]) -> list[dict]:
    names = sorted({
        name.split("|")[0]
        for name in frame[frame.Table.eq(table)].Configuration.astype(str)
    })
    if not names:  # nothing finished yet: still emit the full expected grid
        from research.rq3.matrix import table9, table10

        source = table9() if table == "table9" else table10()
        names = sorted({configuration.name.split("|")[0] for configuration in source})
    rows = []
    for name in names:
        target = name.split("->")[-1]
        rows.append({
            "section": f"Target {LANGUAGE_NAMES[target]} ({target})",
            "path": name.replace("->", " \\rightarrow "),
            "values": [
                [lookup(frame, model, table, f"{name}|{variant}") for variant in variants]
                for model in MODELS
            ],
        })
    return sorted(rows, key=lambda row: (SYMBOLS.index(row["path"].split()[-1]), row["path"]))


def table9_rows(frame: pd.DataFrame) -> list[dict]:
    return _bridge_rows(frame, "table9", TABLE9_VARIANTS)


def table10_rows(frame: pd.DataFrame) -> list[dict]:
    return _bridge_rows(frame, "table10", TABLE10_VARIANTS)


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

def render_latex(rows: list[dict], table: str, path: Path) -> None:
    columns = "l" + "l" * (1 if table == "table7" else 0) + "c" * len(DISPLAY_MODELS)
    lines = [
        "% Generated by research/rq3/aggregate.py -- do not edit by hand.",
        "\\begin{tabular}{" + columns + "}",
        "\\toprule",
    ]
    if table == "table7":
        header = "\\textbf{SL train} & \\textbf{Test} & " + " & ".join(
            f"\\textbf{{{model}}}" for model in DISPLAY_MODELS
        )
    else:
        header = "\\textbf{Training path} & " + " & ".join(
            f"\\textbf{{{model}}}" for model in DISPLAY_MODELS
        )
    lines += [header + " \\\\", "\\midrule"]

    section = None
    for row in rows:
        if row["section"] != section:
            section = row["section"]
            span = len(DISPLAY_MODELS) + (2 if table == "table7" else 1)
            lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\emph{{{section}}}}} \\\\")
        if table == "table7":
            cells = [row["left"], row["right"]]
            cells += [_cell(value) for value in row["values"]]
        elif table == "table8":
            cells = [f"${row['path']}$"] + [_cell(value) for value in row["values"]]
        else:
            cells = [f"${row['path']}$"]
            cells += ["/".join(_cell(value, 1) for value in group) for group in row["values"]]
        cells.append("")  # Pretrained column, intentionally empty
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_png(rows: list[dict], table: str, path: Path) -> None:
    label_width = 0.34 if table in {"table9", "table10"} else 0.26
    figure_width = 15.0 if table == "table10" else 11.0
    height = max(3.0, 0.30 * (len(rows) + 6))
    figure, axis = plt.subplots(figsize=(figure_width, height))
    axis.axis("off")
    left, right = 0.03, 0.985
    group_width = (right - left - label_width) / len(DISPLAY_MODELS)
    font = 8.6
    top = len(rows) + 2.0
    axis.set_xlim(0, 1)

    axis.plot([left, right], [top - 0.2] * 2, color="black", lw=1.3)
    axis.plot([left, right], [top - 1.25] * 2, color="black", lw=0.8)
    header = "SL train / Test" if table == "table7" else "Training path"
    axis.text(left, top - 0.72, header, fontsize=font + 0.6, fontweight="bold", va="center")
    for index, model in enumerate(DISPLAY_MODELS):
        centre = left + label_width + (index + 0.5) * group_width
        axis.text(centre, top - 0.72, model, fontsize=font + 0.6, fontweight="bold",
                  ha="center", va="center")

    y = len(rows) - 0.5
    section = None
    for row in rows:
        if row["section"] != section:
            section = row["section"]
            axis.add_patch(plt.Rectangle((left, y - 0.32), right - left, 0.64,
                                         facecolor="#ebebeb", edgecolor="none", zorder=0))
            axis.text(left + 0.004, y, section, fontsize=font, style="italic", va="center", zorder=2)
            y -= 1
        label = f"{row['left']}  {row['right']}" if table == "table7" else row["path"].replace(
            " \\rightarrow ", " → "
        )
        axis.text(left + 0.004, y, label, fontsize=font, va="center")
        for index, value in enumerate(row["values"]):
            centre = left + label_width + (index + 0.5) * group_width
            text = "/".join(_cell(item, 1) for item in value) if isinstance(value, list) else _cell(value)
            axis.text(centre, y, text, fontsize=font, ha="center", va="center",
                      color="#9a9a9a" if "x" in text else "black")
        # Pretrained column stays blank.
        y -= 1
    axis.plot([left, right], [y + 0.5] * 2, color="black", lw=1.3)
    axis.set_ylim(y + 0.2, top)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


# --------------------------------------------------------------------------
# Section 5.3 statistics
# --------------------------------------------------------------------------

def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _format(value: float | None, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "[MISSING]"


def section_text(frame: pd.DataFrame) -> str:
    lines = ["# Section 5.3 - computed values", ""]
    coverage = len(frame)
    expected = 166 * len(MODELS)
    lines += [
        f"Results available: **{coverage} / {expected}** "
        f"({coverage / expected * 100:.0f}% of the full sweep).",
        "",
    ]

    # Within-language reference
    lines += ["## Within-language reference (Table 7)", ""]
    within = {
        model: {
            symbol: lookup(frame, model, "table7_within", f"{symbol}-{symbol}") for symbol in SYMBOLS
        }
        for model in MODELS
    }
    for model in MODELS:
        values = within[model]
        mean = _mean(list(values.values()))
        present = {key: value for key, value in values.items() if value is not None}
        easiest = max(present, key=present.get) if present else None
        hardest = min(present, key=present.get) if present else None
        lines.append(
            f"- **{model}**: mean {_format(mean, '%')}; "
            f"easiest {LANGUAGE_NAMES.get(easiest, '[MISSING]')} "
            f"({_format(present.get(easiest), '%')}), "
            f"hardest {LANGUAGE_NAMES.get(hardest, '[MISSING]')} "
            f"({_format(present.get(hardest), '%')})"
        )
    lines.append("")

    # Cross-language without supervision
    lines += ["## Cross-language without supervision (Table 7)", ""]
    cross_names = [
        f"{left}-{right}" for index, left in enumerate(SYMBOLS) for right in SYMBOLS[index + 1:]
    ]
    for model in MODELS:
        values = {name: lookup(frame, model, "table7_cross", name) for name in cross_names}
        mean = _mean(list(values.values()))
        present = {key: value for key, value in values.items() if value is not None}
        strongest = max(present, key=present.get) if present else None
        weakest = min(present, key=present.get) if present else None
        reference = _mean(list(within[model].values()))
        drop = (
            (reference - mean) / reference * 100.0
            if reference and mean is not None and reference > 0 else None
        )
        lines.append(
            f"- **{model}**: mean {_format(mean, '%')} across the 6 pairs; "
            f"strongest {strongest or '[MISSING]'} ({_format(present.get(strongest), '%')}), "
            f"weakest {weakest or '[MISSING]'} ({_format(present.get(weakest), '%')}); "
            f"relative drop vs within-language {_format(drop, '%')}"
        )
    lines.append("")

    # Bridge lengths
    lines += ["## Bridge-assisted transfer", ""]
    for table, label in (("table8", "Length-1"), ("table9", "Length-2"), ("table10", "Length-3")):
        lines.append(f"### {label} ({table})")
        for model in MODELS:
            subset = frame[frame.Model.eq(model) & frame.Table.eq(table)]
            mean = float(subset.AccuracyPct.mean()) if len(subset) else None
            lines.append(f"- **{model}**: mean {_format(mean, '%')} over {len(subset)} runs")
        lines.append("")

    # Gap to within-language reference, per target
    lines += ["## Gap to within-language reference, per target", ""]
    for model in MODELS:
        parts = []
        for symbol in SYMBOLS:
            reference = within[model][symbol]
            bridged = frame[
                frame.Model.eq(model)
                & frame.Table.isin(["table8", "table9", "table10"])
                & frame.Target.eq(symbol)
            ]
            mean = float(bridged.AccuracyPct.mean()) if len(bridged) else None
            gap = reference - mean if reference is not None and mean is not None else None
            parts.append(f"{symbol}: {_format(gap, ' pts')}")
        lines.append(f"- **{model}**: " + ", ".join(parts))
    lines.append("")

    # Ranking across the 156 bridge configurations
    lines += ["## Ranking across bridge configurations", ""]
    bridge = frame[frame.Table.isin(["table8", "table9", "table10"])]
    pivot = bridge.pivot_table(
        index=["Table", "Configuration"], columns="Model", values="AccuracyPct"
    )
    complete = pivot.dropna(how="any")
    if len(complete):
        winners = complete.idxmax(axis=1).value_counts()
        spectra_wins = int(winners.get("SPECTRA-Siam", 0))
        lines.append(
            f"- Configurations with all {len(MODELS)} models finished: **{len(complete)}** / 156"
        )
        lines.append(
            f"- SPECTRA-Siam ranks #1 in **{spectra_wins}** of them "
            f"({spectra_wins / len(complete) * 100:.0f}%)"
        )
        if "SPECTRA-Siam" in complete:
            others = complete[[model for model in MODELS if model != "SPECTRA-Siam"]]
            margin = (complete["SPECTRA-Siam"] - others.max(axis=1)).mean()
            best_baseline = others.mean().idxmax()
            lines.append(
                f"- Mean margin over the best baseline per configuration: "
                f"**{margin:+.2f} pts** (strongest baseline overall: {best_baseline})"
            )
    else:
        lines.append("- [MISSING] no configuration has every model finished yet")
    lines.append("")

    # Reinforcement effects
    lines += ["## Effect of intermediate reinforcement", ""]
    for table, variants in (("table9", TABLE9_VARIANTS), ("table10", TABLE10_VARIANTS)):
        subset = frame[frame.Table.eq(table)].copy()
        if subset.empty:
            lines.append(f"- {table}: [MISSING]")
            continue
        subset["base"] = subset.Configuration.astype(str).str.split("|").str[0]
        for variant in variants[1:]:
            deltas = []
            for model in MODELS:
                model_subset = subset[subset.Model.eq(model)]
                none = model_subset[model_subset.Reinforcement.eq("none")].set_index("base").AccuracyPct
                other = model_subset[model_subset.Reinforcement.eq(variant)].set_index("base").AccuracyPct
                shared = none.index.intersection(other.index)
                if len(shared):
                    deltas.append((model, float((other[shared] - none[shared]).mean()), len(shared)))
            if deltas:
                rendered = ", ".join(f"{model} {delta:+.2f}" for model, delta, _ in deltas)
                lines.append(f"- {table} `{variant}` vs none: {rendered} (pts, paired)")
            else:
                lines.append(f"- {table} `{variant}` vs none: [MISSING]")
    lines.append("")

    # Path ordering and final-hop effects
    lines += ["## Path ordering", ""]
    for table in ("table9", "table10"):
        subset = frame[frame.Table.eq(table) & frame.Reinforcement.eq("none")]
        if subset.empty:
            lines.append(f"- {table}: [MISSING]")
            continue
        spread = subset.groupby(["Model", "Target"]).AccuracyPct.agg(["min", "max"])
        spread["range"] = spread["max"] - spread["min"]
        lines.append(
            f"- {table}: mean spread between orderings reaching the same target: "
            f"**{spread['range'].mean():.2f} pts** (max {spread['range'].max():.2f})"
        )
        final_hop = subset.copy()
        final_hop["last_source"] = final_hop.Path.astype(str).str.split("->").str[-2]
        best = final_hop.groupby("last_source").AccuracyPct.mean().sort_values(ascending=False)
        if len(best):
            lines.append(
                "  - final bridge language before the target, mean accuracy: "
                + ", ".join(f"{language} {value:.2f}%" for language, value in best.items())
            )
    lines.append("")
    lines += [
        "## Pretrained column",
        "",
        "Not produced by these notebooks; the tables leave those cells empty.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = load_results(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "rq3_all_results.csv", index=False)

    builders = {
        "table7": table7_rows,
        "table8": table8_rows,
        "table9": table9_rows,
        "table10": table10_rows,
    }
    for table, builder in builders.items():
        rows = builder(frame)
        render_latex(rows, table, args.output_dir / f"{table}.tex")
        render_png(rows, table, args.output_dir / f"{table}.png")
        print(f"wrote {table}.tex and {table}.png ({len(rows)} rows)")

    (args.output_dir / "section_5_3_text.md").write_text(section_text(frame), encoding="utf-8")
    print(f"wrote section_5_3_text.md")
    print(f"\nAll RQ3 outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
