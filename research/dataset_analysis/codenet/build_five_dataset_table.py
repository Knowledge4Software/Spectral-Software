"""Render the compact comparison tables from their canonical CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "five_dataset_metrics.csv"
OUTPUT_PNG = ROOT / "five_dataset_metrics.png"
LANGUAGE_INPUT_CSV = ROOT / "codenet_language_hardness.csv"
LANGUAGE_OUTPUT_PNG = ROOT / "codenet_language_hardness.png"


def _number(value: str) -> str:
    return f"{int(value):,}"


def _decimal(value: str, places: int = 4) -> str:
    return f"{float(value):.{places}f}"


def _percent(value: str) -> str:
    return "N/A" if not value else f"{float(value):.2f}%"


def _degree(row: dict[str, str]) -> str:
    return " / ".join(
        [
            _number(row["degree_min"]),
            f'{float(row["degree_mean"]):.2f}',
            _number(row["degree_median"]),
            _number(row["degree_p95"]),
            _number(row["degree_max"]),
        ]
    )


def _official_overlap(row: dict[str, str]) -> str:
    code = _percent(row["official_code_overlap_pct"])
    hashes = row["official_hash_overlap_count"]
    return f"{code} / {'N/A' if not hashes else _number(hashes) + ' hashes'}"


def load_display_rows() -> list[list[str]]:
    with INPUT_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    return [
        [
            row["dataset"],
            f'{_number(row["positive_pairs"])} / {_number(row["negative_pairs"])}',
            _decimal(row["h_pos"]),
            _decimal(row["h_neg"]),
            _decimal(row["syntax_margin"]),
            _degree(row),
            _percent(row["random_codes_in_multiple_splits_pct"]),
            _percent(row["random_test_both_endpoints_seen_in_train_pct"]),
            _official_overlap(row),
        ]
        for row in rows
    ]


def render_five_dataset_table() -> None:
    headers = [
        "Dataset",
        "Positive / Negative\npairs",
        "H_pos",
        "H_neg",
        "Syntax\nmargin",
        "Degree: Min / Mean /\nMedian / P95 / Max",
        "Codes in >1\nrandom split",
        "Random test: both endpoints\nseen in train",
        "Official overlap:\nCode / Hash",
    ]
    rows = load_display_rows()

    fig, ax = plt.subplots(figsize=(24, 7.4), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#f6f8fb")
    ax.axis("off")

    fig.text(
        0.035,
        0.925,
        "Five-Dataset Hardness & Pair-Graph Leakage Comparison",
        fontsize=23,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.035,
        0.875,
        "Canonical structural token unigram+bigram set Jaccard (full datasets; no sampling)",
        fontsize=12.5,
        color="#536079",
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.20, 0.96, 0.61],
        colWidths=[0.145, 0.155, 0.07, 0.07, 0.075, 0.19, 0.105, 0.14, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.2)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dde8")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#243b64")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(10.7)
            cell.set_height(0.14)
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else "#edf2f8")
            cell.get_text().set_color("#1f2937")
            cell.set_height(0.105)
            if column_index == 0:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("left")

    fig.text(
        0.035,
        0.125,
        "P95: 95% of active code nodes have degree at or below this value; only the highest-degree 5% exceed it.",
        fontsize=11.2,
        color="#344258",
    )
    fig.text(
        0.035,
        0.078,
        "Random-split leakage columns use a deterministic pair-level 80/10/10 simulation. Official overlap uses each dataset's provided splits.",
        fontsize=11.2,
        color="#344258",
    )
    fig.text(
        0.035,
        0.035,
        "Syntax margin = MeanPosSim - H_neg = (1 - H_pos) - H_neg. Higher is better separation; CodeXGLUE has 154 exact-source hashes across official splits.",
        fontsize=11.2,
        color="#344258",
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _read_language_rows() -> list[dict[str, str]]:
    with LANGUAGE_INPUT_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _weighted_summary(rows: list[dict[str, str]], scope: str) -> list[str]:
    selected = rows if scope == "All configurations" else [row for row in rows if row["scope"] == scope]
    positive_pairs = sum(int(row["positive_pairs"]) for row in selected)
    negative_pairs = sum(int(row["negative_pairs"]) for row in selected)
    mean_positive = sum(
        int(row["positive_pairs"]) * float(row["mean_positive_similarity"])
        for row in selected
    ) / positive_pairs
    h_neg = sum(
        int(row["negative_pairs"]) * float(row["h_neg"])
        for row in selected
    ) / negative_pairs
    return [
        scope,
        f"{positive_pairs:,} / {negative_pairs:,}",
        f"{1.0 - mean_positive:.4f}",
        f"{h_neg:.4f}",
        f"{mean_positive - h_neg:.4f}",
    ]


def _weighted_subtype_summary(rows: list[dict[str, str]], scope: str) -> list[str]:
    selected = rows if scope == "All configurations" else [row for row in rows if row["scope"] == scope]
    positive_pairs = sum(int(row["positive_pairs"]) for row in selected)
    mean_positive = sum(
        int(row["positive_pairs"]) * float(row["mean_positive_similarity"])
        for row in selected
    ) / positive_pairs
    hard_pairs = sum(int(row["hard_nonclone_pairs"]) for row in selected)
    hard_h_neg = sum(
        int(row["hard_nonclone_pairs"]) * float(row["hard_nonclone_h_neg"])
        for row in selected
    ) / hard_pairs
    diff_pairs = sum(int(row["diff_problem_pairs"]) for row in selected)
    diff_h_neg = sum(
        int(row["diff_problem_pairs"]) * float(row["diff_problem_h_neg"])
        for row in selected
    ) / diff_pairs
    hard_margin = mean_positive - hard_h_neg
    diff_margin = mean_positive - diff_h_neg
    return [
        f"Summary: {scope}",
        f"{hard_pairs:,}",
        f"{hard_h_neg:.4f}",
        f"{hard_margin:.4f}",
        f"{diff_pairs:,}",
        f"{diff_h_neg:.4f}",
        f"{diff_margin:.4f}",
        f"{diff_margin - hard_margin:.4f}",
    ]


def _style_table(
    table,
    *,
    header_color: str,
    alternate_color: str = "#edf2f8",
    font_size: float = 10.8,
    left_columns: tuple[int, ...] = (0,),
) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dde8")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else alternate_color)
            cell.get_text().set_color("#1f2937")
            if column_index in left_columns:
                cell.get_text().set_ha("left")
                cell.get_text().set_fontweight("bold")


def render_codenet_language_table() -> None:
    rows = _read_language_rows()
    main_rows = [
        [
            row["scope"],
            row["configuration"],
            f'{_number(row["positive_pairs"])} / {_number(row["negative_pairs"])}',
            _decimal(row["mean_positive_similarity"]),
            _decimal(row["h_pos"]),
            _decimal(row["h_neg"]),
            _decimal(row["syntax_margin"]),
        ]
        for row in rows
    ]
    negative_rows = [
        [
            row["configuration"],
            _number(row["hard_nonclone_pairs"]),
            _decimal(row["hard_nonclone_h_neg"]),
            _decimal(float(row["mean_positive_similarity"]) - float(row["hard_nonclone_h_neg"])),
            _number(row["diff_problem_pairs"]),
            _decimal(row["diff_problem_h_neg"]),
            _decimal(float(row["mean_positive_similarity"]) - float(row["diff_problem_h_neg"])),
            _decimal(float(row["hard_nonclone_h_neg"]) - float(row["diff_problem_h_neg"])),
        ]
        for row in rows
    ]
    negative_rows.extend(
        [
            _weighted_subtype_summary(rows, "Single-language"),
            _weighted_subtype_summary(rows, "Cross-language"),
            _weighted_subtype_summary(rows, "All configurations"),
        ]
    )
    summary_rows = [
        _weighted_summary(rows, "Single-language"),
        _weighted_summary(rows, "Cross-language"),
        _weighted_summary(rows, "All configurations"),
    ]

    fig, ax = plt.subplots(figsize=(28, 18), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#f6f8fb")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    fig.text(
        0.035, 0.955,
        "CodeNet 4L: Syntactic Hardness by Language Configuration",
        fontsize=24, fontweight="bold", color="#172033",
    )
    fig.text(
        0.035, 0.925,
        "Canonical structural token unigram+bigram set Jaccard | all 2,861,199 pairs | no sampling",
        fontsize=12.8, color="#536079",
    )

    fig.text(0.035, 0.885, "Hardness and syntax separation", fontsize=15, fontweight="bold", color="#243b64")
    main_table = ax.table(
        cellText=main_rows,
        colLabels=["Scope", "Language configuration", "Positive / Negative\npairs", "MeanPosSim", "H_pos", "H_neg", "Syntax margin"],
        cellLoc="center", colLoc="center",
        bbox=[0.025, 0.535, 0.95, 0.32],
        colWidths=[0.13, 0.16, 0.18, 0.12, 0.10, 0.10, 0.12],
    )
    _style_table(main_table, header_color="#243b64", left_columns=(0, 1))

    fig.text(0.035, 0.485, "Negative-pair subtype hardness and separate syntax margins", fontsize=15, fontweight="bold", color="#2f5f55")
    negative_table = ax.table(
        cellText=negative_rows,
        colLabels=[
            "Language configuration",
            "Hard NC\npairs",
            "Hard NC\nH_neg",
            "Hard NC\nmargin",
            "Different-problem\npairs",
            "Different-problem\nH_neg",
            "Different-problem\nmargin",
            "Margin difference\n(Diff - Hard)",
        ],
        cellLoc="center", colLoc="center",
        bbox=[0.025, 0.105, 0.95, 0.35],
        colWidths=[0.18, 0.10, 0.09, 0.10, 0.13, 0.11, 0.13, 0.13],
    )
    _style_table(
        negative_table,
        header_color="#2f5f55",
        alternate_color="#edf6f2",
        font_size=9.8,
        left_columns=(0,),
    )

    summary_table = ax.table(
        cellText=summary_rows,
        colLabels=["Scope", "Positive / Negative\npairs", "H_pos", "H_neg", "Syntax margin"],
        cellLoc="center", colLoc="center",
        bbox=[0.615, 0.462, 0.36, 0.068],
        colWidths=[0.23, 0.27, 0.13, 0.13, 0.21],
    )
    _style_table(summary_table, header_color="#7b5b24", alternate_color="#f7f1e5", font_size=9.8, left_columns=(0,))

    fig.text(
        0.035, 0.060,
        "Syntax margin = MeanPosSim - subtype H_neg. Margin difference = Different-problem margin - Hard-NC margin; larger values mean the two negative types differ more in syntactic difficulty.",
        fontsize=11.2, color="#344258",
    )
    fig.text(
        0.035, 0.030,
        "Hard NC = same problem, one Accepted and one Wrong Answer. Different-problem NC = two different CodeNet problems.",
        fontsize=11.2, color="#344258",
    )

    fig.savefig(LANGUAGE_OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    render_five_dataset_table()
    render_codenet_language_table()
