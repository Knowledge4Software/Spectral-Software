"""Emit the paper's two RQ2 tables with our measured numbers substituted in.

Reproduces the supervisor's templates exactly -- same column specs, rules,
section bands, adjustbox/resizebox wrappers, and macros -- so each file drops
straight into the paper source. Only the numbers change.

Cells we did not measure, or measured but cannot report (a run attached to the
wrong dataset, or a model that collapsed), stay ``.xx`` and are listed in
``provenance.md`` with the reason.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RQ2_ROOT = _ROOT.parent / "outputs" / "kaggle" / "RQ2"
DATASETS = (("codexglue", "BigCloneBench"), ("atcoder", "AtCoder"), ("codenet", "CodeNet"))
METRICS = ("P", "R", "F1", "Acc")

# Cells withheld from the tables, with the reason recorded in provenance.md.
# These three need far more than one Kaggle session at full data, so they run
# under the shared session_50k profile (50,000 training pairs on every dataset,
# CodeNet included). They stay comparable with each other but not with the
# full-data rows, so the table marks them.
CAPPED_METHODS = {"CDLH", "FA-AST+GGNN", "FA-AST+GMN"}
CAPPED_NOTE = (
    "\\added{$^{\\dagger}$Trained on a fixed 50,000-pair subsample of each "
    "training split; every other row uses the full split. The subsample is "
    "identical across these three methods and all three datasets.}"
)

# Nothing is withheld: every cell reports the run that was actually performed.
# Runs that did not converge are still reported, and flagged in provenance.md so
# a reader can tell an unconverged number from a trained one.
WITHHELD: dict[tuple[str, str], str] = {}

# Recall this high with chance-level precision means the model labelled every
# pair a clone, so the accuracy reflects the label balance rather than learning.
NON_CONVERGED_RECALL = 0.99

# Table 5: row label in the paper -> key we collect results under.
TABLE5_LAYOUT: tuple[tuple[str, str | None], ...] = (
    ("SECTION:Our method", None),
    ("\\textbf{\\method{} (Topo.)}", "SPECTRA-Siam (Topo.)"),
    ("\\textbf{\\method{} (Label)}", "SPECTRA-Siam (Label)"),
    ("\\textbf{\\method{} (Lex.)}", "SPECTRA-Siam (Lex.)"),
    ("SECTION:Tree-based code baselines", None),
    ("Deckard", "Deckard"),
    ("RtvNN", "RtvNN"),
    ("CDLH", "CDLH"),
    ("SECTION:Other graph-based learning methods", None),
    ("ASTNN", "ASTNN"),
    ("FA-AST+GGNN", "FA-AST+GGNN"),
    ("FA-AST+GMN", "FA-AST+GMN"),
    ("SECTION:Hybrid graph + code baseline", None),
    ("DeepSim", "DeepSim"),
    ("SECTION:GNN baselines", None),
    ("GNN-AST", "GNN-AST"),
    ("GNN-CFG", "GNN-CFG"),
    ("GNN-DDG", "GNN-DDG"),
    ("GNN-CPG", "GNN-CPG"),
)

TABLE5_CAPTION = (
    "Test performance of \\method{} and competing representations and clone-detection "
    "methods on BigCloneBench, AtCoder, and CodeNet."
)

# Table 6: (setting, language-pair label, dataset, language key in the breakdown CSV)
TABLE6_ROWS = (
    ("Single-language", "Java--Java", "BigCloneBench", "java"),
    ("Single-language", "Java--Java", "CodeNet", "java"),
    ("Single-language", "Python--Python", "CodeNet", "python"),
    ("Single-language", "C++--C++", "CodeNet", "cpp"),
    ("Single-language", "C\\#--C\\#", "CodeNet", "csharp"),
    ("Cross-language", "Java--Python", "AtCoder", "java->python"),
    ("Cross-language", "Java--Python", "CodeNet", "java->python"),
    ("Cross-language", "Java--C++", "CodeNet", "cpp->java"),
    ("Cross-language", "Java--C\\#", "CodeNet", "csharp->java"),
    ("Cross-language", "Python--C++", "CodeNet", "cpp->python"),
    ("Cross-language", "Python--C\\#", "CodeNet", "csharp->python"),
    ("Cross-language", "C++--C\\#", "CodeNet", "cpp->csharp"),
)
TABLE6_MODELS = (("SPECTRA-Siam", "\\method{}"), ("RtvNN", "RtvNN"),
                 ("ASTNN", "ASTNN"), ("DeepSim", "DeepSim"))
TABLE6_CAPTION = (
    "Language-wise breakdown of the results for the best performing representative of "
    "each method family in \\autoref{tab:rq2_overall}."
)


def two(value) -> str:
    """Paper style: two decimals, leading zero stripped, 1.00 kept whole."""
    text = f"{float(value):.2f}"
    return text if text.startswith("1") else text.lstrip("0")


def three(value) -> str:
    text = f"{float(value):.3f}"
    return text if text.startswith("1") else text.lstrip("0")


def extract(work: Path) -> None:
    """Unpack every RQ2 archive once so the CSVs inside can be read."""
    work.mkdir(parents=True, exist_ok=True)
    for folder, _display in DATASETS:
        for archive in sorted((RQ2_ROOT / folder).glob("*.zip")):
            target = work / folder / archive.stem
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)


def collect_overall(work: Path) -> dict[tuple[str, str], dict[str, float]]:
    results: dict[tuple[str, str], dict[str, float]] = {}
    for folder, display in DATASETS:
        for path in glob.glob(str(work / folder / "**" / "*results.csv"), recursive=True):
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    method = row.get("Method")
                    variant = row.get("MethodVariant") or ""
                    if method == "SPECTRA-Siam":
                        for token, label in (("input_only_topo", "(Topo.)"),
                                             ("input_only_label", "(Label)"),
                                             ("input_only_lex", "(Lex.)")):
                            if token in variant:
                                method = f"SPECTRA-Siam {label}"
                                break
                    if method in ("AST", "CFG", "DDG", "CPG"):
                        method = f"GNN-{method}"
                    if not method or not row.get("P"):
                        continue
                    # Several archives can describe the same cell (an older
                    # full-data attempt beside the capped rerun). Glob order is
                    # arbitrary, so prefer the run that actually trained: a
                    # model stuck at recall 1.0 predicts every pair a clone and
                    # must never displace a converged run.
                    existing = results.get((method, display))
                    if existing is not None:
                        degenerate = float(row.get("R", 0)) >= 0.99
                        kept_degenerate = existing["R"] >= 0.99
                        if degenerate and not kept_degenerate:
                            continue
                    results[(method, display)] = {
                        metric: float(row[metric]) for metric in METRICS
                    }
    return results


def collect_language(work: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Per-language rows, keyed by (model, dataset, language)."""
    wanted = {
        ("SPECTRA-Siam", "codexglue"): "CodeXGlue_spectra_siam_lexical",
        ("SPECTRA-Siam", "atcoder"): "ATCoder_spectra_siam_Lexical",
        ("SPECTRA-Siam", "codenet"): "CodeNet_spectra_siam_Lexical",
        ("RtvNN", "codexglue"): "CodeXGlue_RtvNN",
        ("RtvNN", "atcoder"): "ATCoder_RtvNN",
        ("RtvNN", "codenet"): "CodeNet_rtvnn",
        ("ASTNN", "codexglue"): "CodeXGlue_astnn",
        ("ASTNN", "atcoder"): "ATCoder_astnn",
        ("ASTNN", "codenet"): "CodeNet_astnn",
        ("DeepSim", "codexglue"): "CodeXGlue_deepsim",
        ("DeepSim", "atcoder"): "ATCoder_deepsim",
        ("DeepSim", "codenet"): "CodeNet_deepsim",
    }
    display = dict(DATASETS)
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for (model, folder), stem in wanted.items():
        base = work / folder / stem
        if not base.is_dir():
            continue
        for path in glob.glob(str(base / "**" / "*language_breakdown.csv"), recursive=True):
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    language = (row.get("Language") or "").strip()
                    if not language or not row.get("P"):
                        continue
                    out[(model, display[folder], language)] = {
                        metric: float(row[metric]) for metric in METRICS
                    }
    return out


def pretrained_block(template: Path) -> list[str]:
    """The paper's *Pretrained code models* rows, copied verbatim.

    Those runs are a colleague's work, so their numbers are carried across
    unchanged rather than recomputed here.
    """
    text = template.read_text(encoding="utf-8", errors="ignore")
    start = text.find("\\multicolumn{13}{@{}l}{\\textit{Pretrained code models}}")
    if start == -1:
        return []
    start = text.rfind("\\rowcolor{black!7}", 0, start)
    end = text.find("\\multicolumn{13}{@{}l}{\\textit{GNN baselines}}", start)
    if end == -1:
        return []
    end = text.rfind("\\rowcolor{black!7}", start, end)
    return text[start:end].rstrip("\n").split("\n")


def unixcoder_columns(template: Path) -> dict[tuple[str, str], list[str]]:
    """UniXcoder cells from the paper's language table, keyed by (pair, dataset)."""
    text = template.read_text(encoding="utf-8", errors="ignore")
    marker = text.find("\\label{tab:language}")
    if marker == -1:
        return {}
    body = text[marker:text.find("\\end{table*}", marker)]
    values: dict[tuple[str, str], list[str]] = {}
    lines = [line.strip() for line in body.split("\n")]
    for index, line in enumerate(lines):
        if not line.startswith("& ") or line.startswith("& \\"):
            continue
        pair = line[2:].strip()
        if index + 1 >= len(lines) or not lines[index + 1].startswith("& "):
            continue
        dataset = lines[index + 1][2:].strip()
        groups = []
        for offset in range(2, 8):
            if index + offset >= len(lines):
                break
            candidate = lines[index + offset]
            if not candidate.startswith("&"):
                break
            cells = [cell.strip() for cell in candidate.lstrip("&").replace("\\\\", "").split("&")]
            cells = [cell for cell in cells if cell]
            if len(cells) == 4:
                groups.append(cells)
        if len(groups) == 5:
            values[(pair, dataset)] = groups[4]
    return values


def build_table5(results: dict, missing: list[str], pretrained: list[str]) -> str:
    lines = [
        "% Generated by research/rq2/build_latex.py -- structure matches the paper template.",
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{TABLE5_CAPTION}}}",
        "\\label{tab:rq2_overall}",
        "\\scriptsize",
        "",
        "\\setlength{\\tabcolsep}{1.4pt}",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\setlength{\\aboverulesep}{0.2ex}",
        "\\setlength{\\belowrulesep}{0.2ex}",
        "",
        "\\begin{adjustbox}{",
        "    max width=\\columnwidth,",
        "    max totalheight=1.0\\textheight,",
        "    keepaspectratio,",
        "    center",
        "}",
        "\\begin{tabular}{@{}l*{12}{c}@{}}",
        "\\toprule",
        "\\textbf{Method}",
        "& \\multicolumn{4}{c}{\\shortstack{\\textbf{BigCloneBench}}}",
        "& \\multicolumn{4}{c}{\\textbf{AtCoder}}",
        "& \\multicolumn{4}{c}{\\textbf{CodeNet}} \\\\",
        "\\cmidrule(lr){2-5}",
        "\\cmidrule(lr){6-9}",
        "\\cmidrule(l){10-13}",
        "& \\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{Acc.}",
        "& \\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{Acc.}",
        "& \\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{Acc.} \\\\",
        "\\midrule",
        "",
    ]
    for label, key in TABLE5_LAYOUT:
        if label == "SECTION:GNN baselines" and pretrained:
            lines += [*pretrained, ""]
        if label.startswith("SECTION:"):
            lines += [
                "\\rowcolor{black!7}",
                f"\\multicolumn{{13}}{{@{{}}l}}{{\\textit{{{label.split(':', 1)[1]}}}}}\\\\[0pt]",
                "",
            ]
            continue
        # Rows whose runs are capped carry a dagger so a reader never compares
        # them with the full-data rows by mistake.
        if key in CAPPED_METHODS:
            label = f"{label}\\added{{$^{{\\dagger}}$}}"
        cells = []
        for _folder, display in DATASETS:
            reason = WITHHELD.get((key or "", display))
            values = results.get((key, display))
            if reason is not None:
                cells.append([".xx"] * 4)
                missing.append(f"{key} / {display}: {reason}")
            elif values is None:
                cells.append([".xx"] * 4)
                missing.append(f"{key} / {display}: not run")
            else:
                cells.append([two(values[metric]) for metric in METRICS])
                if values["R"] >= NON_CONVERGED_RECALL and values["P"] < 0.55:
                    missing.append(
                        f"{key} / {display}: reported, but the run did not converge "
                        f"(recall {values['R']:.3f}, precision {values['P']:.3f}) -- "
                        f"every pair predicted a clone"
                    )
        lines += [
            label,
            "& " + " & ".join(cells[0]),
            "& " + " & ".join(cells[1]),
            "& " + " & ".join(cells[2]) + " \\\\",
            "",
        ]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", ""]
    if any(key in CAPPED_METHODS for _label, key in TABLE5_LAYOUT):
        lines += ["\\vspace{0.4ex}", "{\\scriptsize " + CAPPED_NOTE + "}", ""]
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def build_table6(language: dict, missing: list[str], unixcoder: dict) -> str:
    # UniXcoder is a collaborator's run, carried across from the paper source.
    groups = list(TABLE6_MODELS) + [("__unixcoder__", "UniXcoder")]
    lines = [
        "% Generated by research/rq2/build_latex.py -- structure matches the paper template.",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\tiny",
        f"\\caption{{{TABLE6_CAPTION}}}",
        "\\label{tab:language}",
        "",
        "\\setlength{\\tabcolsep}{1.2pt}",
        "\\renewcommand{\\arraystretch}{0.90}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lll*{20}{c}@{}}",
        "\\toprule",
        "",
        "\\multirow{2}{*}{\\textbf{Setting}}",
        "&",
        "\\multirow{2}{*}{\\textbf{Language pair}}",
        "&",
        "\\multirow{2}{*}{\\textbf{Dataset}}",
    ]
    for _key, label in groups:
        lines += ["&", f"\\multicolumn{{4}}{{c}}{{\\textbf{{{label}}}}}"]
    lines += ["\\\\", ""]
    for index in range(len(groups)):
        start = 4 + index * 4
        rule = "(l)" if index == len(groups) - 1 else "(lr)"
        lines.append(f"\\cmidrule{rule}{{{start}-{start + 3}}}")
    lines += ["", "&", "&", "&"]
    for index in range(len(groups)):
        lines.append("\\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{Acc.}")
        lines.append("&" if index < len(groups) - 1 else "\\\\")
    lines += ["", "\\midrule", ""]

    for setting in ("Single-language", "Cross-language"):
        rows = [row for row in TABLE6_ROWS if row[0] == setting]
        if setting == "Cross-language":
            lines += ["\\midrule", ""]
        lines += [f"\\multirow{{{len(rows)}}}{{*}}{{{setting}}}", ""]
        for _setting, pair, dataset, language_key in rows:
            cells = []
            for model, _label in TABLE6_MODELS:
                values = language.get((model, dataset, language_key))
                if values is None:
                    cells.append(["xxxx"] * 4)
                    missing.append(f"Table 6: {model} / {dataset} / {pair}")
                else:
                    cells.append([three(values[metric]) for metric in METRICS])
            # UniXcoder is a colleague's run: copied verbatim from the paper.
            carried = unixcoder.get((pair, dataset))
            if carried is None:
                cells.append(["xxxx"] * 4)
                missing.append(f"Table 6: UniXcoder / {dataset} / {pair} (not in template)")
            else:
                cells.append(carried)
            body = "\n".join(f"& {' & '.join(group)}" for group in cells)
            lines += [f"& {pair}", f"& {dataset}", body + " \\\\", ""]
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path,
                        default=Path(os.environ.get("TEMP", "/tmp")) / "rq2_latex_extract")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "kaggle" / "latex" / "rq2")
    args = parser.parse_args()

    extract(args.work_dir)
    overall = collect_overall(args.work_dir)
    language = collect_language(args.work_dir)
    print(f"collected {len(overall)} overall cells, {len(language)} language rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing5: list[str] = []
    missing6: list[str] = []
    template = _ROOT / "kaggle" / "lecture.txt"
    carried_block = pretrained_block(template)
    carried_unixcoder = unixcoder_columns(template)
    print(f"carried over: {len(carried_block)} pretrained lines, "
          f"{len(carried_unixcoder)} UniXcoder rows")
    (args.output_dir / "table_rq2_overall.tex").write_text(
        build_table5(overall, missing5, carried_block), encoding="utf-8")
    (args.output_dir / "table_rq2_language.tex").write_text(
        build_table6(language, missing6, carried_unixcoder), encoding="utf-8")

    report = ["# RQ2 tables provenance", "",
              "Numbers come from `outputs/kaggle/RQ2`. Listed below are cells left as",
              "`.xx`/`xxxx` because no run exists, and cells whose numbers are reported",
              "but come from a run that did not converge.", ""]
    report += ["## Table 5 (tab:rq2_overall)", ""]
    report += [f"- {item}" for item in missing5] or ["- none"]
    report += ["", "## Table 6 (tab:language)", ""]
    report += [f"- {item}" for item in missing6] or ["- none"]
    report += ["", "## Pretrained / UniXcoder columns", "",
               "Those runs belong to a collaborator, so their cells are copied verbatim",
               "from the paper source (`kaggle/lecture.txt`) rather than recomputed."]
    (args.output_dir / "provenance.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"wrote {args.output_dir/'table_rq2_overall.tex'}  ({len(missing5)} withheld cells)")
    print(f"wrote {args.output_dir/'table_rq2_language.tex'} ({len(missing6)} withheld cells)")


if __name__ == "__main__":
    main()
