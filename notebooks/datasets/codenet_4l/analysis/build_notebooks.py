"""Generate the CodeNet 4L analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _source(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


SETUP = """from pathlib import Path
import sys
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / 'spectral_code').exists():
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError('Open this notebook from the repository.')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from spectral_code.evaluation.codenet_analysis import *

sns.set_theme(style='whitegrid', context='notebook')
ROOT = clean_data_dir()
ROOT"""


NOTEBOOKS = {
    "00_small_graph_inspection.ipynb": [
        markdown("# CodeNet 4L — Small Graph Inspection\n\nSource-release and final graph-export sanity checks."),
        code(SETUP),
        markdown("## Source release"),
        code("release = source_release_summary()\npd.DataFrame(release['pair_buckets'])"),
        markdown("## Graph readiness"),
        code("coverage, summary = graph_coverage(ROOT)\ndisplay(summary)\ndisplay(coverage)"),
        code("codes = load_codes(ROOT, include_code=True)\ncodes[['code_id', 'source_code_id', 'language', 'problem_id', 'is_mutant', 'line_count', 'code']].sample(3, random_state=42)"),
    ],
    "01_dataset_analysis.ipynb": [
        markdown("# CodeNet 4L — Dataset Analysis\n\nSplit balance, language coverage, configuration mix, and pair provenance."),
        code(SETUP),
        code("codes = load_codes(ROOT)\npairs = load_pairs(ROOT)\nprint('codes:', len(codes), 'pairs:', len(pairs))\ndisplay(codes.groupby(['language', 'is_mutant']).size().rename('codes').to_frame())\ndisplay(pairs.groupby(['split', 'label_name']).size().rename('pairs').to_frame())"),
        code("columns = [c for c in ['configuration_id', 'pair_kind', 'split', 'label_name'] if c in pairs]\ndisplay(pairs.groupby(columns).size().rename('pairs').to_frame())"),
        code("sns.countplot(data=pairs, x='split', hue='label_name', order=['train', 'valid', 'test']); plt.show()"),
    ],
    "02_graph_and_spectral_analysis.ipynb": [
        markdown("# CodeNet 4L — Graph and Spectral Analysis\n\nCoverage, representative pairs, and sampled spectral score distributions."),
        code(SETUP),
        code("coverage, summary = graph_coverage(ROOT)\ndisplay(summary)\ndisplay(coverage)"),
        code("sns.catplot(data=coverage, x='graph_type', y='codes', hue='spectral_status', col='language', kind='bar'); plt.show()"),
        markdown("## Clone and non-clone examples"),
        code("examples = select_pair_examples(ROOT, graph_type='cpg', per_label=2, seed=42)\nfor example in examples:\n    display(pd.DataFrame([example['pair']]))\n    display(plot_pair_code_graph_spectra(ROOT, example, graph_type='cpg'))"),
        code("scores = sample_similarity_scores(ROOT, sample_size=10_000, seed=42)\nsns.displot(data=scores, x='pss', hue='label_name', col='graph_type', stat='density', common_norm=False, bins=40, col_wrap=2); plt.show()"),
    ],
    "03_tuning_analysis.ipynb": [
        markdown("# CodeNet 4L — Tuning Analysis\n\nBalanced train-sample threshold sweep for PSS."),
        code(SETUP),
        code("scores = sample_similarity_scores(ROOT, sample_size=10_000, seed=42)\nthresholds, distributions = threshold_summary(scores)\ndisplay(thresholds)\ndisplay(distributions)"),
        code("sns.barplot(data=thresholds, x='graph_type', y='f1'); plt.ylim(0, 1); plt.show()"),
    ],
    "04_analysis_reports.ipynb": [
        markdown("# CodeNet 4L — Analysis Reports\n\nWrite reusable CSV/JSON summaries under `outputs/codenet_4l/reports`."),
        code(SETUP),
        code("paths = write_analysis_reports(ROOT, sample_size=10_000, seed=42)\npaths"),
    ],
}


def main() -> None:
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    for filename, cells in NOTEBOOKS.items():
        path = HERE / filename
        path.write_text(
            json.dumps({"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()

