"""Analysis helpers for the portable Project CodeNet 4L clean-data export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from spectral_code.evaluation import atcoder_analysis as common
from spectral_code.evaluation.codenet_preparation import default_archive_path
from spectral_code.utils.dataset_paths import output_root_for


GRAPH_TYPES = common.GRAPH_TYPES


def clean_data_dir(root: str | Path | None = None) -> Path:
    return Path(root or output_root_for("codenet_4l") / "clean_data").resolve()


def load_codes(root: str | Path | None = None, *, include_code: bool = False) -> pd.DataFrame:
    return common.load_codes(clean_data_dir(root), include_code=include_code)


def load_pairs(root: str | Path | None = None) -> pd.DataFrame:
    return common.load_pairs(clean_data_dir(root))


def graph_coverage(root: str | Path | None = None):
    return common.graph_coverage(clean_data_dir(root))


def select_pair_examples(root: str | Path | None = None, **kwargs):
    return common.select_pair_examples(clean_data_dir(root), **kwargs)


def plot_pair_code_graph_spectra(root: str | Path | None, example: dict, **kwargs):
    return common.plot_pair_code_graph_spectra(clean_data_dir(root), example, **kwargs)


def sample_similarity_scores(root: str | Path | None = None, **kwargs) -> pd.DataFrame:
    return common.sample_similarity_scores(clean_data_dir(root), **kwargs)


threshold_summary = common.threshold_summary


def source_release_summary(archive_path: str | Path | None = None) -> dict:
    with zipfile.ZipFile(Path(archive_path or default_archive_path())) as archive:
        return json.loads(archive.read("summary.json"))


def write_analysis_reports(root: str | Path | None = None, *, sample_size: int = 10_000, seed: int = 42) -> dict[str, Path]:
    clean_root = clean_data_dir(root)
    reports = clean_root.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    codes, pairs = load_codes(clean_root), load_pairs(clean_root)
    coverage, coverage_summary = graph_coverage(clean_root)
    scores = sample_similarity_scores(clean_root, sample_size=sample_size, seed=seed)
    thresholds, distributions = threshold_summary(scores)
    paths = {
        "code_counts": reports / "codenet_4l_code_counts.csv",
        "pair_counts": reports / "codenet_4l_pair_counts.csv",
        "coverage": reports / "codenet_4l_graph_coverage.csv",
        "coverage_summary": reports / "codenet_4l_graph_coverage_summary.json",
        "scores": reports / "codenet_4l_similarity_sample.csv",
        "thresholds": reports / "codenet_4l_threshold_sample.csv",
        "distributions": reports / "codenet_4l_pss_distribution_sample.csv",
    }
    codes.groupby("language").size().reset_index(name="codes").to_csv(paths["code_counts"], index=False)
    group_columns = [column for column in ("split", "configuration_id", "pair_kind", "label_name") if column in pairs]
    pairs.groupby(group_columns).size().reset_index(name="pairs").to_csv(paths["pair_counts"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    paths["coverage_summary"].write_text(json.dumps(coverage_summary, indent=2), encoding="utf-8")
    scores.to_csv(paths["scores"], index=False)
    thresholds.to_csv(paths["thresholds"], index=False)
    distributions.to_csv(paths["distributions"], index=False)
    return paths

