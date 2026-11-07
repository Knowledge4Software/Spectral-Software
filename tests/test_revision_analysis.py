from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from spectral_code.revision.clone_types import report_clone_types
from spectral_code.revision.datasets import _component_assignments, prepare_bundle
from spectral_code.revision.statistics import paired_comparison


def test_duplicate_source_components_cannot_cross_provided_problem_partitions():
    codes = [{"code_id": str(i), "problem_id": str(i), "code": text, "source_code_id": f"s{i}"}
             for i, text in enumerate(("same source", "same source", "unique"))]
    splits = _component_assignments(codes, {"0": "train", "1": "test", "2": "valid"})
    assert splits["0"] == splits["1"] == "excluded_conflicting_component"
    assert splits["2"] == "valid"


def test_reference_preparation_removes_reverse_duplicates_and_keeps_source_disjoint(tmp_path):
    source, output = tmp_path / "source", tmp_path / "ready"
    source.mkdir()
    codes = [{"code_id": str(i), "source_code_id": f"s{i}", "code": f"code {i}", "language": "java"} for i in range(400)]
    with gzip.open(source / "codes.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in codes:
            stream.write(json.dumps(row) + "\n")
    adjacency = {"node_ids": ["0", "1"], "node_types": ["Root", "Return"], "row": [0], "col": [1]}
    with gzip.open(source / "graph_spectra.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in codes:
            stream.write(json.dumps({"code_id": row["code_id"], "graphs": {"ast": {"adjacency": adjacency}}}) + "\n")
    rows = [{"left_id": str(i), "right_id": str((i + offset) % 400), "label": offset % 2, "split": split}
            for split in ("train", "valid", "test") for i in range(400) for offset in (1, 2, 3, 4)]
    rows += [{**row, "left_id": row["right_id"], "right_id": row["left_id"]} for row in rows]
    pd.DataFrame(rows).to_csv(source / "pairs.csv.gz", index=False, compression="gzip")
    (source / "metadata.json").write_text(json.dumps({"format": "spectral_clean_data_v1"}))
    manifest = prepare_bundle(source, output, "bigclonebench", required_layers=("ast",), caps=(100, 40, 40))
    assert manifest["duplicate_unordered_pairs_removed"] > 0
    assert not manifest["problem_disjoint"]
    assert all(value == 0 for counts in manifest["overlap"].values() for value in counts.values())
    frame = pd.read_csv(output / "pairs.csv.gz")
    assert frame.groupby("split").label.nunique().eq(2).all()
    assert "problem" not in manifest["overlap"]["train_test"]


def _write_runs(root, conditions=("path1",)):
    frame = pd.DataFrame({"pair_id": [str(i) for i in range(30)], "left_id": [str(i) for i in range(30)],
                          "right_id": [str(i + 1) for i in range(30)],
                          "left_problem_id": [str(i % 10) for i in range(30)],
                          "right_problem_id": [str((i + 1) % 10) for i in range(30)],
                          "label": [i % 2 for i in range(30)]})
    for condition in conditions:
        for seed in (42, 123, 231):
            for method in ("full", "baseline/astnn"):
                path = root / condition / str(seed) / method.replace("/", "_")
                path.mkdir(parents=True)
                result = {"arm": method, "seed": seed, "test": {"Pairs": len(frame), "Threshold": .5},
                          "implementation_hash": "version", "data_hash": "data", "data_group_hash": "data"}
                (path / "final_test.json").write_text(json.dumps(result))
                (path / "transfer_metadata.json").write_text(json.dumps({"key": condition}))
                predictions = frame.copy()
                predictions["probability"] = np.where(predictions.label, .8, .2)
                if method != "full":
                    predictions.loc[[1, 4, 10, 15], "probability"] = 1 - predictions.loc[[1, 4, 10, 15], "probability"]
                predictions.to_csv(path / "test_predictions.csv.gz", index=False)


def test_repeating_the_same_test_path_does_not_create_artificial_independence(tmp_path):
    _write_runs(tmp_path / "one", ("path1",))
    _write_runs(tmp_path / "two", ("path1", "path2"))
    one = paired_comparison(tmp_path / "one", "full", "baseline/astnn", tmp_path / "one_result", repetitions=50)
    two = paired_comparison(tmp_path / "two", "full", "baseline/astnn", tmp_path / "two_result", repetitions=50)
    assert one["difference_a_minus_b"] == pytest.approx(4 / 30)
    np.testing.assert_allclose(one["ci95_percentile"], two["ci95_percentile"])
    assert "p_value" not in one


def test_cluster_analysis_rejects_changed_labels(tmp_path):
    _write_runs(tmp_path / "runs")
    path = next((tmp_path / "runs").rglob("baseline_astnn/test_predictions.csv.gz"))
    frame = pd.read_csv(path)
    frame.loc[0, "label"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Different test support"):
        paired_comparison(tmp_path / "runs", "full", "baseline/astnn", tmp_path / "result", repetitions=20)


def test_clone_types_require_provenance_and_use_fixed_threshold(tmp_path):
    frame = pd.DataFrame({"pair_id": ["a", "b", "c", "d"], "label": [1, 1, 0, 0], "probability": [.8, .2, .1, .9]})
    types = pd.DataFrame({"pair_id": ["a", "b"], "clone_type": ["Type-IV", "Type-III"]})
    frame.to_csv(tmp_path / "pred.csv", index=False)
    types.to_csv(tmp_path / "types.csv", index=False)
    with pytest.raises(ValueError, match="annotation_source"):
        report_clone_types(tmp_path / "pred.csv", tmp_path / "types.csv", .5, tmp_path / "result")
    types["annotation_source"] = "official-test-fixture"
    types.to_csv(tmp_path / "types.csv", index=False)
    status = report_clone_types(tmp_path / "pred.csv", tmp_path / "types.csv", .5, tmp_path / "result")
    assert status["annotated_positives"] == 2
    scores = pd.read_csv(tmp_path / "result/clone_type_metrics.csv").set_index("clone_type")
    assert scores.loc["Type-IV", "R"] == 1
    assert scores.loc["Type-III", "R"] == 0


def test_problem_bootstrap_resume_matches_uninterrupted_draws(tmp_path):
    from spectral_code.revision.training import SessionBudgetReached
    _write_runs(tmp_path / "runs")
    class Clock:
        def __init__(self):
            self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls > 7:
                raise SessionBudgetReached()
    pending = paired_comparison(tmp_path / "runs", "full", "baseline/astnn", tmp_path / "resumed", repetitions=30, clock=Clock())
    assert pending["status"] == "incomplete_session_budget"
    assert not (tmp_path / "resumed/paired_cluster_summary.json").exists()
    resumed = paired_comparison(tmp_path / "runs", "full", "baseline/astnn", tmp_path / "resumed", repetitions=30)
    full = paired_comparison(tmp_path / "runs", "full", "baseline/astnn", tmp_path / "full", repetitions=30)
    assert resumed == full
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "resumed/bootstrap_differences.csv"),
                                  pd.read_csv(tmp_path / "full/bootstrap_differences.csv"))


def test_aggregation_rejects_changed_pretrained_model_revisions_across_seeds(tmp_path):
    from spectral_code.revision.report import aggregate
    for i, seed in enumerate((42, 123, 231)):
        folder = tmp_path / "runs" / str(seed)
        folder.mkdir(parents=True)
        result = {"identity": str(seed), "arm": "baseline/qwen3_lora", "seed": seed,
                  "data_hash": "data", "data_group_hash": "data", "implementation_hash": "source",
                  "model_config": {"revision": "abc" if i != 2 else "def"},
                  "protocol": {"seed": seed, "max_epochs": 100}, "losses": {},
                  "test": {key: .6 for key in ("Accuracy", "F1", "AUPRC_AP", "AUROC", "MCC")}}
        (folder / "final_test.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Model revision"):
        aggregate(tmp_path / "runs", tmp_path / "report")
    path = tmp_path / "runs/231/final_test.json"
    result = json.loads(path.read_text())
    result["model_config"]["revision"] = "abc"
    path.write_text(json.dumps(result))
    report = aggregate(tmp_path / "runs", tmp_path / "report")
    assert report.iloc[0].Accuracy_mean == pytest.approx(.6)
