from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from spectral_code.revision import checkpoint_review as review
from spectral_code.revision.data import CanonicalGraph
from spectral_code.revision.model import RevisionModelConfig, RevisionPairModel
from spectral_code.revision.protocol import RunConfig, choose_threshold, fingerprint, metrics, run_identity
from spectral_code.revision.screening import VALIDATION_RECOVERY_HASH
from spectral_code.revision.training import LOSS_WEIGHTS


@pytest.fixture
def checkpoint_fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model_config = RevisionModelConfig(hidden_dim=16, latent_nodes=4, slot_iterations=1,
        density_bins=4, heat_samples=4, spectral_bands=3, spectral_signals=2, chebyshev_degree=2)
    model = RevisionPairModel(model_config)
    config = RunConfig(dataset="atcoder", seed=123, batch_size=4, effective_batch_size=4, amp=False)
    frames, graphs = {}, {}
    for split in ("train", "valid"):
        records = []
        for i in range(8):
            key = f"{split}{i}"
            graphs[key] = CanonicalGraph(key, "python", np.asarray([1, 2 + i % 3, 4], dtype=np.int64),
                np.full((3, 4), i + 1, dtype=np.int64), np.zeros(96, dtype=np.int64),
                tuple(np.asarray([[0, 1], [1, 2]], dtype=np.int32) for _ in range(3)), {})
            records.append(dict(pair_id=f"{split}_pair{i}", left_id=key,
                                right_id=f"{split}{(i + 1) % 8}", label=i % 2))
        frames[split] = pd.DataFrame(records)
    state = dict(training=asdict(config), model_config=asdict(model_config), losses=dict(LOSS_WEIGHTS),
        implementation_hash=VALIDATION_RECOVERY_HASH, data_hash="fixture_data", arm="full", complete=True,
        best_model=model.state_dict(), model=model.state_dict(), best_epoch=1, best_threshold=.5,
        history=[dict(epoch=1), dict(epoch=2)],
        best_valid=metrics(frames["valid"].label, np.linspace(.1, .9, 8), .5))
    state["identity"] = fingerprint(dict(run=run_identity(config, state["model_config"], state["losses"], state["data_hash"]),
        implementation=state["implementation_hash"], arm="full"))
    folder = tmp_path / "input/checkpoint"
    folder.mkdir(parents=True)
    path = folder / "progress.pt"
    torch.save(state, path)
    entry = dict(job_id="atcoder_full_seed123", dataset="atcoder", seed=123, arm="full", identity=state["identity"],
        data_hash=state["data_hash"], implementation_hash=state["implementation_hash"],
        checkpoint_sha256=review.sha256(path), archive="fixture.zip", archive_sha256="unused", checkpoint_member="x/progress.pt",
        model_config_hash=fingerprint(state["model_config"]), losses_hash=fingerprint(state["losses"]),
        training_config_hash=fingerprint(state["training"]))
    class Corpus:
        def load_graphs(self, ids, **kwargs):
            return {key: graphs[key] for key in ids}
    monkeypatch.setattr(review, "load_pair_data", lambda *args, **kwargs: (Corpus(), frames, "fixture_data"))
    return entry, path, frames, graphs


def test_diagnostics_are_read_only_resume_and_export_small_reports(checkpoint_fixture, tmp_path, monkeypatch):
    entry, checkpoint, frames, graphs = checkpoint_fixture
    before = checkpoint.read_bytes()
    result = review.run_checkpoint_review([entry], tmp_path, tmp_path / "reports", input_root=tmp_path / "input",
        diagnostics=True, train_pairs=8, descriptor_codes=8, require_gpu=False)
    assert result["status"] == "complete", result.get("errors")
    assert len(result["completed"]) == 2
    assert checkpoint.read_bytes() == before
    for r in result["completed"]:
        assert r["f1_best_epoch_recovered"] is False and r["training_updates"] == 0
        assert r["original_selection_metric"] == "Accuracy" and r["test_evaluated"] is False
        assert r["gradient_probe"]["mode"] == "eval_fp32_no_optimizer_no_updates"
        assert r["gradient_probe"]["gradient_groups"]["classifier"]["classification_norm"] > 0
        assert r["descriptors"]["codes"] == 8
        assert r["validation_f1_calibrated"]["F1"] >= r["recorded_threshold_metrics"]["F1"]
    with zipfile.ZipFile(result["download"]) as archive:
        assert not any(n.endswith(".pt") for n in archive.namelist())
        assert sum(n.endswith("validation_predictions.csv.gz") for n in archive.namelist()) == 2
    monkeypatch.setattr(review, "_predict_report", lambda *a, **k: pytest.fail("A completed view must be reused"))
    resumed = review.run_checkpoint_review([entry], tmp_path, tmp_path / "resumed", input_root=tmp_path / "input",
        diagnostics=True, train_pairs=8, descriptor_codes=8, require_gpu=False, resume_reports=tmp_path / "reports")
    assert resumed["status"] == "complete" and resumed["completed"] == result["completed"]


def test_missing_or_changed_checkpoint_never_reaches_model_loading(checkpoint_fixture, tmp_path, monkeypatch):
    entry, checkpoint, _, _ = checkpoint_fixture
    checkpoint.write_bytes(b"wrong checkpoint bytes")
    monkeypatch.setattr(review, "_read_state", lambda *a: pytest.fail("Unknown bytes must not be deserialized"))
    result = review.run_checkpoint_review([entry], tmp_path, tmp_path / "missing", input_root=tmp_path / "input")
    assert result["status"] == "missing_inputs" and result["missing_inputs"] == [entry["job_id"]]
    assert Path(result["download"]).is_file()


def test_named_zip_and_raw_checkpoint_are_aliases(checkpoint_fixture, tmp_path):
    entry, path, _, _ = checkpoint_fixture
    archive = tmp_path / "input/fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(path, entry["checkpoint_member"])
    entry["archive_sha256"] = review.sha256(archive)
    found = review.discover_checkpoints(tmp_path / "input", [entry])
    assert len(found) == 1 and len(found[entry["job_id"]]) == 2
    entry["archive_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        review.discover_checkpoints(tmp_path / "input", [entry])


def _write_core_reports(folder, seeds=(42, 123, 231)):
    entries = []
    for arm in review.CORE_ARMS:
        for seed in seeds:
            job = f"codenet_{arm}_seed{seed}"
            entry = dict(job_id=job, checkpoint_sha256=job, dataset="codenet", arm=arm, seed=seed)
            entries.append(entry)
            path = folder / job
            path.mkdir(parents=True)
            p = pd.DataFrame(dict(pair_id=["p0", "p1", "p2", "p3"], left_id=["0", "1", "2", "3"],
                right_id=["1", "2", "3", "0"], label=[0, 1, 0, 1], probability=[.1, .8, .2, .9]))
            p.to_csv(path / "validation_predictions.csv.gz", index=False)
            _, f1 = choose_threshold(p.label, p.probability, "F1")
            report = dict(job_id=job, checkpoint_sha256=job, dataset="codenet", status="complete", arm=arm, seed=seed,
                weight_view="best_model", test_evaluated=False, training_updates=0, f1_best_epoch_recovered=False,
                threshold_calibration="posthoc_validation_F1_at_existing_weights", original_selection_metric="Accuracy",
                settings=dict(analysis_source_sha256=review.sha256(Path(review.__file__))),
                validation_support_hash=review.pair_support(p), validation_f1_calibrated=f1, recorded_threshold_metrics=f1,
                artifacts={"validation_predictions.csv.gz": review.sha256(path / "validation_predictions.csv.gz")})
            (path / "analysis_result.json").write_text(json.dumps(report))
    return entries


def test_collector_requires_all_six_conditions_three_seeds_and_checks_support(tmp_path):
    entries = _write_core_reports(tmp_path / "input")
    missing = tmp_path / "input/codenet_full_seed42/analysis_result.json"
    original = missing.read_bytes()
    missing.unlink()
    result = review.collect_checkpoint_reviews(tmp_path / "input", tmp_path / "partial", entries)
    assert result["core_complete"] is False
    assert not (tmp_path / "partial/core_f1_summary.csv").exists()
    missing.write_bytes(original)
    result = review.collect_checkpoint_reviews(tmp_path / "input", tmp_path / "complete", entries)
    assert result["core_complete"] is True
    means = pd.read_csv(tmp_path / "complete/core_f1_summary.csv")
    assert len(means) == 6 and set(means.seeds) == {3} and set(means.F1_mean) == {1.0}
    report = json.loads(missing.read_text()); report["validation_support_hash"] = "different support"
    missing.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="ordered support"):
        review.collect_checkpoint_reviews(tmp_path / "input", tmp_path / "bad", entries)


def test_exported_predictions_cannot_change_without_detecting_it(tmp_path):
    entries = _write_core_reports(tmp_path / "input", seeds=(42,))
    path = tmp_path / "input/codenet_full_seed42/validation_predictions.csv.gz"
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        review.collect_checkpoint_reviews(tmp_path / "input", tmp_path / "out", entries)


def test_diagnostic_failure_is_visible_and_budget_stops_are_not_swallowed():
    errors = []
    def invalid():raise FloatingPointError("non-finite diagnostic gradients")
    result = review._diagnostic_call("gradients", invalid, errors)
    assert result["status"] == "failed" and errors[0]["probe"] == "gradients"
    def exhausted():raise review.SessionBudgetReached()
    with pytest.raises(review.SessionBudgetReached):
        review._diagnostic_call("gradients", exhausted, errors)


def test_recollecting_partial_reports_removes_old_complete_tables(tmp_path):
    entries = _write_core_reports(tmp_path / "input")
    out = tmp_path / "out"
    review.collect_checkpoint_reviews(tmp_path / "input", out, entries)
    assert (out / "core_f1_summary.csv").exists()
    (tmp_path / "input/codenet_full_seed42/analysis_result.json").unlink()
    review.collect_checkpoint_reviews(tmp_path / "input", out, entries)
    assert not (out / "core_f1_summary.csv").exists()


def test_notebooks_compile_and_embedded_runtime_imports_without_repository(tmp_path):
    root = Path(__file__).resolve().parents[1]
    roster = list(__import__("csv").DictReader((root / "revision/queues/checkpoint_review_notebooks.csv").open()))
    assert len(roster) == 5 and sum(r["gpu"] == "True" for r in roster) == 4
    first_bootstrap = None
    for row in roster:
        path = root / "revision" / row["notebook"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert b"\r\n" not in path.read_bytes()
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for c in notebook["cells"]:
            if c["cell_type"] == "code":compile("".join(c["source"]), path.name, "exec")
        config = "".join(notebook["cells"][1]["source"])
        assert "FINAL_TEST = False" in config
        bootstrap = "".join(notebook["cells"][2]["source"])
        if first_bootstrap is None:first_bootstrap = bootstrap
        assert "pip install" not in "".join(notebook["cells"][3]["source"])
    bootstrap = first_bootstrap.replace('Path("/kaggle/working/checkpoint_review_runtime")', repr(tmp_path / "runtime"))
    script = "from pathlib import Path\n" + bootstrap + "\nfrom spectral_code.revision.checkpoint_review import REVIEW_VERSION\nprint(REVIEW_VERSION)\n"
    # repr(Path) uses WindowsPath/PosixPath constructors; provide them explicitly.
    script = "from pathlib import WindowsPath, PosixPath\n" + script
    env = dict(os.environ, PYTHONPATH="")
    runner = tmp_path / "standalone_bootstrap.py"
    runner.write_text(script, encoding="utf-8", newline="\n")
    run = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert review.REVIEW_VERSION in run.stdout
