import json
from types import SimpleNamespace
from datetime import timedelta

import pytest

from scripts.kaggle_queue_manager import (classify_report, remaining_seconds,
                                         report_files, submit_once, kernel_ref, quota_snapshot)


def test_quota_counts_reserved_sessions():
    q = SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=timedelta(hours=30),
        time_used=timedelta(hours=5), time_reserved=timedelta(hours=10)))
    assert remaining_seconds(q) == 15 * 3600


def test_full_day_quota_and_kaggle_receipt_paths():
    q = SimpleNamespace(quota_refresh_time="2026-11-28T00:00:00Z", gpu_quota=SimpleNamespace(
        total_time_allowed=timedelta(hours=30), time_used=timedelta(), time_reserved=timedelta()))
    assert quota_snapshot(q)["gpu_allowed_seconds"] == 108000
    assert kernel_ref("/code/user/slug") == "user/slug"
    assert kernel_ref("https://www.kaggle.com/code/user/slug") == "user/slug"
    assert kernel_ref("user/slug") == "user/slug"


def test_bulk_and_traversal_outputs_are_not_downloaded():
    assert report_files([{"name": "run/progress.pt", "size": 20 * 1024**3},
                         {"name": "run/status.json", "size": 4000}]) == ["run/status.json"]
    with pytest.raises(ValueError):
        report_files([{"name": "../status.json", "size": 1}])
    with pytest.raises(ValueError):
        report_files([{"name": "status.json", "size": 10 * 1024**2}])


def test_worker_success_requires_valid_completed_experiment():
    job = {"data_hash": "d", "implementation_hash": "i", "arm": "mean_max_classification_only", "seed": 42}
    report = {**job, "training": {"seed": 42, "selection_metric": "F1"},
              "test_evaluated": False, "complete": True, "status": "complete"}
    assert classify_report(report, job) == "complete_reports_collected"
    report.update(complete=False, status="incomplete_session_budget")
    assert classify_report(report, job) == "needs_continuation"
    report["training"]["selection_metric"] = "Accuracy"
    assert classify_report(report, job) == "needs_review_protocol"


def test_submission_timeout_is_persisted_before_request(tmp_path):
    import hashlib
    folder = tmp_path / "jobs" / "j"
    folder.mkdir(parents=True)
    (folder / "code.ipynb").write_text("{}")
    job = {"job_id": "j", "kernel": "user/job", "dataset": "user/data", "docker_image": "image",
           "sha256": hashlib.sha256(b"{}").hexdigest()}
    (folder / "kernel-metadata.json").write_text(json.dumps({"code_file": "code.ipynb", "id": "user/job",
        "dataset_sources": ["user/data"], "kernel_sources": [], "enable_internet": False,
        "is_private": True, "enable_gpu": True, "docker_image": "image"}))
    state, snapshots = {}, []
    class Api:
        def kernels_push(self, *args, **kwargs):
            assert snapshots == ["submission_uncertain"]
            raise TimeoutError()
    with pytest.raises(TimeoutError):
        submit_once(Api(), tmp_path, job, state, lambda: snapshots.append(state["phase"]))
    assert state["phase"] == "submission_uncertain"
    class Accepted:
        @classmethod
        def to_dict(cls, obj):
            return {"ref": "/code/user/job", "versionNumber": 1}
    class SuccessfulApi:
        def kernels_push(self, *args, **kwargs):
            return Accepted()
    state.clear()
    submit_once(SuccessfulApi(), tmp_path, job, state, lambda: None)
    assert state["phase"] == "submitted" and state["version"] == 1
