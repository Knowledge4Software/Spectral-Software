"""Small, durable Kaggle queue. Credentials live outside this checkout.

Run with --root /opt/spectral-kaggle --once or --watch. Only explicitly
registered jobs are submitted. Ambiguous submissions require reconciliation;
they are never retried automatically. No training or bulk downloads occur here.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from datetime import datetime, timezone
from urllib.parse import urlparse


REPORT_NAMES = {"status.json", "benchmark.json", "cost.json", "summary.json"}
ACTIVE = {"QUEUED", "RUNNING"}
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_BATCH_BYTES = 32 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def as_dict(value):
    return value.__class__.to_dict(value)


def safe_error(exc):
    # Exception text can include authenticated request headers or signed URLs.
    response = getattr(exc, "response", None)
    return {"type": type(exc).__name__, "http_status": getattr(response, "status_code", None)}


def remaining_seconds(quota):
    # Use SDK timedelta fields, including reserved sessions.
    gpu = quota.gpu_quota
    if gpu is None:
        return 0.0
    return max(0.0, (gpu.total_time_allowed.total_seconds()
                     - gpu.time_used.total_seconds() - gpu.time_reserved.total_seconds()))


def quota_snapshot(quota):
    # SDK2.2.5's generic duration serializer drops whole days (30h -> 6h).
    # Preserve total_seconds explicitly in our ledger.
    gpu = quota.gpu_quota
    refresh = quota.quota_refresh_time
    if isinstance(refresh, datetime) and refresh.tzinfo is None:
        refresh = refresh.replace(tzinfo=timezone.utc)
    return {"refresh_time": str(refresh),
            "gpu_used_seconds": gpu.time_used.total_seconds() if gpu else None,
            "gpu_reserved_seconds": gpu.time_reserved.total_seconds() if gpu else None,
            "gpu_allowed_seconds": gpu.total_time_allowed.total_seconds() if gpu else None,
            "gpu_remaining_seconds": remaining_seconds(quota)}


def kernel_ref(value):
    path = urlparse(value).path.strip("/")
    if path.startswith("code/"):
        path = path[5:]
    return path


def report_files(files):
    chosen = []
    for row in files:
        name = row["name"]
        path = PurePosixPath(name)
        size = int(row["size"])
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe remote output path")
        if path.name in REPORT_NAMES:
            if size < 0 or size > MAX_REPORT_BYTES:
                raise ValueError("Report exceeds download allowance")
            chosen.append(name)
    if sum(int(r["size"]) for r in files if r["name"] in chosen) > MAX_BATCH_BYTES:
        raise ValueError("Reports exceed download allowance")
    return chosen


def classify_report(report, job):
    """A successful Kaggle worker alone is not a completed experiment."""
    if (report.get("data_hash") != job["data_hash"]
            or report.get("implementation_hash") != job["implementation_hash"]
            or report.get("arm") != job["arm"]
            or report.get("training", {}).get("seed") != job["seed"]
            or report.get("training", {}).get("selection_metric") != job.get("metric", "F1")
            or (job.get("identity") and report.get("identity") != job["identity"])
            or report.get("test_evaluated") is not False):
        return "needs_review_protocol"
    if report.get("status") == "complete" and report.get("complete") is True:
        return "complete_reports_collected"
    if report.get("status") == "incomplete_session_budget":
        return "needs_continuation"
    return "needs_review_experiment_status"


def check_sources(root, job):
    folder = root / "jobs" / job["job_id"]
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    code = folder / meta["code_file"]
    if hashlib.sha256(code.read_bytes()).hexdigest() != job["sha256"]:
        raise ValueError("Frozen notebook hash mismatch")
    if (meta["id"] != job["kernel"] or meta.get("dataset_sources") != [job["dataset"]]
            or meta.get("kernel_sources", []) != job.get("kernel_sources", [])
            or meta.get("enable_internet") is not job.get("enable_internet", False)
            or meta.get("is_private") is not True or meta.get("enable_gpu") is not True
            or meta.get("docker_image") != job["docker_image"]):
        raise ValueError("Submission metadata does not match registration")
    return folder


def submit_once(api, root, job, state, save):
    folder = check_sources(root, job)
    # Persist before the network call. A timeout/crash must not launch a duplicate.
    state.update(phase="submission_uncertain", submission_attempted_at=now())
    save()
    with contextlib.redirect_stdout(io.StringIO()):
        response = as_dict(api.kernels_push(str(folder), timeout=39600, acc="NvidiaTeslaT4"))
    state["submission_response"] = response
    invalid = any(response.get(k) for k in ("error", "invalidDatasetSources", "invalidKernelSources",
                                           "invalidCompetitionSources", "invalidModelSources"))
    if invalid or not response.get("versionNumber") or kernel_ref(response.get("ref", "")) != job["kernel"]:
        state["phase"] = "submission_rejected_needs_review"
    else:
        state.update(phase="submitted", version=response["versionNumber"], submitted_at=now())
    save()


def collect(api, root, job, state):
    # Each job has a dedicated slug and is never overwritten by this manager.
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug = job["kernel"].split("/")
    request.version_label = f"v{state['version']}"
    request.page_size = 100
    files = []
    with api.build_kaggle_client() as client:
        for _ in range(100):
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if not response.next_page_token:
                break
            request.page_token = response.next_page_token
        else:
            raise ValueError("Output listing exceeded page bound")
    dest = root / "reports" / job["job_id"] / str(state["version"])
    dest.mkdir(parents=True, exist_ok=True)
    rows = [{"name": item.file_name, "size": 0} for item in files]
    selected = set(report_files(rows))
    write_json(dest / "output_manifest.json", [{"name": row["name"]} for row in rows])
    total = 0
    for item in files:
        if item.file_name not in selected:
            continue
        target = dest / item.file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.part")
        size = 0
        try:
            with requests.get(item.url, stream=True, timeout=(10, 30)) as download:
                download.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in download.iter_content(64 * 1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_REPORT_BYTES or total > MAX_BATCH_BYTES:
                            raise ValueError("Report exceeds download allowance")
                        stream.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    reports = list(dest.rglob("status.json"))
    state["report_directory"] = str(dest)
    if len(reports) != 1:
        state["phase"] = "needs_review_missing_status"
    elif state.get("kaggle_status") != "COMPLETE":
        state["phase"] = "needs_review_kaggle_failure"
    else:
        report = json.loads(reports[0].read_text())
        state["phase"] = classify_report(report, job)
        state["experiment"] = {key: report.get(key) for key in
                               ("status", "complete", "epoch", "best_epoch", "best_valid", "stop_reason")}
    state["collected_at"] = now()


def tick(root, credentials):
    config = json.loads((root / "queue.json").read_text())
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}, "accounts": {}}
    accounts_by_job = {job["job_id"]: job["account"] for job in config["jobs"]}
    submission_accounts = config.get("submission_accounts")
    minimum_gap = float(config.get("submission_gap_seconds", 0))
    # Importing the Kaggle package authenticates once as a side effect.
    # Use the permitted submission account for that import when configured.
    bootstrap_account = next(iter(submission_accounts or credentials))
    os.environ["KAGGLE_API_TOKEN"] = credentials[bootstrap_account]
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
    def save():
        state["updated_at"] = now()
        write_json(state_path, state)
    for job in config["jobs"]:
        item = state["jobs"].setdefault(job["job_id"], {"phase": "registered"})
        if item["phase"] not in {"registered", "waiting_quota", "submitted", "monitoring"}:
            continue
        try:
            os.environ["KAGGLE_API_TOKEN"] = credentials[job["account"]]
            api = KaggleApi()
            api.authenticate()
            quota = api.quota_view()
            state["accounts"][job["account"]] = {"checked_at": now(), "quota": quota_snapshot(quota)}
            if item["phase"] in {"registered", "waiting_quota"}:
                if submission_accounts is not None and job["account"] not in submission_accounts:
                    continue
                if minimum_gap > 0:
                    attempts = [
                        datetime.fromisoformat(attempt.get("submitted_at") or attempt["submission_attempted_at"])
                        for entry in state["jobs"].values()
                        for attempt in (entry, entry.get("cancelled_attempt") or {})
                        if attempt.get("submission_attempted_at")
                    ]
                    if attempts and (datetime.now(timezone.utc) - max(attempts)).total_seconds() < minimum_gap:
                        continue
                if remaining_seconds(quota) < job.get("quota_reservation_seconds", 39600):
                    item["phase"] = "waiting_quota"
                    save()
                    continue
                # Kaggle still checks accelerator capacity and interactive sessions.
                limit = int(config.get("account_concurrency", {}).get(job["account"], 1))
                active = sum(
                    entry.get("phase") in {"submitted", "monitoring", "submission_uncertain"}
                    for job_id, entry in state["jobs"].items()
                    if accounts_by_job.get(job_id) == job["account"]
                )
                if active >= limit:
                    continue
                api.dataset_list_files(job["dataset"], page_size=1)
                # Do not overwrite an existing notebook, including a prior run
                # whose ledger was lost. Search all pages of this specific slug.
                existing = api.kernels_list(mine=True, search=job["kernel"].split("/")[1], page_size=100)
                if item.get("cancelled_attempt"):
                    cancelled = item["cancelled_attempt"]
                    latest = api.kernels_status(job["kernel"])
                    if latest.status.name != "CANCEL_ACKNOWLEDGED":
                        item["phase"] = "needs_reconciliation_cancelled_kernel"
                        save()
                        continue
                    versioned = api.kernels_status(f"{job['kernel']}/{cancelled['version']}")
                    if versioned.status.name != "CANCEL_ACKNOWLEDGED":
                        item["phase"] = "needs_reconciliation_cancelled_kernel"
                        save()
                        continue
                elif any(k.ref == job["kernel"] for k in existing):
                    item["phase"] = "needs_reconciliation_existing_kernel"
                    save()
                    continue
                submit_once(api, root, job, item, save)
            if item["phase"] in {"submitted", "monitoring"}:
                status = as_dict(api.kernels_status(f"{job['kernel']}/{item['version']}"))
                # QUEUED is enum zero and generic SDK serialization omits it.
                kaggle_status = status.get("status", "QUEUED")
                item.update(phase="monitoring", kaggle_status=kaggle_status, checked_at=now())
                if kaggle_status == "RUNNING":
                    item["observed_running"] = True
                if (kaggle_status == "CANCEL_ACKNOWLEDGED"
                        and job.get("restart_after_cancel_version") == item["version"]
                        and not item.get("observed_running")
                        and not item.get("cancelled_attempt")):
                    cancelled = dict(item)
                    item.clear()
                    item.update(phase="registered", cancelled_attempt=cancelled)
                    save()
                elif kaggle_status in TERMINAL:
                    collect(api, root, job, item)
                elif kaggle_status not in ACTIVE and kaggle_status != "CANCEL_REQUESTED":
                    item["phase"] = "needs_review_unknown_kaggle_status"
            item.pop("last_error", None)
        except (Exception, SystemExit) as exc:
            item["last_error"] = safe_error(exc)
            item["last_error_at"] = now()
        finally:
            os.environ.pop("KAGGLE_API_TOKEN", None)
            save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/spectral-kaggle"))
    parser.add_argument("--credentials", type=Path, default=Path("/root/.config/spectral-kaggle/accounts.json"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()
    os.umask(0o077)
    args.root.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock prevents duplicate foreground/service instances.
    import fcntl
    with (args.root / "manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        credentials = json.loads(args.credentials.read_text())
        while True:
            if shutil.disk_usage(args.root).free < 1024 ** 3:
                raise RuntimeError("Less than 1 GiB free; manager stopped before downloads")
            state = tick(args.root, credentials)
            print(json.dumps({"at": now(), "jobs": {k: {a: v.get(a) for a in
                  ("phase", "kaggle_status", "version", "last_error")} for k, v in state["jobs"].items()}}), flush=True)
            if not args.watch:
                break
            time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
