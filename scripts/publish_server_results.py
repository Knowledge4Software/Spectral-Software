"""Publish scoped validation summaries from a GPU run to a dedicated Git branch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time


METRICS = ("F1", "Accuracy", "AUROC", "AUPRC_AP", "P", "R", "MCC", "BalancedAccuracy")
VALID_STATUS = {
    "training", "validation_pending", "complete", "incomplete_session_budget",
    "failed_nonfinite", "failed_validation_nonfinite", "incomplete_tokenization",
}


def snapshot(root):
    results = {}
    for path in sorted(root.glob("*/*/status.json")):
        name = path.relative_to(root).parts[0]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or path.is_symlink():
            continue
        try:
            if path.stat().st_size > 8 * 1024**2:
                continue
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Another process may be rewriting the status file.
            continue
        if not isinstance(status, dict) or status.get("status") not in VALID_STATUS:
            continue
        entry = {"status": status["status"], "complete": status.get("complete") is True}
        for key in ("epoch", "batch", "best_epoch", "optimizer_steps"):
            if type(status.get(key)) is int:
                entry[key] = status[key]
        valid = status.get("best_valid")
        if isinstance(valid, dict):
            entry["best_validation"] = {
                key: valid[key] for key in METRICS
                if type(valid.get(key)) in (int, float) and math.isfinite(valid[key])
            }
        results[name] = entry
    return results


def git(path, *args, capture=False):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=capture, text=True, timeout=90).stdout


def publish(args):
    if args.ssh:
        command = shlex.join(["python3", args.remote_script, str(args.results), "--snapshot"])
        process = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", args.ssh, command],
            check=True, capture_output=True, text=True, timeout=45,
        )
        updates = json.loads(process.stdout)
    else:
        updates = snapshot(args.results)
    if not updates:
        return
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    branch = f"gpu-results/{args.run_id}"
    if not (work / ".git").is_dir():
        git(work, "init", "-b", branch)
        git(work, "config", "user.name", "SPECTRA results")
        git(work, "config", "user.email", "spectra-results@users.noreply.github.com")
        git(work, "remote", "add", "origin", args.remote)
    if (git(work, "symbolic-ref", "--short", "HEAD", capture=True).strip() != branch
            or git(work, "remote", "get-url", "origin", capture=True).strip() != args.remote):
        raise ValueError("Use a dedicated publisher directory for this run and remote")
    report = work / "validation-progress.json"
    prior = json.loads(report.read_text()) if report.is_file() else {}
    merged = {**prior, **updates}
    if merged != prior:
        report.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # A previous interruption may have occurred after writing, staging, or committing.
    git(work, "add", "--", report.name)
    if git(work, "diff", "--cached", "--name-only", capture=True).strip() == report.name:
        git(work, "commit", "-m", "Update SPECTRA validation progress")
    head = git(work, "rev-parse", "HEAD", capture=True).strip()
    receipt = work / ".git" / "spectra-published"
    if not receipt.is_file() or receipt.read_text().strip() != head:
        # Retrying a failed push is safe. Never force-push a branch with newer data.
        git(work, "push", "origin", f"HEAD:refs/heads/{branch}")
        receipt.write_text(head + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--remote")
    parser.add_argument("--run-id")
    parser.add_argument("--ssh", help="SSH alias or user@host with a pinned host key")
    parser.add_argument("--remote-script", help="absolute path of this script on the GPU node")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(args.results)))
        return
    if (not args.work or not args.remote or not args.run_id
            or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id)):
        parser.error("Provide --work, --remote and a unique --run-id")
    if args.ssh and not args.remote_script:
        parser.error("--ssh requires --remote-script")
    while True:
        try:
            publish(args)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f"Publish pending: {type(exc).__name__}", flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(max(15, args.interval))


if __name__ == "__main__":
    main()
