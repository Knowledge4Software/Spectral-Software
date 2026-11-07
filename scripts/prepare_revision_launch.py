"""Make a parallel launch list from the baseline queue and reported progress."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.revision_layout import REVISION_ROOT


def prepare(root: Path = REVISION_ROOT, capacity: int = 24) -> None:
    if capacity < 1:
        raise ValueError("capacity must be positive")
    queue_root = root / "queues"
    with (queue_root / "stage1_run_queue.csv").open(encoding="utf-8", newline="") as stream:
        jobs = list(csv.DictReader(stream))
    with (root / "results/run_progress.csv").open(encoding="utf-8", newline="") as stream:
        progress = {row["job_id"]: row for row in csv.DictReader(stream)}
    if progress.keys() - {job["job_id"] for job in jobs}:
        raise ValueError("Progress tracker contains jobs outside the baseline queue")
    # A manual report keeps a successful job out of the launch list, but does
    # not turn it into an authenticated score in the scientific aggregator.
    excluded = {"complete_reported", "complete_verified", "running", "incomplete_session_budget"}
    pending = [job for job in jobs if progress.get(job["job_id"], {}).get("status") not in excluded]
    for name, selected in (("launch_24.csv", pending[:capacity]), ("launch_next.csv", pending[capacity:])):
        with (queue_root / name).open("w", encoding="utf-8", newline="") as stream:
            fields = ["launch_slot", "account", "account_slot", *jobs[0].keys()]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, job in enumerate(selected, start=1):
                writer.writerow({"launch_slot": index, "account": (index + 1) // 2,
                                 "account_slot": (index - 1) % 2 + 1, **job})
    lines = ["# Parallel launch list", "", "Paths below open the actual `.ipynb` files in this workspace.",
             "Attach the dataset named in each row. Keep seed 42 and `FINAL_TEST=False`.",
             "Account numbers are planning labels; no credentials are stored.", "",
             "Reported complete/running jobs are excluded. Confirm the tracker before starting",
             "another copy of a job. Runtime and memory still need verification per method.", ""]
    for label, selected in ((f"Launch now: {min(capacity, len(pending))} jobs", pending[:capacity]),
                            (f"Start when slots free: {max(0, len(pending) - capacity)} jobs", pending[capacity:])):
        lines.extend(["## " + label, "", "| Slot | Dataset | Notebook | Internet |",
                      "| ---: | --- | --- | --- |"])
        for index, job in enumerate(selected, start=1):
            lines.append(f"| {index} | {job['dataset']} | [{job['job_id']}](../{job['notebook']}) | {job['internet']} |")
        lines.append("")
    lines.extend(["## Exact dataset versions", "",
                  "See [the dataset guide](../datasets/README_FA.md) for ZIP paths and hashes.",
                  "Preserve every saved output and [update the run tracker](../results/run_progress.csv).", ""])
    (queue_root / "LAUNCH_24.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Prepared {min(capacity, len(pending))} launch jobs and {max(0, len(pending) - capacity)} waiting jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=24)
    args = parser.parse_args()
    prepare(capacity=args.capacity)
