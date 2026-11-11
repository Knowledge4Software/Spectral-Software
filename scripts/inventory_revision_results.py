"""Inventory archived results without silently selecting among conflicting runs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def inventory(source: Path, output: Path):
    rows = []
    for archive_path in sorted(source.rglob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                name = info.filename.lower()
                if not (name.endswith(".csv") and "result" in name) or any(s in name for s in ("language", "breakdown", "combined")):
                    continue
                raw = archive.read(info)
                try:
                    table = pd.read_csv(io.BytesIO(raw))
                except (ValueError, pd.errors.EmptyDataError):
                    continue
                if "F1" not in table:
                    continue
                for record in table.to_dict("records"):
                    fields = ("Dataset", "Method", "RunProfile", "P", "R", "F1", "Acc", "TrainPairs", "ValidPairs", "TestPairs",
                              "Seed", "BestEpoch", "ValidationSelectionMetric", "ValidationBalanced", "TrainableParameters", "RuntimeSeconds")
                    rows.append({"archive": str(archive_path.resolve()), "archive_bytes": archive_path.stat().st_size,
                                 "member": info.filename, "member_crc32": info.CRC,
                                 "result_sha256": hashlib.sha256(raw).hexdigest(),
                                 **{key: record.get(key) for key in fields}})
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "legacy_result_inventory.csv", index=False)
    (output / "README.md").write_text(
        "Archived submission results only. Each row identifies the exact result member and SHA-256. "
        "Multiple runs are retained rather than automatically choosing the best test score. "
        "These rows must not be pooled with new problem-disjoint revision experiments.\n", encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "outputs/paper/kaggle/RQ2")
    parser.add_argument("--output", type=Path, default=ROOT / "output/revision/result_inventory")
    args = parser.parse_args()
    frame = inventory(args.source, args.output)
    print(f"Inventoried {len(frame)} archived result rows in {args.output}")
