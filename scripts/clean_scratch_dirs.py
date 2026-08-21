"""Delete regenerable scratch left over from building the clean datasets.

Graph extraction writes large intermediates (Joern CPG batches, per-record
caches) beside its outputs. Once a dataset's ``clean_data/`` exists and has been
published to ``outputs/kaggle_datasets``, those intermediates are only useful
for resuming an interrupted build.

Nothing here is an input to the paper: every table and figure is built from
``outputs/kaggle/`` (Kaggle result archives) and the published clean-data ZIPs.

Prints what it would remove and exits; pass --delete to actually remove.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = _ROOT.parent / "outputs"

# (path, why it is safe to remove)
SCRATCH = [
    ("codenet_4l_clone_50k",
     "Joern batch scratch; the dataset it built is published as "
     "kaggle_datasets/codenet_4l_clean_data.zip"),
    ("codenet_4l_distributed",
     "per-laptop non-clone extraction caches from the distributed build"),
    ("graph_record_cache_laptop2",
     "per-record graph cache from the same distributed build"),
    ("codenet_4l_clone50k_diff50k_nonclone_java_cpp_cache", "non-clone pairing cache"),
    ("codenet_4l_clone50k_diff50k_nonclone_java_cpp_work", "non-clone pairing scratch"),
    ("codenet_4l_clone50k_diff50k_nonclone_py_csharp_cache", "non-clone pairing cache"),
    ("codenet_4l_clone50k_diff50k_nonclone_py_csharp_work", "non-clone pairing scratch"),
]

# Removed only with --include-stale: superseded result exports, kept by default
# because they are small and may still be cited in older drafts.
STALE = [
    ("canonical_experiments", "August 1 preflight, superseded by the V3 runs"),
    ("visualizations", "HTML previews of the non-clone scope study"),
]


def size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true",
                        help="actually remove (default: report only)")
    parser.add_argument("--include-stale", action="store_true",
                        help="also remove the superseded result exports")
    args = parser.parse_args()

    targets = SCRATCH + (STALE if args.include_stale else [])
    total = 0
    for name, why in targets:
        path = OUTPUTS / name
        if not path.exists():
            print(f"  absent   {name}")
            continue
        bytes_ = size(path)
        total += bytes_
        print(f"  {'removing' if args.delete else 'would remove'} "
              f"{bytes_/2**30:6.2f} GB  {name}\n             {why}")
        if args.delete:
            shutil.rmtree(path, ignore_errors=True)

    print(f"\n{'freed' if args.delete else 'would free'}: {total/2**30:.2f} GB")
    if not args.delete:
        print("re-run with --delete to remove")


if __name__ == "__main__":
    main()
