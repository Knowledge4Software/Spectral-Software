"""Build the fixed 12k CodeNet package for the non-clone scope study.

The package contains exactly 4,000 pairs from each of these strata:
clone, hard_nonclone (Accepted/Wrong-Answer), and nonclone_diff_problem.
Mutation pairs are intentionally excluded without modifying the source archive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_RUNNER = PROJECT_ROOT / "notebooks/datasets/codenet_4l/run_pipeline/run_all.py"
DATA_ROOT = PROJECT_ROOT.parent / "data"
OUTPUT_ROOT = PROJECT_ROOT.parent / "outputs"

ARCHIVE = DATA_ROOT / "codenet dataset.zip"
PREPARED_DIR = DATA_ROOT / "codenet_4l_nonclone_12k_prepared"
GRAPH_OUTPUT_DIR = OUTPUT_ROOT / "codenet_4l_nonclone_12k"
PAIR_KINDS = "clone,hard_nonclone,nonclone_diff_problem"
GRAPH_TYPES = "ast,cfg,ddg,cpg"
SAMPLE_SIZE = 12_000
MIN_PROGRAM_LINES = 20
MAX_PROGRAM_LINES = 50
STAGES = ("01", "02", "03", "05")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=STAGES, default="01")
    parser.add_argument("--stop-after", choices=STAGES, default="05")
    parser.add_argument("--languages", help="Optional comma-separated graph-extraction subset")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if int(args.start_at) > int(args.stop_after):
        parser.error("--start-at must not be after --stop-after")

    command = [
        sys.executable,
        str(BASE_RUNNER),
        "--archive", str(ARCHIVE),
        "--prepared-dir", str(PREPARED_DIR),
        "--output-dir", str(GRAPH_OUTPUT_DIR),
        "--sample-size", str(SAMPLE_SIZE),
        "--pair-kinds", PAIR_KINDS,
        "--graph-types", GRAPH_TYPES,
        "--min-program-lines", str(MIN_PROGRAM_LINES),
        "--max-program-lines", str(MAX_PROGRAM_LINES),
        "--start-at", args.start_at,
        "--stop-after", args.stop_after,
    ]
    if args.languages:
        command.extend(("--languages", args.languages))
    for enabled, flag in (
        (args.force_prepare, "--force-prepare"),
        (args.keep_intermediates, "--keep-intermediates"),
        (args.no_zip, "--no-zip"),
    ):
        if enabled:
            command.append(flag)

    print("[*] Fixed CodeNet non-clone scope study")
    print(f"    archive:  {ARCHIVE}")
    print(f"    prepared: {PREPARED_DIR}")
    print(f"    output:   {GRAPH_OUTPUT_DIR}")
    print(f"    pairs:    {SAMPLE_SIZE:,} ({PAIR_KINDS})")
    print(f"    graphs:   {GRAPH_TYPES}")
    print(f"    lines:    {MIN_PROGRAM_LINES}..{MAX_PROGRAM_LINES} inclusive, both endpoints")
    print("[*]", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Rerun the same command to resume the active language safely.")
        raise SystemExit(130)
