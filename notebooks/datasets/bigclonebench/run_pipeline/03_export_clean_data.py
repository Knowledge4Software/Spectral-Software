"""Final clean export for any already-extracted BigCloneBench variant."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.pipeline_cli import run_dataset_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and finalize a BigCloneBench variant.")
    parser.add_argument("--variant", required=True, help="For example: 1, 2, 3/moderate, 4, or non_clone.")
    args = parser.parse_args()
    run_dataset_section("bcb", args.variant, "export", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
