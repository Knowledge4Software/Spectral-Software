"""Final clean export for any already-extracted Semantic Benchmark language."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.pipeline_cli import run_dataset_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and finalize a Semantic Benchmark language.")
    parser.add_argument("--language", required=True, help="For example: c, cs, java, or python.")
    args = parser.parse_args()
    run_dataset_section("semantic", args.language, "export", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
