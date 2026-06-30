from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[7]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import run_bcb_data_extraction


if __name__ == "__main__":
    run_bcb_data_extraction(
        "3",
        sys.argv[1:],
        variant="3/strong",
        defaults=[
            "--target-pairs", "1000000",
            "--positive-fraction", "1.0",
            "--max-positive-pairs", "1000000",
            "--type3-min-similarity", "0.70",
            "--type3-max-similarity", "0.90",
            "--preselect-positive-pairs-before-code-scan",
            "--positive-only",
        ],
    )
