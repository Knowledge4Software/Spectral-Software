from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import run_bcb_data_extraction


if __name__ == "__main__":
    run_bcb_data_extraction(
        "1",
        sys.argv[1:],
        variant="non_clone",
        defaults=[
            "--max-non-clone-code-ids", os.getenv("BCB_NON_CLONE_MAX_CODE_IDS", "0"),
            "--negative-pool", "false-positives",
            "--non-clone-only",
            "--keep-getter-setters",
            "--drop-non-clone-pairs-both-three-line",
            "--write-all-filtered-non-clone-code-ids",
            "--all-filtered-non-clone-pairs",
        ],
    )
