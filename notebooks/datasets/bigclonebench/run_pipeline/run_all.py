"""Run only the BigCloneBench cross-type threshold experiments.

Dataset extraction lives under ``types/<variant>/build``. Portable Type-4
plus non-clone export is handled
by ``04_create_balanced_subset.py`` in this directory.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.pipeline_cli import run_numbered_pipeline


STAGES = [
    ("00", "00_train_balanced_type123_type4_threshold.py"),
    ("01", "01_train_type123_threshold_test_type4.py"),
]
if __name__ == "__main__":
    run_numbered_pipeline(
        STAGES,
        description="Run BCB cross-type threshold experiments.",
        completion_message="All selected BCB cross-type pipeline stages completed.",
        run_dir=Path(__file__).resolve().parent,
    )
