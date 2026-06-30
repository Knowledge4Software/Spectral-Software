from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import merge_bcb_type3_all_artifacts


if __name__ == "__main__":
    merge_bcb_type3_all_artifacts()
