from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import run_bcb_spectral_feature_extraction


if __name__ == "__main__":
    run_bcb_spectral_feature_extraction("4")
