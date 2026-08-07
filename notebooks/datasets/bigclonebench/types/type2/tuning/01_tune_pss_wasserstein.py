from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import run_bcb_metric_tuning
from spectral_code.utils.pipeline_cli import run_spectral_tuning_cli


if __name__ == "__main__":
    run_spectral_tuning_cli(
        description="Tune Type-2 BCB spectral metric thresholds.",
        output_folder="outputs/bcb/type2",
        runner=lambda: run_bcb_metric_tuning("2", metrics=["pss"], stage_name="04_tune_pss_wasserstein"),
    )
