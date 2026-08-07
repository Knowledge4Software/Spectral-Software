from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.pipeline_section_runner import SectionConfig, run_pss_wasserstein_tuning
from spectral_code.utils.pipeline_cli import run_spectral_tuning_cli


if __name__ == "__main__":
    run_spectral_tuning_cli(
        description="Tune Semantic Benchmark Java spectral metric thresholds.",
        output_folder="outputs/semantic_benchmark/java",
        runner=lambda: run_pss_wasserstein_tuning(
            SectionConfig(dataset="semantic", variant="java", run_dir=Path(__file__).resolve().parent)
        ),
        allow_pair_score_csv=False,
    )
