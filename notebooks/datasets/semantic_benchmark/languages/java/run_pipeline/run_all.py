from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.pipeline_cli import run_numbered_pipeline


STAGES = [("01", "01_extract_data.py"), ("02", "02_extract_graphs.py"), ("03", "03_extract_spectral_features.py"), ("04", "04_tune_pss_wasserstein.py")]


if __name__ == "__main__":
    run_numbered_pipeline(STAGES, description="Run all Semantic Benchmark Java pipeline stages.", completion_message="All selected Semantic Benchmark Java pipeline stages completed.", run_dir=Path(__file__).resolve().parent)
