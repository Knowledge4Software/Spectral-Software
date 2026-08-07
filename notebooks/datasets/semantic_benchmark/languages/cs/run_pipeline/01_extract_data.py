from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.pipeline_cli import run_dataset_section

if __name__ == "__main__":
    run_dataset_section("semantic", "cs", "prepare", Path(__file__).resolve().parent)
