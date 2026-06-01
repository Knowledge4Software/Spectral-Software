import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

# Project Root (Spectral-Software folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# External Outputs (at the same level as Spectral-Software)
# This fulfills the requirement of being "outside" the project for weight/VSCode performance.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "outputs"

# Allow override via .env
OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))

# Subdirectories
JAVA_DIR = OUTPUT_ROOT / "java_files"
CPG_DIR = OUTPUT_ROOT / "cpg"
DOT_DIR = OUTPUT_ROOT / "dot"
RAW_FEATURES_DIR = OUTPUT_ROOT / "dataset_features"
CLEAN_GRAPHS_DIR = OUTPUT_ROOT / "clean_graphs"
SPECTRAL_FEATURES_DIR = OUTPUT_ROOT / "spectral_features"
MODELS_DIR = OUTPUT_ROOT / "models"
TIMING_STATS_FILE = OUTPUT_ROOT / "timing_stats.json"

def get_paths():
    """Returns a dictionary of all relevant paths."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "java_dir": str(JAVA_DIR),
        "cpg_dir": str(CPG_DIR),
        "dot_dir": str(DOT_DIR),
        "raw_features_dir": str(RAW_FEATURES_DIR),
        "clean_graphs_dir": str(CLEAN_GRAPHS_DIR),
        "spectral_features_dir": str(SPECTRAL_FEATURES_DIR),
        "models_dir": str(MODELS_DIR),
        "timing_stats": str(TIMING_STATS_FILE),
    }

def ensure_dirs():
    """Create all necessary output directories if they don't exist."""
    dirs = [
        OUTPUT_ROOT, JAVA_DIR, CPG_DIR, DOT_DIR, 
        RAW_FEATURES_DIR, CLEAN_GRAPHS_DIR, 
        SPECTRAL_FEATURES_DIR, MODELS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Output Root:  {OUTPUT_ROOT}")
    ensure_dirs()
