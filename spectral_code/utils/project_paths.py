import os
from pathlib import Path
from spectral_code.utils.dataset_paths import output_root_for, path_from_env

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            return False

        loaded = False
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded = True
        return loaded


# Load .env if python-dotenv is installed and a file exists.
load_dotenv()

# Project Root (Spectral-Software folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Pipeline artifacts live outside the repository by default so graph structures,
# timing files, reports, and tuning outputs stay in one shared output root.
BCB_CLONE_TYPE = os.getenv("BCB_CLONE_TYPE", "1")
DEFAULT_OUTPUT_BASE = PROJECT_ROOT.parent / "outputs"
DEFAULT_OUTPUT_ROOT = output_root_for("bcb", BCB_CLONE_TYPE)

# Allow override via .env
OUTPUT_ROOT = path_from_env("OUTPUT_DIR", DEFAULT_OUTPUT_ROOT)

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
