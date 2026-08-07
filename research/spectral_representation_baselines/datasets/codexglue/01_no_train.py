from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.spectral_representation_baselines.runner import launch
launch("codexglue", "no_train")
