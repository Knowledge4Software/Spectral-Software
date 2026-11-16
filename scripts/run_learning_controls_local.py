"""Bounded local AtCoder memorization diagnostic; uses no old checkpoints."""
from pathlib import Path
import time

STARTED = time.monotonic()

import json
import torch
from spectral_code.revision.learning_controls import run_controls
from spectral_code.revision.progress import install_progress

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    install_progress()
    result = run_controls(root.parent / "data/working/revision/atcoder_revision_common_support_v2",
                       root.parent / "outputs/revesion/05/atcoder_learning_controls_v2_local",
                       budget_seconds=540, started=STARTED)
    print(json.dumps({k:v for k,v in result.items() if k != "cases"}, indent=2), flush=True)
