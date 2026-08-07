"""Set the shared 250k/20k cross-dataset budget in SPECTRA notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "kaggle" / "experiments" / "02_cross_dataset" / "train_on_codexglue.ipynb",
    ROOT / "kaggle" / "experiments" / "02_cross_dataset" / "01_spectra_siam_train_on_codexglue.ipynb",
)
PRESET = '''# Cross-dataset transfer budget shared with ASTNN, RtvNN, and DeepSim.
RUN_PRESETS["transfer_250k"] = {
    **RUN_PRESETS["comparison_50k"],
    "max_train_pairs": 250_000,
    "max_valid_pairs": 20_000,
    "max_test_pairs": 20_000,
}

'''


def main() -> None:
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        config = "".join(notebook["cells"][1]["source"])
        config = config.replace('RUN_PROFILE = "comparison_50k"', 'RUN_PROFILE = "transfer_250k"', 1)
        anchor = '# Final paper protocol: every graph-evaluable pair in each official split.\n'
        if anchor not in config:
            raise RuntimeError(f"Preset insertion point missing: {path}")
        config = config.replace(anchor, PRESET + anchor, 1)
        config = config.replace("TRANSFER_TRAIN_CAP = 50_000", "TRANSFER_TRAIN_CAP = 250_000", 1)
        config = config.replace("TRANSFER_VALID_CAP = 10_000", "TRANSFER_VALID_CAP = 20_000", 1)
        config = config.replace("TRANSFER_TEST_CAP = 10_000", "TRANSFER_TEST_CAP = 20_000", 1)
        notebook["cells"][1]["source"] = config.splitlines(keepends=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("updated", path)


if __name__ == "__main__":
    main()
