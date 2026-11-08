# Continue the three CodeNet text runs from their saved outputs

The first small report archive was reviewed on 2026-11-08. All three jobs paused
on `incomplete_session_budget`; none is a completed baseline yet. The reports
pass their manifest, identity, settings, cursor and counts-based metric checks.
Weights and optimizer tensors were not supplied in the small ZIP and were not
independently audited.

| Original notebook | Validated epochs | Saved position | Best validation Accuracy / F1 |
| --- | ---: | --- | --- |
| [CodeBERT](../01_baselines_seed42/codenet/codenet_codebert_seed42.ipynb) | 6 | Epoch 7, batch 896; 3,320 optimizer steps | 79.42% / 80.54% |
| [C4-adapted](../01_baselines_seed42/codenet/codenet_c4_adapted_seed42.ipynb) | 6 | Epoch 7, batch 8,608; 3,802 optimizer steps | 92.66% / 92.35% |
| [Qwen3-LoRA](../01_baselines_seed42/codenet/codenet_qwen3_lora_seed42.ipynb) | 0 | Epoch 1 training finished; 544 optimizer steps, validation pending | Not available |

These are provisional validation scores and receive no completed screening
rank. C4's current Accuracy is 12.75 percentage points above SPECTRA/full's
79.91%; the result must be retained and investigated through the planned fair
comparisons. It is not a final test result. The C4 adaptation uses training
problem IDs for its task-contrastive objective; disclose this supervision and
do not present it as an exact original-paper reproduction.

## Inputs and configuration

1. Open each original notebook above in a fresh Kaggle session. Enable GPU and
   Internet, and retain the supported Hugging Face dependency setup.
2. Attach the same `codenet_revision_common_support` dataset and that method's
   previous **full saved training output**. Do not attach only the small report
   ZIP: it contains no checkpoint or token arrays. The full output can stay on
   Kaggle and does not need to be downloaded to your laptop.
3. Keep the original dataset, method, seed, batch size, learning rate, optimizer,
   losses and stopping settings. Add this snippet at the **end of the configuration
   cell**, after its existing `RESUME_CHECKPOINT`, `RESUME_TOKENS_DIR` and
   `MODEL_REVISION` assignments:

```python
from pathlib import Path
import json

previous = list(Path("/kaggle/input").rglob(
    f"revision_{DATASET}_{METHOD}_seed{SEED}/progress.pt"
))
assert len(previous) == 1, previous
RESUME_CHECKPOINT = str(previous[0])
RESUME_TOKENS_DIR = str(previous[0].parent / "tokens")

MODEL_REVISION = {
    "codebert": "3b0952feddeffad0063f274080e3c23d75e7eb39",
    "c4_adapted": "3b0952feddeffad0063f274080e3c23d75e7eb39",
    "qwen3_lora": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}[METHOD]

for name in ("tokens.json", "tokens.npy", "masks.npy"):
    assert (Path(RESUME_TOKENS_DIR) / name).is_file(), name
cache = json.loads((Path(RESUME_TOKENS_DIR) / "tokens.json").read_text())
assert cache["complete"] and cache["revision"] == MODEL_REVISION, cache

FINAL_TEST = False
ALLOW_NUMERICAL_UPGRADE = False
```

CodeBERT and C4 must each use their own token cache. C4's whitespace-normalized
cache has a different identity even though both use the same pretrained model.

4. Save & Run All. Preserve every saved output version. On later continuations,
   attach the latest output and remove older inputs for that same method so the
   checkpoint search finds exactly one match. Never restart from scratch merely
   because the previous session reached its time budget.
5. Collect another small report ZIP after the next session. Once a job reports
   `complete=true`, stop continuing it; final tests stay off until choices are frozen.

The training identity remains unchanged. The existing 10-hour session setting
and 45-minute output reserve stay below the 12-hour session cap. Multiple saved
sessions may be required; the 100-epoch ceiling is not a requirement to run 100.

## Qwen timing

Qwen spent about 9h11 in recorded training/validation and completed the 544
training updates of epoch 1. No validation result was saved. On continuation,
the current runtime skips the completed training batches and retries validation;
it does not repeat epoch-1 training. Seeing epoch 1 in that validation log is
expected.

This is a substantial T4 cost. Its first full training epoch nearly filled a
session, so convergence under the existing minimum/early-stopping policy will
require many hours across multiple continuations. A single additional session
is not a promise of completion. Obtain the first validation result and assess
the measured cost before launching repeated Qwen seeds. Do not silently change
the sequence length, batch size, precision or architecture on this checkpoint.

CDLH also remains incomplete; its separate continuation instructions are in the
local batch report `C:\PyProjects\spectrals\outputs\revesion\01\RESULTS_REVIEW.md`.
All six completed CodeNet runs, including SPECTRA/full, should be reused.
