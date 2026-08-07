# Kaggle experiment notebooks

Each V3 benchmark has an independent folder with the same nine baselines and the canonical SPECTRA-Siam method notebook. Attach the matching ZIP from `../outputs/kaggle_datasets/`, import the corresponding notebook, and run all cells.

All notebooks discover their input recursively below `/kaggle/input`; no owner-specific path needs editing. They support both the normal ZIP layout (`codes.jsonl.gz`) and Kaggle's occasionally nested temporary layout (`codes.jsonl/codes.jsonl.gz.tmp`).

The default `RUN_PROFILE = "quick_1h"` is the preliminary, bounded run. Change only that value to `"extended_6_7h"` for the larger protocol; see `RUN_PROFILES.md`.
