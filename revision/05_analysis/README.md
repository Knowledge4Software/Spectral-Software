# Analysis and final evaluation

- `validation_screening.ipynb`: CPU collector, usable during stage 1. Attach
  saved outputs and optionally `queues/stage1_run_queue.csv`. It reads validation
  status only; incomplete runs are listed but not ranked.
- `validation_spectral_diagnostics.ipynb`: use a SPECTRA checkpoint and its exact
  dataset for spectral/slot/graph-size diagnostics on validation.
- `paired_problem_uncertainty.ipynb`: later, after final tests and repeated
  matched comparisons. Paths share programs/problems; do not treat them as
  independent replicates. BigCloneBench has no authoritative problem IDs for
  this analysis.

Final tests use the original training notebooks with a completed checkpoint and
`FINAL_TEST=True`, after every model/protocol choice is frozen. This is separate
from the validation collector. Keep the test predictions and selected thresholds.
