# Analysis and final evaluation

- `validation_screening.ipynb`: CPU collector, usable during stage 1. Attach
  saved training notebook outputs and optionally `queues/stage1_run_queue.csv`.
  No GPU, Internet, training dataset, or model download is needed. Download only
  `/kaggle/working/revision_small_reports.zip`: validation status, available
  benchmarks/support reports, summary CSVs and a provenance manifest. It never
  opens or copies checkpoint weights, token arrays or final tests. Incomplete
  runs are listed but not ranked. Preserve the full saved outputs on Kaggle;
  the small ZIP supports summary review, not an independent weight-integrity
  check. See [the download guide](../00_plan/SMALL_REPORT_DOWNLOADS.md).
- `validation_spectral_diagnostics.ipynb`: use a SPECTRA checkpoint and its exact
  dataset for spectral/slot/graph-size diagnostics on validation.
- `paired_problem_uncertainty.ipynb`: later, after final tests and repeated
  matched comparisons. Paths share programs/problems; do not treat them as
  independent replicates. BigCloneBench has no authoritative problem IDs for
  this analysis.

Final tests use the original training notebooks with a completed checkpoint and
`FINAL_TEST=True`, after every model/protocol choice is frozen. This is separate
from the validation collector. Keep the test predictions and selected thresholds.
