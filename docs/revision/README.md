# Revision documentation

- [Full revision roadmap](REVISION_MASTER_PLAN.md): reviewer concerns, dependencies,
  experiment counts and manuscript work.
- [Persian guide](REVISION_PLAN_FA.md): goals, rationale and planning estimates.
- [First Kaggle run](../../experiments/kaggle/revision/FIRST_RUN.md): exact input,
  notebook settings, progress and checkpoint continuation.
- [T4 numerical checkpoint recovery](../../experiments/kaggle/revision/NUMERICAL_RECOVERY.md):
  AMP retry/FP32 fallback and explicit source-version upgrade with lineage.
- [Frozen experiment protocol](../../experiments/kaggle/revision/EXPERIMENT_PROTOCOL.md).
- [Seed-42 run queue](../../experiments/kaggle/revision/stage1_run_queue.csv).
- [Reviewer coverage and remaining limitations](../../experiments/kaggle/revision/REVIEWER_COVERAGE.md).

The progress-display release preserves the scientific training fingerprint.
Large dataset ZIPs, release packets, audit outputs and checkpoints are local
artifacts, excluded from Git. Dataset checksum references remain in the protocol.
Historical paper notebooks are retained separately for provenance; use the
revision queue for new comparisons.

Time estimates are conditional on measured pilot speed and the final experiment
scope. Available concurrent runs reduce compute time but cannot remove model
selection, validation, analysis and manuscript dependencies. No GPU training
result or manuscript acceptance is established by code preparation.
