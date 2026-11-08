# Revision documentation

For execution start with [the Persian workspace guide](../START_HERE_FA.md)
and [the current launch list](../queues/LAUNCH_24.md). The CodeNet/full checkpoint
is verified complete; reuse it. Original roadmap dates and checklists are retained below
as planning history, while the protocol and current guide describe execution.

- [Full revision roadmap](REVISION_MASTER_PLAN.md): reviewer concerns, dependencies,
  experiment counts and manuscript work.
- [Persian guide](REVISION_PLAN_FA.md): goals, rationale and planning estimates.
- [First Kaggle run](FIRST_RUN.md): exact input,
  notebook settings, progress and checkpoint continuation.
- [T4 numerical checkpoint recovery](NUMERICAL_RECOVERY.md):
  AMP retry/FP32 fallback and explicit source-version upgrade with lineage.
- [Hugging Face setup recovery](HF_SETUP_RECOVERY.md): supported dependency
  versions and fresh-session instructions for the missing `_commit_hash` error.
- [Small report downloads](SMALL_REPORT_DOWNLOADS.md): review large text-run
  outputs on CPU without downloading multi-gigabyte checkpoints.
- [Text-run continuation](TEXT_RUN_CONTINUATION.md): the reviewed CodeNet
  CodeBERT/C4/Qwen pauses, exact checkpoint/token-cache settings and Qwen timing.
- [Frozen experiment protocol](EXPERIMENT_PROTOCOL.md).
- [Seed-42 run queue](../queues/stage1_run_queue.csv).
- [Reviewer coverage and remaining limitations](REVIEWER_COVERAGE.md).

The progress-display release preserves the scientific training fingerprint.
Large dataset ZIPs, release packets, audit outputs and checkpoints are local
artifacts, excluded from Git. Dataset checksum references remain in the protocol.
Historical paper notebooks are retained separately for provenance; use the
revision queue for new comparisons.

Time estimates are conditional on measured pilot speed and the final experiment
scope. Available concurrent runs reduce compute time but cannot remove model
selection, validation, analysis and manuscript dependencies. No GPU training
result or manuscript acceptance is established by code preparation.
