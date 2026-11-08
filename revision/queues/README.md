# Queues and provenance

Notebook paths in these CSVs are relative to the `revision/` directory.

- [LAUNCH_24.md](LAUNCH_24.md): eight jobs without submitted results as of
  2026-11-09; verified completions and resumable pauses are excluded.
- `launch_24.csv`: slot/account allocation for those eight jobs. This retains
  the original roster, including costly Qwen/CDLH candidates; it is not an
  instruction to launch every remaining job immediately. See the
  [current evidence and priorities](../results/BATCH_2026-11-09.md).
- `stage1_run_queue.csv`: all 29 seed-42 jobs, including the completed pilot.
- `notebook_manifest.csv`: all maintained notebook paths and byte checksums.
- `stage1_notebook_manifest.csv`: the 29 jobs plus the validation collector.
- `stage1_protocol.json`: frozen scientific settings and dataset checksums.
- `focused_transfer_plan.csv`: planned transfer quotas; run after selection.
- `transfer_plan.csv`: historical full matrix; not the required first-stage queue.
- `path_migration.csv`: old and current project-relative paths; notebook bytes
  are preserved by the folder move.

The launch allocation is a planning aid. Update [results/run_progress.csv](../results/run_progress.csv)
as outputs arrive. Machine validation uses actual `status.json`, not this manual
tracker. A model that reaches the session guard needs continuation before ranking.
