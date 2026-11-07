# Queues and provenance

Notebook paths in these CSVs are relative to the `revision/` directory.

- [LAUNCH_24.md](LAUNCH_24.md): clickable list of the next 24 jobs and the four
  queued behind them, using the completion reported on 2026-11-07.
- `launch_24.csv`: slot/account allocation for those 24 independent jobs.
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
