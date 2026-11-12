# Queues and provenance

Notebook paths in these CSVs are relative to the `revision/` directory.

- **[Checkpoint review: run next](../05_analysis/continuation/README_FA.md)**:
  four GPU inference jobs, then one CPU collector. `checkpoint_review_notebooks.csv`
  pins the five new notebooks; `checkpoint_review_inputs.csv` maps each of 21
  existing checkpoints to its job. `checkpoint_review_manifest.json` pins their
  bytes and identities. These reporting notebooks are separate from the 193
  frozen original notebooks and the completed independent training roster.

- **[RUN_NEXT_INDEPENDENT_BATCH.md](RUN_NEXT_INDEPENDENT_BATCH.md)**:
  The original 19-job roster has completed evidence for all 19 job identities; BCB/full/231 has two distinct attempts pending provenance. No fresh launches remain. All 20 CodeNet seed42 controls are complete.
  Includes an estimate for every notebook and exact dataset/version links.
  `next_independent_roster.csv` retains all 19; `next_independent_run_queue.csv`
  excludes jobs already reported running/complete. Use the linked numerical-fix
  copies; original frozen notebooks remain preserved.

- [RUN_NOW_INPUT_V2.md](RUN_NOW_INPUT_V2.md) and `input_v2_run_queue.csv`:
  the original eight-job priority batch is complete; zero pending rows.
- [RUN_NEXT_CORE_REPEATS.md](RUN_NEXT_CORE_REPEATS.md) and
  `core_repeat_run_queue.csv`: zero fresh launches.
  `core_repeat_resume_queue.csv`: zero continuations. All 12 repeats are verified
  complete; reuse them. `core_repeat_roster.csv` retains the original 12.
  The four fixed notebook candidates and their source hashes remain in
  `validation_recovery_manifest.csv`; three now have verified original-runtime
  completions and need no rerun. Existing CodeNet data is reused.

- [LAUNCH_24.md](LAUNCH_24.md): remaining baseline jobs without submitted results;
  the completed corrected full runs and resumable pauses are excluded.
- `launch_24.csv`: slot/account allocation for that remaining roster. This retains
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
