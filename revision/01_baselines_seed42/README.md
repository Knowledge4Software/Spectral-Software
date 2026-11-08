# Stage 1: baseline screening, seed 42

Open the notebooks directly: [CodeNet](codenet), [AtCoder](atcoder),
[BigCloneBench](bigclonebench). There are 10 + 10 + 9 = 29 jobs.

Check [the run tracker](../results/run_progress.csv) before launching a job.
CodeNet/full/42 and five CodeNet graph baselines have verified complete outputs.
Reuse them. CodeNet/CDLH/42 must resume its saved checkpoint,
not restart. The remaining jobs are listed in
[the parallel launch list](../queues/LAUNCH_24.md). Jobs with unreported status
may already be running; confirm before starting another copy.

Attach the [exact dataset version](../datasets/README_FA.md) for each folder.
Keep `FINAL_TEST=False`. Only completed runs are ranked. Transformer jobs need
Internet. Preserve checkpoints, status, benchmarks and token caches.

Read [START_HERE_FA.md](../START_HERE_FA.md) for settings and continuation.
