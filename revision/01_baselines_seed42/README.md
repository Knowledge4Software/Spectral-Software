# Stage 1: baseline screening, seed 42

Open the notebooks directly: [CodeNet](codenet), [AtCoder](atcoder),
[BigCloneBench](bigclonebench). There are 10 + 10 + 9 = 29 jobs.

Check [the run tracker](../results/run_progress.csv) before launching a job.
As of 2026-11-09, 17 outputs are verified complete: six CodeNet, five AtCoder
and six BigCloneBench jobs. Reuse them. Four CodeNet jobs remain paused;
CDLH must resume its saved checkpoint, not restart. Eight jobs have no submitted
results; see [the combined review and priorities](../results/BATCH_2026-11-09.md).
The remaining original-roster jobs are listed in
[the parallel launch list](../queues/LAUNCH_24.md). Jobs with unreported status
may already be running; confirm before starting another copy.

CodeBERT, C4-adapted and Qwen3-LoRA also paused on their session budgets. Their
reports are reviewed, but they are not completed screening results. Follow
[the text continuation guide](../00_plan/TEXT_RUN_CONTINUATION.md) to retain
their checkpoints, pinned model revisions and method-specific token caches.

Attach the [exact dataset version](../datasets/README_FA.md) for each folder.
Keep `FINAL_TEST=False`. Only completed runs are ranked. Transformer jobs need
Internet. Preserve checkpoints, status, benchmarks and token caches.

Read [START_HERE_FA.md](../START_HERE_FA.md) for settings and continuation.
