# Stage 3: selected seeds 123 and 231

This folder is a library of candidates, not a queue to run in full.

**Current: [11 core repeats complete; classification-only/123 restart running](../results/CORE_REPEATS_PROGRESS.md).**
The user confirmed the fixed fresh restart is running on Kaggle. Wait for its
saved output. [No new launch is needed from these twelve jobs](../queues/RUN_NEXT_CORE_REPEATS.md).
The original notebooks remain frozen; reuse all 11 complete results and do not
duplicate the live job. Model/loss/training settings and final-test hold remain
unchanged. These twelve jobs are part of the 28-run follow-up below.

Select comparisons using completed seed-42 validation results and the frozen
[protocol](../00_plan/EXPERIMENT_PROTOCOL.md). Then run only their matching seed
123 and 231 files. Reuse seed 42 when data/settings/source are unchanged.

The core follow-up adds 28 runs: the six central CodeNet readout/objective
conditions, full models on AtCoder/BigCloneBench, and the top two eligible
external baselines per dataset, each with two extra seeds. Extra loss or
cross-benchmark sibling replications need their own ledger entries.

Preserve dataset versions, physical batch sizes, learning rates and pretrained
commits across seeds. `MODEL_REVISION` comes from the first run's token cache.
