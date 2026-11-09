# Stage 3: selected seeds 123 and 231

This folder is a library of candidates, not a queue to run in full.

**Current: [8 core repeats complete, 1 validation failure, 3 without submitted output](../results/CORE_REPEATS_PROGRESS.md).**
Use the [four pending notebooks](../queues/RUN_NEXT_CORE_REPEATS.md) in
`codenet/validation_recovery/`. The failed classification-only/123 job continues
its own checkpoint; the three fresh jobs need only the existing CodeNet data.
The original twelve notebooks remain frozen. Reuse completed results; do not
duplicate existing live jobs. Model/loss/training settings and final-test hold
remain unchanged. These twelve jobs are part of the 28-run follow-up below.

Select comparisons using completed seed-42 validation results and the frozen
[protocol](../00_plan/EXPERIMENT_PROTOCOL.md). Then run only their matching seed
123 and 231 files. Reuse seed 42 when data/settings/source are unchanged.

The core follow-up adds 28 runs: the six central CodeNet readout/objective
conditions, full models on AtCoder/BigCloneBench, and the top two eligible
external baselines per dataset, each with two extra seeds. Extra loss or
cross-benchmark sibling replications need their own ledger entries.

Preserve dataset versions, physical batch sizes, learning rates and pretrained
commits across seeds. `MODEL_REVISION` comes from the first run's token cache.
