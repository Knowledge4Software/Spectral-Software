# Stage 3: selected seeds 123 and 231

This folder is a library of candidates, not a queue to run in full.

Select comparisons using completed seed-42 validation results and the frozen
[protocol](../00_plan/EXPERIMENT_PROTOCOL.md). Then run only their matching seed
123 and 231 files. Reuse seed 42 when data/settings/source are unchanged.

The core follow-up adds 28 runs: the six central CodeNet readout/objective
conditions, full models on AtCoder/BigCloneBench, and the top two eligible
external baselines per dataset, each with two extra seeds. Extra loss or
cross-benchmark sibling replications need their own ledger entries.

Preserve dataset versions, physical batch sizes, learning rates and pretrained
commits across seeds. `MODEL_REVISION` comes from the first run's token cache.
