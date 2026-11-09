# Stage 3: selected seeds 123 and 231

This folder is a library of candidates, not a queue to run in full.

**Current: [all 12 core repeats are verified complete](../results/CORE_REPEATS_PROGRESS.md).**
The seed-123 restart completed after 53 epochs, selecting epoch 43. Its earlier
failed attempt remains historical. [No launch or continuation is needed from these twelve jobs](../queues/RUN_NEXT_CORE_REPEATS.md).
The original notebooks remain frozen; reuse all 12 complete results. Model/loss/training settings and final-test hold remain
unchanged. These twelve jobs are part of the 28-run follow-up below.

Four more prespecified full-model repeats are now [ready independently](../queues/RUN_NEXT_INDEPENDENT_BATCH.md):
AtCoder and BigCloneBench with seeds 123 and 231. Use the linked numerical-fix
copies. Bringing these forward does not select external finalists or release tests.
Their progress is tracked separately in `results/selected_followup_progress.csv`.

Select comparisons using completed seed-42 validation results and the frozen
[protocol](../00_plan/EXPERIMENT_PROTOCOL.md). Then run only their matching seed
123 and 231 files. Reuse seed 42 when data/settings/source are unchanged.

The core follow-up adds 28 runs: the six central CodeNet readout/objective
conditions, full models on AtCoder/BigCloneBench, and the top two eligible
external baselines per dataset, each with two extra seeds. Extra loss or
cross-benchmark sibling replications need their own ledger entries.

Preserve dataset versions, physical batch sizes, learning rates and pretrained
commits across seeds. `MODEL_REVISION` comes from the first run's token cache.
