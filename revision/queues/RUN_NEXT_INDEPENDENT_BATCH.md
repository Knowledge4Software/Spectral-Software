# Next independent batch: controls and full-model repeats

**8 fresh jobs ready from a 19-job preplanned roster.** All 12 CodeNet core repeats are complete; do not repeat them.

The original roster contains 15 CodeNet seed-42 controls and four full-model repeats on AtCoder/BigCloneBench. Verified completions are excluded from the fresh list below. All were already in the revision plan. The four full repeats are independent of external-finalist selection. The other selected external repeats and transfer study still wait for completed baseline screening. Nothing is selected using test results.

## Exact notebooks and T4 estimates

| Slot | Dataset | Condition | Seed | Hours per notebook | Notebook |
| ---: | --- | --- | ---: | --- | --- |
| 10 | codenet | without_topology_reconstruction | 42 | 2-8 | [codenet_without_topology_reconstruction_seed42_validation_recovery.ipynb](../02_controls_seed42/codenet/validation_recovery/losses/codenet_without_topology_reconstruction_seed42_validation_recovery.ipynb) |
| 12 | codenet | without_graph | 42 | 2-8 | [codenet_without_graph_seed42_validation_recovery.ipynb](../02_controls_seed42/codenet/validation_recovery/losses/codenet_without_graph_seed42_validation_recovery.ipynb) |
| 14 | codenet | auxiliary_half | 42 | 2-8 | [codenet_auxiliary_half_seed42_validation_recovery.ipynb](../02_controls_seed42/codenet/validation_recovery/losses/codenet_auxiliary_half_seed42_validation_recovery.ipynb) |
| 15 | codenet | auxiliary_double | 42 | 2-8 | [codenet_auxiliary_double_seed42_validation_recovery.ipynb](../02_controls_seed42/codenet/validation_recovery/losses/codenet_auxiliary_double_seed42_validation_recovery.ipynb) |
| 16 | atcoder | full | 123 | 2-6 | [atcoder_full_seed123_validation_recovery.ipynb](../03_selected_repeats/atcoder/validation_recovery/atcoder_full_seed123_validation_recovery.ipynb) |
| 17 | atcoder | full | 231 | 2-6 | [atcoder_full_seed231_validation_recovery.ipynb](../03_selected_repeats/atcoder/validation_recovery/atcoder_full_seed231_validation_recovery.ipynb) |
| 18 | bigclonebench | full | 123 | 1-4 | [bigclonebench_full_seed123_validation_recovery.ipynb](../03_selected_repeats/bigclonebench/validation_recovery/bigclonebench_full_seed123_validation_recovery.ipynb) |
| 19 | bigclonebench | full | 231 | 1-4 | [bigclonebench_full_seed231_validation_recovery.ipynb](../03_selected_repeats/bigclonebench/validation_recovery/bigclonebench_full_seed231_validation_recovery.ipynb) |

The 15 CodeNet controls address spectral attribution, learned/projected graph views, descriptor components, loss contributions and coarse auxiliary-weight sensitivity (especially R6.2/R6.5). Full-model repeats measure variability beyond CodeNet; they do not prove superiority or complete baseline comparisons.

## Inputs and settings

Reuse the existing uploads; no dataset rebuild or upload is needed:

- CodeNet controls: `codenet_revision_common_support.zip`.
- AtCoder repeats: `atcoder_revision_common_support_v2.zip`.
- BigCloneBench repeats: `bigclonebench_revision_common_support_v2.zip`.

Attach only the matching dataset. GPU on, Internet off, RESUME_CHECKPOINT=None, ALLOW_NUMERICAL_UPGRADE=False, FINAL_TEST=False. All are fresh jobs. The files include the approved numerical validation fix; use these linked files rather than the frozen originals. Model/data/loss/stopping settings and tqdm are preserved. New output paths end in `_input_v2_validation_fix`.

## How the estimates were obtained

These are planning ranges per notebook on one T4, including normal setup, not measured timings for the 15 new arms. CodeNet's completed reference conditions took about 1.7-6.0h (24-86 epochs), roughly 3.8-4.1 minutes/epoch. Using the slowest observed core rate, 100 epochs plus setup and 15% planning margin is about 8.0h. The new graph/loss variants may behave differently; the common range deliberately avoids inventing an arm-specific speedup.

AtCoder full/42 took 4.00h for 60 epochs, with 3.96h training/validation. Its extrapolated 100-epoch estimate with setup and 15% margin is 7.7h. BigCloneBench full/42 took 1.44h for 24 epochs (1.43h training/validation); the comparable cap estimate is 6.9h. The 2-6h and 1-4h ranges assume earlier validation stopping, not a promise for seeds 123/231.

Each session still has a 10h internal budget with a 45-minute reserve (work stops around 9h15). Keep the 100-epoch ceiling, minimum 10 and patience 10. Numerical retries or slower hardware can require another session; preserve progress rather than shortening the schedule. Each job's own runtime forecast becomes more informative after its first completed epochs.

The 8 listed jobs require 8 GPU-job slots; account for any other active jobs within your 24-slot limit. Check existing Kaggle jobs before starting duplicates. Allow roughly 8-10h for the parallel batch if no job needs continuation; this is not an estimate for completing the entire revision.

## Modern baselines are still pending

CodeBERT/C4 remain important review gaps. The old CodeNet jobs need their own saved checkpoints/token caches and original archived notebooks; see [continuation instructions](../00_plan/TEXT_RUN_CONTINUATION.md). They completed only about six epochs in the first approximately 9h work session, so further 10h sessions may be needed. Starting AtCoder/BigCloneBench text baselines also needs runtime pilots; do not label them as 2-8h graph jobs.

Qwen nearly filled a session with epoch-1 training alone, and CDLH also paused early. They remain in the roster but are outside this short batch; no convergence time is promised. Do not launch extra seeds or transfer jobs for unfinished external methods. Final tests remain off.

See [protocol](../00_plan/EXPERIMENT_PROTOCOL.md) and [reviewer coverage](../00_plan/REVIEWER_COVERAGE.md). Check existing Kaggle jobs before launching a duplicate. Generating this queue does not mean any job has started.
