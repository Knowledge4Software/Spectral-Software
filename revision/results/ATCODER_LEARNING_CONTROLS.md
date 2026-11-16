# AtCoder diagnostic controls v2: 2026-11-16

Completed locally in **284.4 seconds (4m44s)**, below the ten-minute limit. These add two healthy-seed42 objective controls and a full-precision seed231/full control to the retained v1 diagnostic. All used the identical 64 training/256 validation pairs, 60 updates, architecture, learning rate and effective batch128. Initial weights match within seed42 and between seed231 FP32 here and its original AMP v1 counterpart. Main checkpoints, frozen model/data/training sources and all original outcomes are preserved.

| Seed | Objective | Precision | Train F1 | Train AUROC | Train BCE | Small validation AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 42 | full | AMP | 85.25% | 0.9648 | 0.3360 | 0.4502 |
| 42 | classification_only | AMP | 81.97% | 0.9023 | 0.4851 | 0.4680 |
| 231 | full | FP32 | 69.57% | 0.7646 | 0.5949 | 0.5089 |

## Revised interpretation

- Objective effects depend on initialization: at healthy seed42, full learned the tiny sample better than classification-only, reversing the objective ordering observed at seeds123/231 in v1. Do not claim auxiliary losses are universally harmful or that removing them is a confirmed repair.
- Healthy seed42 also failed the strict AUROC>=0.95 AND BCE<0.2 memorization criterion within 60 updates. Full/42 reached AUROC0.9648 but BCE0.3360. The fixed tiny budget is therefore insufficient to treat criterion failure as an explanation of the original seed-specific full-data failures. The criterion was not relaxed after seeing outcomes.
- Seed231/full in FP32 reached AUROC0.7646/BCE0.5949 versus AMP0.7637/0.6024 in v1. Switching training precision alone did not rescue this tiny case. Longer-training precision effects remain untested.
- No numerical recovery or fatal errors occurred. All three cases completed 60 updates and their prediction-derived metrics, support hashes and initial-weight matching independently reproduced.

## Next gate: Kaggle full-data confirmation

The remaining hypothesis needs full-data evidence and F1-based checkpoint selection applied equally to both objectives. Six fresh matched jobs (full/classification-only x seeds42/123/231) are prepared in `revision/06_atcoder_confirmation_v1/`. They keep AMP and all other original settings fixed. They use the existing AtCoder v2 dataset, require no old checkpoints, and retain all weak outcomes. No fresh six-job training was launched locally.

The notebook budget is 100 epochs maximum, minimum10, patience10, with F1 epoch/threshold selection. Each session has 10h including45min reserve. Approximate T4 planning range1-6h on early stopping, about7.7h extrapolated cap budget; continuation may be required. These exceed the user's ten-minute local limit and are the Kaggle handoff.

These diagnostics are not main benchmark improvements. Existing CodeNet F1 evidence still does not establish spectral superiority, and original AtCoder failures remain part of the report. Full-data objective confirmation also does not replace missing external baselines, controlled transfer or final-test uncertainty.
