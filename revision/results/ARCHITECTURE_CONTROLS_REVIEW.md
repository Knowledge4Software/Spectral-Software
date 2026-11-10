# Additional CodeNet graph/descriptor controls: seed 42

**All five additional runs are complete and passed all 180 checks (the original four passed 144; density/heat passed another 36).** Their ZIPs were renamed; every archive SHA-256 is preserved. No continuation or rerun is needed for these five results.

The control tracker now has **10 of 20 controls verified complete, 10 without submitted results**. Including the existing full reference, 11 of 21 CodeNet seed-42 arms are complete. The latest 19-job independent roster now has five verified completions and 14 jobs without submitted outputs (10 CodeNet controls plus four AtCoder/BigCloneBench full repeats). Unreported jobs may already be running; check Kaggle before launching duplicates.

| Condition | Validation Accuracy | Validation F1 | Epochs / best | Parameters | Recorded hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference full | 79.67% | 80.22% | 39 / 29 | 2,977,383 | 2.70 |
| projected | 78.56% | 78.80% | 51 / 41 | 2,846,308 | 3.55 |
| projected_ast | 78.80% | 79.59% | 35 / 25 | 2,846,308 | 2.67 |
| projected_ddg | 56.00% | 44.18% | 11 / 1 | 2,846,308 | 0.87 |
| energy_only | 79.15% | 79.30% | 50 / 40 | 2,910,219 | 3.61 |
| density_heat | 76.99% | 77.61% | 69 / 59 | 2,599,329 | 4.48 |

Accuracy is the prespecified primary metric; these are validation results from seed 42, not final-test performance. Recorded time includes setup and training/validation, excluding final benchmark/export.

## Integrity and protocol

All runs use the same input-v2 CodeNet support: 69538 training, 14896 validation and 14906 test pairs; validation contains 7448 positives and 7448 negatives. The complete archived scientific source set matches the approved numerical-recovery hash `bc9d583ca609c7929cf4373737e07a53b1bfdc79e14a23672e40cb3d8d02e00b`. Only the documented numerical amendment differs from the full reference runtime `8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26`; its finite AMP path and optimizer updates are unchanged. Original implementation hashes are retained.

Model configurations and loss weights match each declared arm; no scientific settings were changed during this review. Current/best weights load strictly, and weights, optimizer, scaler and histories are finite. Checkpoint identities, update counts, RNG state, validation support, counts-based metrics, best epochs/thresholds and patience counters agree. All five stopped on the declared validation patience. Tests were not released.

Benchmarks completed on the same T4/Torch/batch settings and 256-pair support as the full reference. Validation needed no FP32 prediction retry in these five runs. All training FP32 fallback groups recovered: nine for projected_ddg, one for projected, none for the other three. There is no fatal numerical artifact. This validates recorded execution; it does not establish statistical superiority or guarantee perfect numerical conditioning.

## What the scores mean

The full reference has 79.67% Accuracy. Projected AST+DDG has 78.56%, projected AST 78.80%, and energy-only 79.15%. The differences are approximately 1.11, 0.87 and 0.52 percentage points below full, respectively. These are single-seed validation point differences; paired uncertainty cannot be reconstructed from confusion counts alone. Other completed simple-readout controls still match or exceed full, so these results do not overturn the earlier attribution caveat.

**Projected DDG is weak: 56.00% Accuracy / 44.18% F1, best epoch 1, stopping after epoch 11.** The validation set still has both classes. Its AUROC is 0.5754 at the selected epoch and does not improve over the next ten validations. Its low score is not a missing-epoch count or session-budget stop. It correctly uses `prior_relation_indices=(2,)` for the projected adjacency while the shared encoder continues to use AST+DDG relations `(0,2)`; it is not a wholly DDG-only encoder.

The checkpoint/configuration checks found no mismatch explaining that score. Recovered training overflows are logged, including nine FP32 fallback groups; therefore do not call this proof of ideal numerical conditioning. DDG coverage, graph structure and conditioning need separate measured diagnostics before claiming a cause. A low score alone does not establish a dataset bug or justify discarding/replacing the run. Keep this unfavorable result and qualify DDG-related claims.

All five graph/descriptor controls are now complete. Density/heat-only has 76.99% Accuracy / 77.61% F1, with 69 validated epochs and best epoch 59. It took 4.48h including setup/training/validation, had 18 recovered same-group AMP retries, and needed no FP32 training fallback or validation retry. It stopped normally on patience. Its primary score is 2.68 percentage points below full; energy-only is 0.52 points below full. These are exploratory single-seed component comparisons, not evidence of statistical significance or equal capacity: trainable counts differ (2,599,329 for density/heat-only, 2,910,219 for energy-only, 2,977,383 for full). All ten additional loss/weight controls still have no supplied output. These comparisons will clarify descriptor components and auxiliary losses, but individual-weight tuning and broader interactions are not exhausted by this single-seed study.

## Files and remaining work

- [Current independent queue](../queues/RUN_NEXT_INDEPENDENT_BATCH.md): the five verified jobs are excluded from fresh launches.
- [Original six-cell core review](CORE_CONTROLS_REVIEW.md) and [core repeat progress](CORE_REPEATS_PROGRESS.md): all 12 core repeats are now verified complete, including the successful seed-123 fresh restart. Reuse those results.
- Local evidence: `_audit/architecture_controls_additions_audit.json` and `_audit/architecture_controls_rename_manifest.json`; the per-upload density/heat audit is `_audit/density_heat_seed42_audit.json`; original five-control core evidence remains untouched.
- `control_progress.csv` has 10 complete and 10 unreported controls. The baseline and selected-followup rosters are separate.

No GPU training, new inference scores or final-test evaluation was performed during this audit. AUROC/AP were checked for finiteness/range but not independently reproduced without validation predictions.
