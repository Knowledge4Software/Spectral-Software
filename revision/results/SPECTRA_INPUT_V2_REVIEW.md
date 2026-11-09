# SPECTRA input-v2 results review — 2026-11-09

**All three corrected-input SPECTRA full runs are complete and verified. All 102 checks passed. None needs continuation or another full-model run.**

The three old full-model ZIPs were moved out of the active folder into `_archive/input_v1/`. The new ZIPs were renamed with dataset, seed, input version and completion status. All six SHA-256 hashes were preserved; no historical result was deleted.

These are seed-42 validation results with final tests off. CodeNet/AtCoder use Accuracy; BigCloneBench uses F1.

| Dataset | Primary metric | Old input v1 | Current input v2 | Change, percentage points | Epochs / best epoch | Recorded hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| codenet | Accuracy | 79.91% | **79.67%** | -0.24 | 39 / 29 | 2.70 |
| atcoder | Accuracy | 76.10% | **78.34%** | +2.24 | 60 / 50 | 4.00 |
| bigclonebench | F1 | 67.92% | **97.15%** | +29.23 | 24 / 14 | 1.44 |

## Interpretation

BigCloneBench F1 improved from 67.92% to 97.15%, a 29.23-point increase after preserving Java callee features. This is consistent with the diagnosed information loss being consequential. It is one training seed and does not establish statistical significance or final-test performance. The leading completed BigCloneBench baseline, FA-AST-GGNN, still has 98.27% F1, about 1.12 points higher.

AtCoder Accuracy improved by 2.24 points to 78.34%, but RtvNN remains ahead at 82.91%. CodeNet Accuracy decreased by 0.24 points to 79.67%; DeepSim remains ahead at 81.33%. The fix should not be described as a universal performance improvement or overall SPECTRA superiority.

BigCloneBench still has only 127 unique validation programs and 128 test programs. The lexical-input repair does not enlarge this independent support. Generalization claims must retain that limitation.

## Integrity

All three checkpoint/source fingerprints match input v2 (`8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26`), and the actual embedded runtime contains the corrected callee-feature code. Dataset identities match the previously verified unchanged bundles. Model configuration, all losses, optimizer settings, validation selection and stopping budgets match the frozen amended protocol.

Current and best weights load strictly into the planned 2,977,383-parameter model and are finite, as are optimizer/scaler/history values. Confusion-matrix metrics, best epochs, thresholds, RNG state, cursor/update counts and patience counters are consistent. Every run stopped after ten validations without improvement, with test evaluation off.

AMP replay events were 11 for CodeNet, 18 for AtCoder and six for BigCloneBench; all were recognized same-group retries. No fatal numerical artifact or unrecovered FP32 fallback was found. Benchmarks passed support and consistency checks; their timings exclude data loading, transfers, optimizer steps and checkpoint I/O.

AUROC/AP cannot be independently reproduced without validation predictions. This audit did not retrain models or run held-out test evaluation.

## Current completed comparisons

### codenet — Accuracy

| Rank | Model | Accuracy | F1 | AUROC |
| ---: | --- | ---: | ---: | ---: |
| 1 | DeepSim | 81.33% | 81.91% | 0.8927 |
| 2 | FA-AST-GMN | 79.81% | 79.94% | 0.8766 |
| 3 | SPECTRA full (input v2) | 79.67% | 80.22% | 0.8795 |
| 4 | FA-AST-GGNN | 79.16% | 79.36% | 0.8750 |
| 5 | RtvNN | 78.95% | 79.85% | 0.8730 |
| 6 | ASTNN | 71.23% | 74.24% | 0.7745 |

### atcoder — Accuracy

| Rank | Model | Accuracy | F1 | AUROC |
| ---: | --- | ---: | ---: | ---: |
| 1 | RtvNN | 82.91% | 83.26% | 0.9056 |
| 2 | ASTNN | 81.29% | 82.20% | 0.8809 |
| 3 | FA-AST-GGNN | 80.26% | 80.72% | 0.8762 |
| 4 | SPECTRA full (input v2) | 78.34% | 79.36% | 0.8561 |
| 5 | DeepSim | 71.27% | 72.82% | 0.7721 |

### bigclonebench — F1

| Rank | Model | Accuracy | F1 | AUROC |
| ---: | --- | ---: | ---: | ---: |
| 1 | FA-AST-GGNN | 99.55% | 98.27% | 0.9987 |
| 2 | DeepSim | 99.44% | 97.83% | 0.9993 |
| 3 | FA-AST-GMN | 99.42% | 97.77% | 0.9944 |
| 4 | SPECTRA full (input v2) | 99.26% | 97.15% | 0.9870 |
| 5 | RtvNN | 99.02% | 96.19% | 0.9910 |
| 6 | ASTNN | 95.41% | 81.56% | 0.9546 |

## Next jobs

Reuse the three current full models and the 14 completed external baselines. The baseline roster has 17 verified completions, four paused CodeNet runs and eight without submitted results. The paused C4-adapted CodeNet result remains provisional at 92.66% Accuracy; it is not silently discarded or included in completed ranking.

The next priority is the five CodeNet controls: mean/max, attention, full classification-only, mean/max classification-only and attention classification-only. All use the same corrected input-v2 representation. See [the current run list](../queues/RUN_NOW_INPUT_V2.md). If already running, preserve those outputs rather than launching duplicates. Remaining expensive baselines can stay deferred for now.

## Evidence

- Current ZIPs: `codenet_full_seed42_input_v2_complete.zip`, `atcoder_full_seed42_input_v2_complete.zip`, `bigclonebench_full_seed42_input_v2_complete.zip`.
- `_audit/spectra_input_v2_audit.json`: all 102 new checks.
- `_audit/input_v2_archive_plan.json`: old/new names, archive destinations and preserved hashes.
- `_audit/audit.json`: 21 current reported runs plus three separately retained historical full-model records.
- `_audit/validation_results.csv`: current scores and completed-only ranks; old full-model scores are excluded.
- `_archive/input_v1/`: the three superseded SPECTRA ZIPs. Earlier audit snapshots and `INPUT_V1_RESULTS_REVIEW.md` are retained.

Recorded hours are saved setup plus training/validation time; they exclude final benchmarking and Kaggle export overhead. No training code, notebook or dataset changed during this review.
