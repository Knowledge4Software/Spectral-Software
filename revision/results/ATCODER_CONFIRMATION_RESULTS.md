# AtCoder F1 confirmation complete: 2026-11-17

**All six jobs are verified complete; all 156 integrity checks passed.** All stopped normally on validation patience, with no final-test evaluation. None needs continuation or replacement. The final classification-only/123 upload selected epoch22 and finished after32 epochs, taking2.24 recorded hours including setup.

| Seed | Full objective F1 | Full AUROC | Classification-only F1 | Classification-only AUROC |
| --- | ---: | ---: | ---: | ---: |
| 42 | 66.83% | 0.5065 | 81.70% | 0.8780 |
| 123 | 66.73% | 0.4907 | 82.25% | 0.8866 |
| 231 | 66.72% | 0.5138 | 81.72% | 0.8835 |

Mean validation F1 +/- sample SD: **full 66.76% +/- 0.06 points; classification-only 81.89% +/- 0.31 points**. The mean matched difference is +15.13 percentage points for classification-only. SD is descriptive, not a confidence interval or a significance test.

## Interpretation

The full objective shows near-chance ranking in all three submitted runs. Its approximately66.7% F1 is essentially the all-positive baseline on balanced validation. Classification-only learns useful discrimination in all three seeds with the same spectral architecture, dataset and training budget. In this matched confirmation, the full auxiliary-loss configuration is unreliable and removing auxiliary losses improves validation performance consistently. This does not identify an individual problematic term or establish universal superiority across datasets.

The results do not justify silently presenting classification-only as the unchanged full method. It is a documented ablation/variant. They also do not establish a spectral-versus-mean/max advantage: the CodeNet readout comparison remains separate and showed no consistent spectral superiority.

The older successful full/seed42 result remains preserved alongside this failed rerun. Selection metric alone does not alter the gradient objective, so it is not a sufficient explanation for the differing training trajectories. No declared source/model/loss/data mismatch was found; the cause of cross-run instability remains unresolved. Do not discard failed runs or repeat until a favorable outcome appears.

These are validation-selected results. Final tests remain off while design and remaining baseline/transfer choices are unresolved. No held-out performance or submission-readiness claim follows automatically.

## Verification and storage

All six matched the registered F1-selection protocol, full/classification-only loss weights, model configuration, source hash, data identity and fresh lineage. Current/best weights loaded strictly and were finite, as were optimizer/scaler/history. Confusion metrics, best epochs/thresholds, sequential histories, 547 updates per epoch, normal patience/cursors, RNG and benchmarks were consistent. Independent AUROC/AP and threshold-search reproduction requires saved per-pair predictions; these training ZIPs do not contain those exports.

The full audit is `../outputs/revesion/06_atcoder_confirmation_v1/_audit/confirmation_complete_audit_2026-11-17.json`. `validation_summary.csv` holds all six rows. The previous five-run audit/review is retained. The final ZIP is renamed `atcoder_classification_only_seed123_f1_confirmation_v1_complete.zip` with its SHA256 preserved. All original outcome files remain unchanged.

This batch is complete. No training rerun or continuation is requested. The CPU small-report collector is optional for packaging; local checkpoint review already covers all six submitted ZIPs. Remaining scientific work includes resolving the objective/stability interpretation, completing comparator/transfer evidence and freezing choices before final tests.
