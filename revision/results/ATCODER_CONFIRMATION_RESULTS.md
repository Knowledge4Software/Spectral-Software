# AtCoder F1 confirmation: five runs received 2026-11-17

**Five complete runs passed all 130 checkpoint/source/configuration/recorded-metric checks.** All stopped normally on the prescribed validation patience; none needs continuation. Classification-only/seed123 is the only missing output. No final test has been evaluated.

| Seed | Full F1 | Full AUROC | Classification-only F1 | Classification-only AUROC |
| --- | ---: | ---: | ---: | ---: |
| 42 | 66.83% | 0.5065 | 81.70% | 0.8780 |
| 123 | 66.73% | 0.4907 | Not submitted | Not submitted |
| 231 | 66.72% | 0.5138 | 81.72% | 0.8835 |

## Findings

All three full-objective runs have AUROC approximately0.49-0.51 and almost-all-positive decisions. F1 around66.7% is essentially the all-positive baseline on the balanced validation set, not useful clone discrimination.

Classification-only retains the same 2,977,383-parameter spectral architecture and inputs. Seeds42/231 reach F1 approximately81.7%, AUROC0.8780/0.8835. Matched F1 differences are approximately+14.87/+15.01 percentage points. The architecture/input pipeline can therefore support useful learning; the full-loss recipe is unreliable in these submitted runs. This supports the objective-interaction concern, but does not isolate an individual auxiliary loss or establish a universal cause.

The new full/seed42 failed despite the older Accuracy-selected seed42 run learning useful ranking. Both outcomes must be retained. Merely changing the selection metric does not itself alter the gradient objective; it is not a sufficient diagnosis for the failed training trajectory. The audit found no declared model/loss/data/source mismatch. Non-bitwise reproducibility, numerical trajectories or unrecorded environmental factors remain possible explanations; none is established here. Do not rerun and retain only a favorable full result.

No three-seed classification-only mean or uncertainty claim is produced while seed123 is missing. Classification-only is an ablation/variant, not a silently substituted full method. Claims that the full auxiliary objective is necessary or robust are not supported. These results are not final-test performance, and the external baseline/transfer evidence remains separate.

## Integrity and scope

Pinned AtCoder data identity, embedded numerical-recovery runtime, original planned model/loss weights, shared F1 selection, AdamW/batch/stopping settings and fresh lineage matched. Current/best weights strictly load and are finite, as are optimizer/scaler/history. Sequential epochs, confusion-count metrics, F1-selected best epochs/thresholds, 547 optimizer updates per epoch, cursors/patience and benchmark settings are consistent. Independent AUROC/AP and threshold optimization reproduction needs per-pair probabilities; these ZIPs contain training checkpoints/status, not exported validation predictions.

| Job | Completed epochs | Best epoch | Recorded hours |
| --- | ---: | ---: | ---: |
| atcoder_classification_only_seed42_f1_confirmation_v1 | 29 | 19 | 1.90 |
| atcoder_full_seed42_f1_confirmation_v1 | 21 | 11 | 1.32 |
| atcoder_full_seed123_f1_confirmation_v1 | 27 | 17 | 1.82 |
| atcoder_classification_only_seed231_f1_confirmation_v1 | 32 | 22 | 2.16 |
| atcoder_full_seed231_f1_confirmation_v1 | 11 | 1 | 0.85 |

## Next

Supply `atcoder_classification_only_seed123_f1_confirmation_v1` when available. Check the existing Kaggle job before launching a duplicate. Preserve all five completed outputs; do not repeat them for a better score. After the sixth output arrives, finish the matched summary and collect validation probabilities for any subsequent paired analysis. Final tests remain off.

Evidence: `../outputs/revesion/06_atcoder_confirmation_v1/_audit/confirmation_audit_2026-11-17.json`, the rename manifest and `validation_summary.csv`. Original ZIP bytes are preserved. No scientific source or notebook was changed during this review.
