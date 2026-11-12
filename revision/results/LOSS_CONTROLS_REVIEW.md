# CodeNet loss controls: seven verified loss-ablation runs

**All seven audited seed-42 loss uploads passed 252/252 checks and stopped normally on validation patience.** The control tracker now has all 20 controls complete, including the three additions reviewed on 2026-11-12 below. Final tests remain off.

| Condition | Validation Accuracy | Validation F1 | Change from full, Accuracy points | Epochs / best | Recorded hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full reference | 79.67% | 80.22% | 0.00 | 39 / 29 | 2.70 |
| without_auc_ranking | 79.10% | 79.40% | -0.58 | 43 / 33 | 2.95 |
| without_hard_negative | 80.16% | 80.81% | +0.48 | 46 / 36 | 3.18 |
| without_reconstruction | 79.77% | 79.97% | +0.10 | 41 / 31 | 2.87 |
| without_spectral | 80.12% | 81.10% | +0.44 | 43 / 33 | 3.20 |
| without_spectral_and_hard_negative | 79.90% | 80.04% | +0.23 | 40 / 30 | 2.85 |
| without_spectral_variance | 79.36% | 79.60% | -0.31 | 37 / 27 | 2.59 |
| without_topology_reconstruction | 78.97% | 79.17% | -0.70 | 46 / 36 | 3.53 |

## Interpretation

`without_spectral` removes the spectral auxiliary loss; it retains the spectral readout. Likewise, the joint arm removes spectral and hard-negative auxiliary losses while keeping the other objectives. These are loss-attribution comparisons, not spectral-versus-nonspectral readout comparisons.

Removing the spectral or hard-negative loss slightly improves the primary validation Accuracy in this seed; removing AUC ranking or topology reconstruction lowers it. The topology-reconstruction ablation is 0.70 Accuracy points below full in this seed; this is a validation point difference, not a significance result. The results do not support a claim that every auxiliary term is necessary or individually improves Accuracy. Small single-seed validation differences are not significance tests, and confusion matrices alone cannot reconstruct paired disagreements. Preserve all arms and revise necessity claims accordingly.

## Integrity and limitations

All seven use the original input-v2 training hash `8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26`, identical to the full reference. The entire archived scientific source set was fingerprinted. The later conditional FP32 validation amendment is not needed to relabel these successful original-runtime runs; their raw identities are preserved.

All use the same 69,538/14,896/14,906 CodeNet train/validation/test support, seed 42, Accuracy selection, 100-epoch ceiling, minimum 10, patience 10 and physical/effective batches 32/128. Every declared model configuration and loss weight matched the plan; all have 2,977,383 trainable parameters.

Current/best weights strictly load. Weights, optimizer/scaler/history values are finite; identities, RNG presence, validation support, counts-based metrics, optimizer cursors, best epoch/threshold and stopping counters are consistent. All 77 logged AMP same-group retries recovered without FP32 training fallback; no fatal artifact was present. Benchmarks match full on T4/Torch/batch/support settings. These checks validate recorded execution, not ideal conditioning.

AUROC/AP were checked for finite values and valid ranges; independent reproduction requires validation predictions. Timings are single-run compute measurements with the recorded benchmark scope, not uncertainty estimates. No new training or final-test evaluation was performed.

## Latest status: 2026-11-12

All 20 controls are complete. The new `without_graph`, `auxiliary_half` and `auxiliary_double` outputs passed 108 checks; their F1 scores are 79.76%, 80.61% and 79.19%, respectively, versus full 80.22%. They retain Accuracy-based checkpoint/threshold selection and remain single-seed comparisons. See [the latest review](FOLLOWUP_RESULTS_REVIEW.md) and batch 02 `validation_summary.csv` for the complete 21-arm table.

The independent queue now has zero fresh launches; AtCoder/full/231 is complete with F1 47.43%. BCB/full/231 has two completed attempts pending provenance; no representative is selected. Expensive baseline/transfer work is separate.

The seven-loss combined evidence remains batch 02 `_audit/loss_controls_seed42_audit.json`; final three controls are in `_audit/final_three_controls_2026-11-12_audit.json`. Original audit snapshots and ZIPs are preserved.
