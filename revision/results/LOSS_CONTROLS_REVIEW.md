# CodeNet loss controls: six additional verified runs

**All six new seed-42 uploads passed 216/216 checks and stopped normally on validation patience.** The control tracker now has 16 of 20 controls complete; four controls still have no submitted output. Final tests remain off.

| Condition | Validation Accuracy | Validation F1 | Change from full, Accuracy points | Epochs / best | Recorded hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full reference | 79.67% | 80.22% | 0.00 | 39 / 29 | 2.70 |
| without_auc_ranking | 79.10% | 79.40% | -0.58 | 43 / 33 | 2.95 |
| without_hard_negative | 80.16% | 80.81% | +0.48 | 46 / 36 | 3.18 |
| without_reconstruction | 79.77% | 79.97% | +0.10 | 41 / 31 | 2.87 |
| without_spectral | 80.12% | 81.10% | +0.44 | 43 / 33 | 3.20 |
| without_spectral_and_hard_negative | 79.90% | 80.04% | +0.23 | 40 / 30 | 2.85 |
| without_spectral_variance | 79.36% | 79.60% | -0.31 | 37 / 27 | 2.59 |

## Interpretation

`without_spectral` removes the spectral auxiliary loss; it retains the spectral readout. Likewise, the joint arm removes spectral and hard-negative auxiliary losses while keeping the other objectives. These are loss-attribution comparisons, not spectral-versus-nonspectral readout comparisons.

Removing the spectral or hard-negative loss slightly improves the primary validation Accuracy in this seed; removing AUC ranking lowers it. The results do not support a claim that every auxiliary term is necessary or individually improves Accuracy. Small single-seed validation differences are not significance tests, and confusion matrices alone cannot reconstruct paired disagreements. Preserve all arms and revise necessity claims accordingly.

## Integrity and limitations

All six use the original input-v2 training hash `8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26`, identical to the full reference. The entire archived scientific source set was fingerprinted. The later conditional FP32 validation amendment is not needed to relabel these successful original-runtime runs; their raw identities are preserved.

All use the same 69,538/14,896/14,906 CodeNet train/validation/test support, seed 42, Accuracy selection, 100-epoch ceiling, minimum 10, patience 10 and physical/effective batches 32/128. Every declared model configuration and loss weight matched the plan; all have 2,977,383 trainable parameters.

Current/best weights strictly load. Weights, optimizer/scaler/history values are finite; identities, RNG presence, validation support, counts-based metrics, optimizer cursors, best epoch/threshold and stopping counters are consistent. All 64 logged AMP same-group retries recovered without FP32 training fallback; no fatal artifact was present. Benchmarks match full on T4/Torch/batch/support settings. These checks validate recorded execution, not ideal conditioning.

AUROC/AP were checked for finite values and valid ranges; independent reproduction requires validation predictions. Timings are single-run compute measurements with the recorded benchmark scope, not uncertainty estimates. No new training or final-test evaluation was performed.

## Still awaiting outputs

- `without_topology_reconstruction`
- `without_graph`
- `auxiliary_half`
- `auxiliary_double`

The four AtCoder/BigCloneBench full-model repeats are tracked separately. The original independent 19-job roster now has 11 verified completions and eight jobs without local output. Check Kaggle for live jobs before launching duplicates.

Exact notebooks: [current independent queue](../queues/RUN_NEXT_INDEPENDENT_BATCH.md). The local evidence is batch 02 `_audit/loss_controls_seed42_audit.json` with its rename manifest.
