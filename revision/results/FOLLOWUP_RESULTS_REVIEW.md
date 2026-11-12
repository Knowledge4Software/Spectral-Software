# Full-model follow-up results: 2026-11-12

Five uploaded archives passed all 180 checkpoint/source/configuration/metric checks and stopped normally on validation patience. They cover **all four planned job identities**; the BCB/231 representative remains unresolved because two distinct attempts were supplied. No final test was evaluated.

| Dataset | Seed | Validation F1 | AUROC | Epochs / best | Recorded hours | Treatment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bigclonebench | 231 | 92.00% | 0.9686 | 27 / 17 | 1.64 | Distinct attempt; selection pending |
| atcoder | 123 | 56.10% | 0.5255 | 12 / 2 | 0.81 | Complete; retain |
| bigclonebench | 231 | 92.84% | 0.9883 | 19 / 9 | 1.13 | Distinct attempt; selection pending |
| bigclonebench | 123 | 95.88% | 0.9865 | 27 / 17 | 1.64 | Complete; retain |
| atcoder | 231 | 47.43% | 0.5125 | 12 / 2 | 0.80 | Complete; retain |

## Two distinct BCB seed231 attempts

Former `results(2).zip` and `results(4).zip` both record BigCloneBench, full, seed231, the same run identity, source hash, model/loss settings, dataset hash and benchmark support. Both have empty continuation lineage. They have different checkpoint weights and histories, starting from their first epoch, so neither is a duplicate upload nor a saved continuation of the other. F1 is 92.00% versus 92.84%. Preserve both with hash-based attempt suffixes; neither is assigned as the authoritative seed231 representative, and no three-seed BCB average is published until provenance is resolved. Two attempts at the same seed are not two independent seeds.

**User confirmation, 2026-11-12:** this was an accidental second execution of the same experiment. The reason for two uploads is now known. Both are retained as attempts at one seed; run order and the primary attempt remain unspecified, so neither is selected by score. This does not block the four checkpoint-review jobs and requires no new training run.

The files do not include the Kaggle notebook/version identities or enough environment detail to establish why the trajectories differ. Numerical or environment nondeterminism is a possibility, not an established diagnosis. Do not choose the higher score or reinterpret either as an AtCoder seed231 result.

## Weak AtCoder repeats retained

Seed123 has F1 56.10%, Accuracy 52.67% and AUROC 0.5255. Seed231 has F1 47.43%, Accuracy 51.24%, Precision 51.45%, Recall 43.99% and AUROC 0.5125. Both selected epoch 2 and stopped after 12 validated epochs following ten non-improvements. Seed231 trained for 0.760h plus 0.039h setup; it completed 6,564 optimizer updates. Its training loss decreased from 0.93489 to 0.87789, while validation AUROC stayed near 0.5 (0.4926-0.5125). Threshold recalibration alone cannot repair near-chance score ranking.

Neither run crashed or exhausted its session budget. Saved weights, optimizer/scaler/history are finite and load strictly; dataset identities, model/loss/source settings, stopping counters and confusion-count metrics match the plan. Neither recorded numerical recovery events. File consistency does not identify the cause of poor learning or rule out a modeling/input pipeline issue. These completed weak runs remain in the evidence; no outcome-driven replacement was scheduled.

Across seeds 42/123/231, AtCoder validation F1 is 79.36%, 56.10% and 47.43%: mean 60.96% with sample SD 16.51 percentage points. These descriptive values are not confidence intervals. The large drop across seeds prevents a robust-performance claim. Investigate training/representation behavior before proposing further training or changing the model.

AtCoder checkpoints and thresholds here were selected by Accuracy. BCB used F1. Reporting F1 does not change the selection protocol after training. The F1 reassessment discussed previously remains separate work.

## Remaining output and evidence

**No fresh job remains without output in the 19-job independent roster.** All 19 identities have completed evidence, with 18 unambiguous representatives and BCB/231 pending provenance. Do not rerun completed jobs automatically. All 20 CodeNet seed42 controls and all 12 CodeNet core repeats are complete. Expensive external baselines and transfer work remain separate; this is not completion of the entire revision.

Outputs: `../outputs/revesion/03/` relative to the repository root. `followup_validation_summary.csv` records all five attempts, with both BCB seed231 rows excluded from seed averages. `_audit/atcoder_full_seed231_audit.json` records the latest 36/36 checks and its rename manifest. `_audit/new_uploads_2026-11-12_audit.json` retains the previous seven-upload checks; `_audit/new_uploads_2026-11-12_moves.json` preserves original names, destinations and hashes. No training code, notebook or dataset changed during this cleanup.
