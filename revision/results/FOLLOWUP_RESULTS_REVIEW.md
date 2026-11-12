# Full-model follow-up results: 2026-11-12

Four uploaded archives passed all 144 checkpoint/source/configuration/metric checks and stopped normally on validation patience. They cover **three of four planned job identities**. No final test was evaluated.

| Dataset | Seed | Validation F1 | AUROC | Epochs / best | Recorded hours | Treatment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bigclonebench | 231 | 92.00% | 0.9686 | 27 / 17 | 1.64 | Distinct attempt; selection pending |
| atcoder | 123 | 56.10% | 0.5255 | 12 / 2 | 0.81 | Complete; retain |
| bigclonebench | 231 | 92.84% | 0.9883 | 19 / 9 | 1.13 | Distinct attempt; selection pending |
| bigclonebench | 123 | 95.88% | 0.9865 | 27 / 17 | 1.64 | Complete; retain |

## Two distinct BCB seed231 attempts

Former `results(2).zip` and `results(4).zip` both record BigCloneBench, full, seed231, the same run identity, source hash, model/loss settings, dataset hash and benchmark support. Both have empty continuation lineage. They have different checkpoint weights and histories, starting from their first epoch, so neither is a duplicate upload nor a saved continuation of the other. F1 is 92.00% versus 92.84%. Preserve both with hash-based attempt suffixes; neither is assigned as the authoritative seed231 representative, and no three-seed BCB average is published until provenance is resolved. Two attempts at the same seed are not two independent seeds.

The files do not include the Kaggle notebook/version identities or enough environment detail to establish why the trajectories differ. Numerical or environment nondeterminism is a possibility, not an established diagnosis. Do not choose the higher score or reinterpret either as the missing AtCoder seed231 result.

## Weak AtCoder seed123 result retained

F1 is 56.10%, Accuracy 52.67% and AUROC 0.5255 at the selected epoch 2. Training stopped after 12 validated epochs, following ten non-improvements; it did not crash or run out of Kaggle time. Saved weights and optimizer values are finite and load strictly, model/data/loss/source settings match the plan, and no numerical recovery or fatal artifact was recorded. This audit establishes file and protocol consistency; it does not establish a cause for the near-chance result or rule out a modeling issue. Retain it as a completed weak result, without an outcome-driven replacement. It materially limits robustness claims given seed42 F1 79.36%.

AtCoder checkpoints and thresholds here were selected by Accuracy. BCB used F1. Reporting F1 does not change the selection protocol after training. The F1 reassessment discussed previously remains separate work.

## Remaining output and evidence

Only **AtCoder / full / seed231** remains without submitted output in the 19-job independent roster. Check Kaggle before launching a duplicate. All 20 CodeNet seed42 controls and all 12 CodeNet core repeats are complete. Expensive external baselines and transfer work remain separate; this is not completion of the entire revision.

Outputs: `../outputs/revesion/03/` relative to the repository root. `followup_validation_summary.csv` records all four attempts, with both seed231 rows excluded from seed averages. `_audit/new_uploads_2026-11-12_audit.json` has all seven new archive checks; `_audit/new_uploads_2026-11-12_moves.json` preserves original names, destinations and hashes. No training code, notebook or dataset changed during this cleanup.
