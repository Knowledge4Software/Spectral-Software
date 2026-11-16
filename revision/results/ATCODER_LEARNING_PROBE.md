# AtCoder tiny learning probe v1: 2026-11-16

**Completed locally in 338.7 seconds (5m39s), within the requested ten-minute limit.** Four fresh-training cases, 60 optimizer updates each, used the existing RTX3060 Laptop GPU. No prior checkpoint was modified or replaced. No final-test evaluation was performed.

The plan was registered in `queues/atcoder_learning_probe_manifest.json` and the standalone Kaggle fallback before the run: seeds123/231, full versus classification-only, the same balanced 64 training pairs and 256 validation pairs, original model and loss definitions, learning rate1e-4, physical/effective batches32/128. Each effective batch contains two permutations of the tiny sample. Siblings started with identical hashed weights and initial predictions; pair order and random seeds were matched. All cases ran the full fixed budget, with no early stopping or model selection. The frozen main training implementation remains unchanged.

| Seed | Objective | Final train F1 at 0.5 | Final train AUROC | Final train BCE | Small validation AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| 123 | full | 84.85% | 0.8896 | 0.4956 | 0.5311 |
| 123 | classification_only | 93.75% | 0.9785 | 0.2161 | 0.4976 |
| 231 | full | 59.65% | 0.7637 | 0.6024 | 0.5026 |
| 231 | classification_only | 95.24% | 0.9775 | 0.2016 | 0.5334 |

Both classification-only siblings learned substantially more of this tiny training sample. Full/123 also learned meaningful ranking, and full/231 improved somewhat; the gradient/input/head path is therefore not universally disconnected. The objective choice affects optimization in this controlled probe. This supports investigating auxiliary-loss interactions, rather than increasing the main training epoch cap without diagnosis.

**The strict prespecified memorization criterion was train AUROC>=0.95 AND BCE<0.2. None of the four cases met both within 60 updates.** Classification-only BCE remained 0.2161/0.2016; the criterion was not relaxed after seeing results. Thus this is stronger learning, not demonstrated perfect memorization. There is no healthy-seed42 control in this four-case pilot, so it cannot establish that this budget is sufficient for every initialization.

Validation AUROC remains approximately 0.50-0.53 after training on only 64 pairs. These are diagnostic development observations, not benchmark scores, evidence of repaired generalization or replacements for the original weak AtCoder runs. No improvement claim for full-dataset performance is established. No numerical replay or fatal error occurred in these 240 updates; this does not exclude numerical issues later in the original long training.

Next: use this evidence to register a broader matched objective/optimization confirmation, with a healthy-seed control and shared F1 selection/budget if full training is proposed. Preserve the original full-model failures and compare the same inputs and budgets. No additional training is launched by this report.

The Kaggle fallback `05_analysis/continuation/04_atcoder_learning_probe.ipynb` needs only the current AtCoder v2 dataset, no previous output/checkpoint. It is retained for reproduction; this completed probe does not require a Kaggle rerun. The local command is `.\.venv\Scripts\python.exe -m scripts.run_learning_probe_local`, with a nine-minute compute budget and a measured five-minute-thirty-nine-second completion here.

Reports: `../outputs/revesion/05/atcoder_learning_probe_v1_local/`, plus its `_small_reports.zip`. Final prediction-derived metrics, support hashes, sibling initialization and initial-metric equality, fixed update counts, history steps, criterion flags, source fingerprints and notebook checksum were independently verified. The original main-training checkpoints remain untouched.
