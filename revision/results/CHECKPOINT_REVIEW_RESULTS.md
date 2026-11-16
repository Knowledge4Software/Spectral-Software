# Checkpoint-review results received 2026-11-16

**Two analyses are verified complete: AtCoder diagnosis and CodeNet F1 seed42. All 198 audit checks passed.** No training update, final-test evaluation, missing input, numerical recovery or failed diagnostic probe was reported. Each archive contains six completed weight views: six CodeNet models or best/last weights for three AtCoder seeds.

The audit checked ZIP CRC/member uniqueness, exact embedded scientific/reporting sources, pinned checkpoint identities and hashes, every exported artifact checksum, exact ordered pair/label support against the unchanged local dataset bundles, and all probability-derived metrics and F1 thresholds. Training-sample support, diagnostic array shapes/finiteness, descriptor effective ranks and weighted loss totals also reproduced. The original checkpoint weights are not included in these reports; their integrity is based on pinned hashes and prior checkpoint audits, not new local inference.

## CodeNet seed42: validation F1

| Readout | With auxiliary losses | Classification only |
| --- | ---: | ---: |
| Spectral | 80.69% | 79.06% |
| Mean/max | 80.61% | 80.67% |
| Attention | 79.51% | 80.95% |

With matched auxiliary objectives, full spectral exceeds mean/max by only **0.08 percentage points**. Classification-only attention is highest overall in this seed (80.95%). Within classification-only training, spectral is lower than both simple readouts. This does not establish general spectral superiority. Seed123/231 F1 reports are still required before the prespecified three-seed summary; no partial aggregate, confidence interval or significance claim is produced.

All six original-threshold best-model F1 scores reproduced exactly. Every run uses the same 14,896 validation pairs, with 7,448 positives. Thresholds were calibrated for F1 at existing Accuracy-selected epochs; this is not retroactive F1 epoch selection or final-test performance.

## AtCoder: best saved weights

| Seed | Original-threshold F1 | Calibrated F1 | Train-sample AUROC | Validation AUROC | Descriptor effective rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 79.36% | 79.85% | 0.9708 | 0.8561 | 3.348 |
| 123 | 56.10% | 66.70% | 0.5128 | 0.5255 | 1.042 |
| 231 | 47.43% | 66.68% | 0.4875 | 0.5125 | 1.004 |

Seeds123/231 perform near chance on the same fixed **2,048 training pairs**, as well as all 15,000 validation pairs. This supports a failure to learn useful discrimination, beyond merely a validation generalization gap. Best-model score standard deviations are 0.000956 and 0.000073, compared with 0.39416 for seed42: the weak models output nearly constant scores around 0.5. Their final saved weights also remain near chance (train AUROC 0.5010/0.5026; validation AUROC 0.5139/0.4944), so later saved weights do not rescue them.

The apparently improved calibrated F1 near 66.67% is essentially the balanced-set all-positive baseline, not recovered discrimination. Keep the failed training outcomes in the evidence. The last-weight views are diagnostic observations and do not replace the originally selected checkpoints.

Descriptor effective rank on the same 256 validation programs is about 1.00-1.04 for the weak seeds versus 3.35 at the best seed42 weights. This supports low-dimensional representation degeneration, but effective rank alone does not identify its cause. The fixed-batch FP32 classification loss is about 0.693 for both weak seeds versus 0.073 for seed42. Classification gradients are connected and finite, but much smaller in the weak spectral extractor. The auxiliary/total gradient direction differs from classification alone in the weak seeds; a single saved-weight probe does not prove auxiliary losses caused the training failure.

On the fixed 128-pair precision probe, weak-seed FP32/recorded-precision probabilities differ by only approximately 0.000005-0.000011. This provides no evidence that a large inference-precision error explains these weak results. It does not rule out effects of mixed precision during the original training, or other model/input/optimization issues. No dataset bug or specific repair has been established by this audit.

## Next work

1. Keep both completed analyses; they need no rerun.
2. Supply the outputs of `02_codenet_f1_seed123.ipynb` and `02_codenet_f1_seed231.ipynb`. Check any existing Kaggle jobs before launching duplicates.
3. Run the CPU collector after those reports arrive. Local partial collection correctly records 12 missing core reports and does not generate a three-seed core mean table.
4. Use the AtCoder evidence to design targeted training/representation checks before further full training. Increasing epochs or selecting a more favorable seed is not an established fix. Any training amendment needs a shared declared comparison protocol.

## Storage and evidence

The received files remain in `../outputs/revesion/05/` relative to the repository root; this user-selected arrival folder is preserved even though the initial guide suggested `04_checkpoint_review`.

- `01_atcoder_diagnosis_complete.zip` (formerly `results.zip`).
- `02_codenet_f1_seed42_complete.zip` (formerly `results(1).zip`).
- `_audit/checkpoint_review_audit_2026-11-16.json`: all 198 checks, raw report values and source/checkpoint bindings.
- `_audit/rename_manifest_2026-11-16.json`: unchanged ZIP hashes and original names.
- `core_seed42_f1.csv`, `atcoder_diagnostics_summary.csv`: reviewed per-model/per-view summaries.

Model evaluation took approximately 25-33 seconds per weight view on T4; those saved timings exclude initial input preparation. No scientific code, training notebook or dataset changed during this review.
