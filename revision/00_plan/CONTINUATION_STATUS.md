# Current continuation gates: 2026-11-16

| Work | Evidence now | Next action |
| --- | --- | --- |
| Readout attribution / reviewer 6 | All six CodeNet core conditions x three seeds complete; consistent spectral superiority is not established | All 18 reviewed locally; mean F1 full 80.46%, mean/max 80.59% with auxiliaries; no demonstrated spectral advantage |
| Auxiliary-loss contribution | All 20 CodeNet seed42 controls complete; some loss removals outperform full | Retain every result and narrow necessity claims; no further weight search scheduled |
| AtCoder robustness | Full/123 and /231 near chance; full/42 substantially higher | V1/v2 local probes complete; objective effects reverse at healthy seed42, and FP32 did not rescue tiny seed231. Six matched full-data F1 confirmations are prepared for Kaggle, not yet run. |
| BCB repeated results | Full/123 complete; user confirmed accidental repeat of full/231 | Preserve both as one seed; primary attempt unspecified, neither selected by score; no new run needed |
| Modern baselines | CodeNet CodeBERT/C4/CDLH/Qwen locally incomplete; other dataset text rows not all reported | Retain continuation checkpoints; use existing version-specific guides after the diagnostic gate, without claiming convergence from partial runs |
| External baseline replication | Portable graph adaptations complete at seed42; selected extra-seed comparisons remain | Freeze eligible comparators from completed validation screening; preserve equal budgets and provenance |
| Controlled transfer / unseen-language claims | Focused matched-budget notebooks exist; revised confirmation remains incomplete | Run a justified subset after comparator/protocol selection; do not reuse old unmatched transfer rows as revised evidence |
| Learned vs fixed spectra | Existing representation diagnostics do not prove a matched-classifier advantage | Assess available diagnostics and specify any additional matched comparison before claiming causality |
| Final tests / uncertainty | Final tests remain off | Freeze design, then evaluate matched sets with fixed thresholds and dependent-pair uncertainty |
| Manuscript and reviewer response | Current results limit strong superiority/generalization claims | Rewrite from completed revised evidence; disclose derived BCB support, adaptations, failed attempts and protocol changes |

**Current:** the CodeNet three-seed checkpoint review and CPU collector completed locally; AtCoder diagnostics were already verified. No additional Kaggle F1 or collector run is needed. See [the reviewed results](../results/CHECKPOINT_REVIEW_RESULTS.md). AtCoder training/representation checks and the remaining external-baseline/transfer work remain separate gates.

Analysis reports go under `../outputs/revesion/04_checkpoint_review` relative to
the repository root. [The registered analysis plan](F1_CHECKPOINT_REASSESSMENT.md)
defines the F1 calibration and diagnostic limits. The next scientific decision
depends on those reports; completion of old training queues is not completion
of the reviewer response.

The [tiny AtCoder learning probe](../results/ATCODER_LEARNING_PROBE.md) is complete. Its 240 fresh diagnostic updates do not replace any main training run. The Kaggle fallback is retained for reproduction, not required again now.

**Kaggle handoff:** [six matched AtCoder F1 confirmations](../06_atcoder_confirmation_v1/README.md). The local v2 controls took4m44s; longer full-data confirmation exceeds the ten-minute local limit. The [v2 report](../results/ATCODER_LEARNING_CONTROLS.md) preserves the revised interpretation; removing auxiliaries is not declared a fix.

**2026-11-17 confirmation update:** five complete runs passed130 checks. All three full objectives are near chance; classification-only/42 and /231 reach approximately81.7% F1. Await classification-only/123 before a complete three-seed comparison. The new failure of full/42, previously successful, is a reproducibility concern with cause unresolved. See [the confirmation review](../results/ATCODER_CONFIRMATION_RESULTS.md). No prior result is replaced.

**Final 2026-11-17 intake:** all six AtCoder confirmation runs verified (156 checks). Classification-only mean F1=81.89% +/-0.31 points; full=66.76% +/-0.06. This batch needs no rerun or continuation. Earlier five-run pending notes are historical. See the current confirmation review for limitations and remaining gates.

**Next registered readout check, 2026-11-17:** [three AtCoder mean/max classification-only jobs](../07_atcoder_readout_check_v1/README.md), seeds42/123/231. Mean/max was selected using completed CodeNet three-seed evidence before new AtCoder readout results. Reuse all three completed spectral F1-selected references; no new spectral jobs. Same encoder/objective/training/F1-selection budget, different head capacity reported. This conditional comparison addresses reviewer6 attribution; original full-loss failures remain visible and modern-baseline/transfer gaps remain separate.
