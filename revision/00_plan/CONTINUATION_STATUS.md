# Current continuation gates: 2026-11-16

| Work | Evidence now | Next action |
| --- | --- | --- |
| Readout attribution / reviewer 6 | All six CodeNet core conditions x three seeds complete; consistent spectral superiority is not established | Seed42 F1 verified (spectral 80.69%, mean/max 80.61% with auxiliary losses); await seed123/231 analysis outputs |
| Auxiliary-loss contribution | All 20 CodeNet seed42 controls complete; some loss removals outperform full | Retain every result and narrow necessity claims; no further weight search scheduled |
| AtCoder robustness | Full/123 and /231 near chance; full/42 substantially higher | Diagnosis complete: both weak seeds are also near chance on a fixed train sample, with near-constant scores and low-rank descriptors; investigate training before changes |
| BCB repeated results | Full/123 complete; user confirmed accidental repeat of full/231 | Preserve both as one seed; primary attempt unspecified, neither selected by score; no new run needed |
| Modern baselines | CodeNet CodeBERT/C4/CDLH/Qwen locally incomplete; other dataset text rows not all reported | Retain continuation checkpoints; use existing version-specific guides after the diagnostic gate, without claiming convergence from partial runs |
| External baseline replication | Portable graph adaptations complete at seed42; selected extra-seed comparisons remain | Freeze eligible comparators from completed validation screening; preserve equal budgets and provenance |
| Controlled transfer / unseen-language claims | Focused matched-budget notebooks exist; revised confirmation remains incomplete | Run a justified subset after comparator/protocol selection; do not reuse old unmatched transfer rows as revised evidence |
| Learned vs fixed spectra | Existing representation diagnostics do not prove a matched-classifier advantage | Assess available diagnostics and specify any additional matched comparison before claiming causality |
| Final tests / uncertainty | Final tests remain off | Freeze design, then evaluate matched sets with fixed thresholds and dependent-pair uncertainty |
| Manuscript and reviewer response | Current results limit strong superiority/generalization claims | Rewrite from completed revised evidence; disclose derived BCB support, adaptations, failed attempts and protocol changes |

**Current:** two GPU analyses verified complete (AtCoder and CodeNet/42); await CodeNet/123 and /231 reports, followed by the CPU collector in
[the continuation guide](../05_analysis/continuation/README.md). The detailed findings are in [the reviewed results](../results/CHECKPOINT_REVIEW_RESULTS.md). No new training experiment or architecture change
has been scheduled as part of this analysis batch.

Analysis reports go under `../outputs/revesion/04_checkpoint_review` relative to
the repository root. [The registered analysis plan](F1_CHECKPOINT_REASSESSMENT.md)
defines the F1 calibration and diagnostic limits. The next scientific decision
depends on those reports; completion of old training queues is not completion
of the reviewer response.
