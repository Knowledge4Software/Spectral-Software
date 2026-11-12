# Validation checkpoint review: registered 2026-11-12

This analysis is specified before running the new Kaggle evaluation notebooks.
It adds no training epochs and changes no dataset, model, losses, stopping rule,
saved checkpoint or original selection metric.

## Fixed inputs and comparisons

- CodeNet: the existing six readout/objective conditions, each at seeds
  42/123/231, for 18 validation-selected checkpoints.
- AtCoder: full at seeds 42/123/231, evaluating both its saved best and last
  weights. The same 2,048-pair class-stratified training sample and up to 256
  validation programs are used across seeds, with sample seed 9173.
- Exact archive/checkpoint hashes, identities and settings are pinned in
  `queues/checkpoint_review_manifest.json` relative to `revision/`.
- The failed classification-only/123 attempt and both ambiguous BCB/231
  attempts are excluded from these analyses. Their original evidence remains.

## F1 threshold calibration

Every CodeNet condition receives the identical validation F1 threshold search.
Export all pair probabilities and report both the original selected threshold
and the newly calibrated threshold. All 18 conditions are retained, regardless
of ordering. Existing checkpoint epochs were selected by Accuracy; this
post-hoc calibration cannot recover the best F1 epoch, change early stopping,
or establish held-out test performance. All-positive F1 is also reported: on
a balanced set it is 2/3, so a higher F1 after calibration alone does not
establish useful ranking or resolve near-chance AUROC.

The collector requires all six conditions and three seeds, common ordered
pair/label support, and verified prediction artifact hashes before producing
core means/sample SD and matched-seed differences. Spectral versus mean/max
and attention are compared separately within each loss setting. The loss
interaction is reported with matched readouts. No seed, threshold procedure
or auxiliary-loss setting is selected to favor the spectral model. Sample SD
is descriptive, not a confidence interval or a significance test.

## AtCoder diagnostic scope

Compare train-sample and full-validation metrics, score distributions,
descriptor diversity, slot occupancy, latent density and one fixed FP32
gradient probe at saved weights. The gradient probe uses eval mode and has
no optimizer or updates. Export raw and weighted loss components, group
gradient norms and classification-versus-total gradient cosine. These probes
do not reconstruct training trajectories or establish a causal mechanism.
Precision comparison uses a fixed validation sample; original evaluation
precision is retained for full prediction passes with explicit whole-pass
FP32 recovery when needed. Both best and last weight views are labeled.

A failed diagnostic probe is recorded explicitly and does not discard other
seeds' validation predictions. Reports with diagnostic failures retain that
status in the collector. A session-budget stop remains incomplete; complete
per-view artifacts can be resumed with checked hashes.

## Gates for subsequent work

Interpret the diagnostics before scheduling architecture or loss changes.
If a fix/new variant is justified, register its version, shared comparator
budget and F1 epoch/threshold selection protocol before launching it. Preserve
all existing weak runs. Additional modern baselines, selected repeated
comparisons, controlled transfer and final-test uncertainty remain separate
requirements; this review does not mark the reviewer response complete.

Final tests remain off until the design and comparison choices are frozen.
