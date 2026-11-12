# Current revision results and remaining work

Updated 2026-11-12. CSV trackers are authoritative; `not_reported` means no
verified local output, not that a job is idle or has never started.

**Next analysis batch is ready, not yet run:** [AtCoder diagnostics and CodeNet
F1 reassessment](../05_analysis/continuation/README_FA.md). Four GPU inference
jobs reuse the existing checkpoints; their CPU collector follows. The pinned
inputs preserve original epoch selection and keep final tests off.

| Tracker | Verified complete | Other status |
| --- | ---: | --- |
| `run_progress.csv` | 18 / 29 baseline/full seed-42 jobs | Four incomplete; seven without submitted output. |
| `control_progress.csv` | 20 / 20 CodeNet seed-42 controls | Complete; no control reruns needed. |
| `core_repeat_progress.csv` | 12 / 12 core repeats at seeds 123/231 | Complete; do not rerun. |
| `selected_followup_progress.csv` | 3 / 4 unambiguous full repeats | BCB/231 has two verified attempts pending provenance; all four identities have completed evidence. |
| `checkpoint_review_progress.csv` | 0 / 4 GPU analyses | New inference notebooks ready; CPU collector waits for their reports. |

The [current independent queue](../queues/RUN_NEXT_INDEPENDENT_BATCH.md) has **zero fresh launches**. All 19 planned independent job identities have completed evidence; BCB/full/231 has two distinct attempts pending provenance, excluded from seed averages. AtCoder/full/231 completed with F1 47.43% and AUROC 0.5125; both extra AtCoder seeds performed poorly. Investigate before scheduling further training. External baselines and transfer remain separate work.

## Read the evidence

- [New controls and cross-benchmark follow-ups](FOLLOWUP_RESULTS_REVIEW.md): all seven uploads passed 252 checks; weak AtCoder/123 retained, two BCB/231 attempts preserved.

- [Seven loss controls](LOSS_CONTROLS_REVIEW.md): 252/252 checks passed; small
  improvements after removing some terms do not establish universal loss necessity.
- [Graph/descriptor controls](ARCHITECTURE_CONTROLS_REVIEW.md): five complete,
  including the weak projected-DDG result retained without assuming a dataset bug.
- [Core readout/objective comparison](CORE_CONTROLS_REVIEW.md): seed-42 core.
- [Three-seed core results](CORE_REPEATS_PROGRESS.md): all 18 core runs including
  seed 42; descriptive means/sample SD, not final-test superiority.
- [Current full models](SPECTRA_INPUT_V2_REVIEW.md) and [GMN results](GMN_RESULTS_REVIEW.md).
- [Historical input-v1 batch](BATCH_2026-11-09.md): earlier protocol, not current rankings.

These are validation results. Keep `FINAL_TEST=False` until comparisons and
selection are frozen. Preserve unfavorable outcomes and failed attempts.

## Local storage and continuation

From the project root, reviewer-response ZIPs live at `../outputs/revesion/01`,
`02` and `03`. Batch 02 has one current `RESULTS_REVIEW.md`
and `validation_summary.csv`; earlier review/audit snapshots are archived.
Original submitted-paper results are under `../outputs/paper/` relative to the
project root. Use the [external results index](../../../outputs/README.md)
for direct links.

Preserve the whole saved Kaggle output: `progress.pt`, `status.json`, benchmarks,
numerical events and, for transformer jobs, the token cache and pinned model
revision. Local `revision/results/kaggle/` is an ignored staging directory.
Keep older saved output versions when continuing a session.

CodeNet CodeBERT, C4, CDLH and Qwen still have incomplete local reports. Follow
[the continuation guide](../00_plan/TEXT_RUN_CONTINUATION.md) with each job's own
checkpoint/runtime; the [small-report collector](../00_plan/SMALL_REPORT_DOWNLOADS.md)
can report large text outputs without downloading their weights. Do not infer
completion or final scores from intermediate epoch logs.
