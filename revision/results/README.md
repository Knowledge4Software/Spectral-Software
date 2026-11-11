# Current revision results and remaining work

Updated 2026-11-11. CSV trackers are authoritative; `not_reported` means no
verified local output, not that a job is idle or has never started.

| Tracker | Verified complete | Other status |
| --- | ---: | --- |
| `run_progress.csv` | 18 / 29 baseline/full seed-42 jobs | Four incomplete; seven without submitted output. |
| `control_progress.csv` | 16 / 20 CodeNet seed-42 controls | Four without submitted output. |
| `core_repeat_progress.csv` | 12 / 12 core repeats at seeds 123/231 | Complete; do not rerun. |
| `selected_followup_progress.csv` | 0 / 4 AtCoder/BigCloneBench full repeats | Four without submitted output. |

The [current independent queue](../queues/RUN_NEXT_INDEPENDENT_BATCH.md) contains
eight jobs without local outputs: four controls plus four full repeats. Check
Kaggle for active jobs before starting duplicates. Earlier baseline continuations
and unreported external baselines are a separate roster.

## Read the evidence

- [Six new loss controls](LOSS_CONTROLS_REVIEW.md): 216/216 checks passed; small
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
