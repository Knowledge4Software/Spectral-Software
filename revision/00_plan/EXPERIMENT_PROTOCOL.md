# Frozen first-stage experiment protocol

**2026-11-09 amendment:** [input v2](INPUT_V2_FIX.md) preserves Java callee
lexical features. Dataset bundles and scientific settings stay unchanged.
The current machine-readable release is `spectra_revision_stage1_input_v2`;
the original v1 JSON and notebook bytes are in `../archive/input_v1/`.
All new SPECTRA models/controls use v2; old external baselines remain reusable
under the explicitly documented compatibility exception.

Version `spectra_revision_stage1_v1`, 2026-11-06. This specification replaces the
older run-order suggestions. Its machine-readable companion is
`../queues/stage1_protocol.json`; `../queues/stage1_run_queue.csv` identifies
each exact notebook. Paths in these manifests are relative to `revision/`.
The 2026-11-07 folder reorganization preserves scientific settings and notebook
bytes. The CodeNet pilot is now reported complete; see
[the current execution guide](../START_HERE_FA.md).

## Questions and claim boundaries

1. Does the spectral readout help beyond the same encoder with mean/max or
   attention readout, with and without auxiliary supervision?
2. How do the methods perform on the same disjoint derived benchmark support?
3. In a prespecified Java-to-Python case study, how do direct and longer bridge
   paths behave under matched training/validation and target-label budgets?

CodeNet and AtCoder are the cross-language benchmarks. BigCloneBench is a
Java-only semantic detection control. The revised primary comparison will use
new derived splits; submitted four-epoch numbers are historical results and
cannot be inserted into this table. Results may weaken the original claims.

The initial transfer scope is a focused case study. It does not establish
universal behavior across the original 60 paths. Broader experiments are a later
decision requiring an explicit protocol extension, independent of test scores.

## Datasets and leakage policy

| Dataset archive | Train | Validation | Test | Separation |
| --- | ---: | ---: | ---: | --- |
| codenet_revision_common_support.zip | 69,538 | 14,896 | 14,906 | Problem, endpoint, submission and exact source |
| atcoder_revision_common_support_v2.zip | 70,000 | 15,000 | 15,000 | Problem, endpoint, submission and exact source |
| bigclonebench_revision_common_support_v2.zip | 70,000 | 7,129 | 7,566 | Endpoint, submission and exact source; problem labels unavailable |

ZIP SHA-256 values are frozen in `stage1_protocol.json`. All compared methods
use the same retained pair support, not a private subset per baseline. Do not
attach the older CodeNet problem-disjoint AST-only archive. The prepared graph
exports and DDG alignment sidecars are reused; no full parser rerun is needed.

Problem identity is a functionality proxy, not an execution proof. User IDs and
historical parser failure logs are unavailable. BigCloneBench has only 127
validation and 128 test programs; its large pair count does not imply equally
many independent observations. Original split label prevalences are retained.
Official Type-IV annotations are unavailable; do not infer them from predictions.

## Baseline roster

Initial screening: SPECTRA full plus ASTNN, RtvNN, CDLH, DeepSim, FA-AST GGNN,
FA-AST GMN, CodeBERT Siamese, C4-adapted and Qwen3-Embedding-0.6B LoRA.
C4-adapted is omitted on BigCloneBench because authoritative task IDs are absent.
This produces **10 CodeNet + 10 AtCoder + 9 BigCloneBench = 29 jobs**, seed 42.

The six legacy encoders are portable project adaptations. C4-adapted is a
task-masked contrastive adaptation using training problem labels; it is not an
exact reproduction of the authors' full protocol. Qwen is LoRA fine-tuned;
the separate frozen Qwen probe is optional and is not one of these 29 jobs.
Document encoder pretraining and C4's extra task-label supervision.

Original GraphCodeBERT, plain GNN/SNN, Deckard and standalone fixed-spectrum
rows are excluded from the new primary comparison unless separately rebuilt
and rerun on this support. Projected AST/DDG controls address a narrower
attribution question; next-token edges must never be called CFG.

Exact FSD-CLCD and ZC3 are not provided as completed reproductions. In particular,
R3's request for FSD-CLCD remains a known limitation requiring a later verified
implementation or an explicit explanation and narrower comparison claims.
The current roster is sufficient to begin screening, not to declare every
reviewer request satisfied. No result is represented as an unimplemented method.

## Selection, budgets and configuration policy

- Primary validation metric: Accuracy on CodeNet/AtCoder; F1 on BigCloneBench.
  Thresholds are optimized on validation only. Checkpoint ties use the other
  metric; remaining ties retain the earlier checkpoint. Report precision,
  recall, Accuracy, F1, macro-F1, balanced accuracy, AUROC, average precision
  (`AUPRC_AP`) and MCC. Average precision is not trapezoidal PR area.
- One declared default configuration per method in this stage; no learning-rate
  search. Exact learning rates, decay and physical/effective batches are in
  `stage1_protocol.json`. Defaults are not claimed to be each method's optimum.
  A later tuning extension must declare candidates before running, use separate
  output directories, record every candidate and choose on validation only.
- Main training: maximum 100 epochs, minimum 10, patience 10 validations without
  improvement; effective batch 128. The ceiling is not a demand to finish 100
  epochs in one session. Record patience stop, epoch cap and incomplete status
  separately. Do not reduce individual models' epoch caps after seeing scores.
- SPECTRA uses at most 256 input nodes, latent size 32, hidden size 256. Full
  model/loss settings are frozen in JSON. Non-spectral controls retain the same
  encoder inputs. Actual parameter counts are reported.
  Text inputs use at most 512 tokens per program; their model/objective/pooling
  and LoRA settings are also in JSON. A node cap and token cap are different
  coverage limits; report truncation separately for each input representation.
- Physical batches differ for memory. Accumulation does not give contrastive
  training the same negatives as a physical 128-example batch. Fix each method's
  physical batch after its hardware pilot. An OOM-driven change creates a new
  configuration requiring a fresh run, not continuation of the old checkpoint.
- Each Save & Run has a 10-hour internal budget with 45 minutes reserved. Work
  stops around 9h15 from the configuration cell, including preparation. Periodic
  optimizer-boundary checkpoints preserve progress. A clock-limited job is
  incomplete until continued. There is no verified guarantee that an entire
  model converges within one 12-hour session. Runtime forecasts are estimates.
- Pretrained commits resolve on first model download. Record `tokens/tokens.json`
  and use its commit in `MODEL_REVISION` for continuation and later seeds.
  GPU memory, downloads and actual speed must be checked on Kaggle.
- Training-source changes invalidate continuation; analysis-only changes do not.
  The AMP numerical-fix release permits one explicit, allowlisted upgrade from
  the known earlier implementation with identical data/model/loss/settings.
  It records lineage and does not accept arbitrary source changes; see
  `NUMERICAL_RECOVERY.md`. Overflow attempts replay the same effective batch
  with restored RNG/buffers, bounded scale backoff and a final FP32 attempt.
  The explicit dependency list is in `revision/versioning.py`. Preserve the
  release notebooks and dataset hashes with every output. Never edit a generated
  embedded runtime by hand. A scientific configuration mismatch raises an error.

## Stage gates and repetition

1. Run CodeNet full seed 42 first; examine finite loss, validation progress,
   checkpoint output, memory and runtime. Continue it if incomplete. A successful
   research-protocol pilot is reused as queue job 1.
2. Check the ASTNN and CodeBERT hardware paths while finishing the 29-job queue.
   Independent jobs can run in parallel; the current launch list uses 24 slots.
   Keep `FINAL_TEST=False`. Collect `status.json` with the validation screening
   notebook. Only completed runs can be ranked; resumed snapshots are deduplicated.
3. Run CodeNet's 21 architecture/loss arms with seed 42; reuse the full-model job.
   This includes the readout-by-objective controls, projected AST/DDG variants,
   descriptor components, loss removals and half/double auxiliary weights.
4. Freeze finalists from validation only: the six central readout-by-objective
   controls on CodeNet, plus the top two eligible external baselines per dataset.
   Rank by its primary metric, then the other of Accuracy/F1, then fewer trainable
   parameters, then the roster order above. Add seeds 123/231 only to selected
   comparisons. Do not rerun every baseline with three seeds automatically.
   Cross-benchmark follow-ups and additional loss replications require an explicit
   ledger entry; any single-seed findings remain exploratory.
5. Run focused transfer after choosing its models; freeze every model/condition
   before test release. Then release final tests from completed checkpoints,
   run diagnostics/cost/uncertainty, and rewrite the manuscript from the evidence.

Single-seed screening and validation-selected winners are not unbiased estimates
of superiority. Final comparisons need independent test reporting and repeated
seeds. Multiple paths share programs/problems and cannot be treated as independent
replicates. BigCloneBench cannot use problem-clustered inference without labels.

## Prespecified focused transfer (later stage; do not run now)

J=Java, C=C++, S=C#, P=Python. All conditions use the **same Python-Python test**.
Each language-pair quota is balanced 50/50 by label. The code implements these
under `focused|...`; `focused_transfer_plan.csv` records the exact budgets.

| Condition | Training quotas | Validation quotas |
| --- | --- | --- |
| Source only | JJ 6000 | JJ 1200 |
| Direct | JJ 4500, JP 1500 | JJ 900, JP 300 |
| Two hops | JJ 2250, JC 2250, CP 1500 | JJ 450, JC 450, CP 300 |
| Three hops | JJ 1500, JC 1500, CS 1500, SP 1500 | 300 each |
| Target reference | PP 6000 | PP 1200 |

Three bridge conditions each have 1,500 target-containing training pairs and
300 target-containing validation pairs. Source-only and target-reference are
contextual controls with intentionally different exposure. Non-target language
composition still changes with path length, preventing a pure causal path-length
claim. Source-only means no target task-training/validation labels; pretrained
encoders can already have seen the language. Optional two-hop reinforcement
uses JJ/JC/CC/CP 1500 training and 300 validation pairs each.

Use 30 epochs and 1,410 optimizer updates, selecting the best validation
checkpoint within the fixed schedule. Compare full SPECTRA, the stronger
non-spectral sibling and the strongest eligible external comparator, selected on
completed CodeNet main validation results. Five conditions x three models x
three seeds gives 45 jobs; start seed 42 first. This stage is not in the current
29-job package. SPECTRA transfer headers accept `ARM="mean_max"` or `"attention"`
and preserve a distinct arm ID/output path. The old `table7_cross` condition
trains on both endpoint languages and tests mixed pairs; it is not source-only.

## Required evidence and open work

Retain status/history, checkpoints, model commit/cache, GPU/software information,
configurations, runtime forecasts and benchmarks. Later retain test predictions
and validation-selected thresholds. Do not use a session's best partial score as
a completed result. Do not report unavailable annotations or hypothetical scores.

This release prepares experiments. It does not establish improved performance,
full convergence, language invariance or paper acceptance. Exact specialized
baselines, annotation-dependent analysis, theoretical claims, figures, related
work and manuscript rewriting remain separate review-response work.
