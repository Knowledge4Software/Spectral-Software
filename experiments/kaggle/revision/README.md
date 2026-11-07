# Reviewer-response experiments

These are new experiments. Their results must be reported separately from the
submitted four-epoch results. No new accuracy or superiority result is claimed
until the Kaggle jobs finish and the final test is explicitly released.

## Start here

If a T4 run failed with `Non-finite gradient norm`, use the fixed v3 notebook
and follow [NUMERICAL_RECOVERY.md](NUMERICAL_RECOVERY.md) to continue a retained
checkpoint. The numerical upgrade is explicit and records its source lineage.

The current execution instructions are [FIRST_RUN.md](FIRST_RUN.md) and the
frozen [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Start one CodeNet full
seed-42 pilot, then complete the 29-job seed-42 baseline queue before adding
selected repetitions. `stage1_run_queue.csv` gives the exact files. The separate
`spectra_revision_stage1_v1.zip` packet contains only this first stage.

1. Upload `../data/benchmarks/codenet_revision_common_support.zip` as a private
   Kaggle dataset. Use this SAME bundle for SPECTRA and the external baselines.
   It reuses graph exports, includes repaired Python/C# DDG alignment, and
   freezes common graph support. Preserve the earlier AST-only revision bundle
   as a separate experiment version; do not mix their results.
2. Enable a T4-class GPU and attach that dataset to the notebook.
3. Start with `01_readout_controls/codenet_full_seed42.ipynb`; inspect saved
   status and runtime before expanding. Continue its checkpoint if incomplete.
4. Complete the baseline queue with seed 42 on the three datasets; keep
   `FINAL_TEST=False`. Use the CPU validation collector for a screening summary.
5. Then run CodeNet architecture/loss screening. Add seeds 123/231 only to
   selected central comparisons under the frozen protocol.

Read [RUN_ORDER.md](RUN_ORDER.md) for the staged run list and continuation steps,
and [REVIEWER_COVERAGE.md](REVIEWER_COVERAGE.md) for the exact limits of the fixes.
The notebooks are an experiment library; they are not intended to run together
in one session. Each training notebook handles one model/arm, seed and condition.

Dataset preparation command, for reproducibility:

```powershell
python -m spectral_code.revision.prepare --source-zip ../data/benchmarks/codenet.zip --output ../data/benchmarks/codenet_revision_problem_disjoint
python -m spectral_code.revision.audit ../data/benchmarks/codenet_revision_problem_disjoint
python -m spectral_code.revision.alignment ../data/benchmarks/codenet_revision_problem_disjoint
python scripts/package_revision_dataset.py ../data/benchmarks/codenet_revision_problem_disjoint --output ../data/benchmarks/codenet_revision_problem_disjoint.zip
python scripts/build_revision_notebooks.py
```

The derived comparison bundles are prepared separately with
`python -m spectral_code.revision.datasets`. CodeNet starts from the audited
revision directory. AtCoder uses `--source-metadata-zip '../data/sources/base datasets.zip'`
to recover authoritative problem IDs. BigCloneBench retains original pair
labels, filters to endpoint/source-disjoint partitions, deduplicates unordered
pairs, and preserves each original split's label prevalence. These are derived
benchmark results, not scores on the unfiltered official split.
Run `python -m spectral_code.revision.audit <bundle> --by-split` for coverage on
the actual retained endpoints, separately by language and partition. Projected
DDG edge counts include the runtime's symmetrization/deduplication, so they are
not fractions of the directed exported edge count.

The preparation command requires an empty destination and preserves the old
release. Problems sharing submission IDs or exact source text are grouped
before the 70/15/15 split. New positive/negative pairs are formed inside each
partition, balanced by configuration and label. The prepared 100K release has
zero problem, submission, endpoint-ID and exact-source overlap across splits.
User-disjointness is unverified because user IDs are absent. Labels still use
problem identity as a proxy for functionality; no execution proof is claimed.
The exported graph corpus supplies the available program pool. Results do not
represent all original CodeNet submissions or eliminate parser-selection bias.
The available valid-AST pool contains 134,890 programs; 84,016 are referenced by
the generated 100K pairs. Pair generation samples eligible problems and then
endpoints within them, and removes repeated pairs. It does not impose the old
pool's submission-reuse cap: maximum endpoint reuse in this release is 26.
Report this new sampling policy explicitly when describing the revised benchmark.

The original Python/C# fallback DDGs use statement IDs independent of AST IDs.
The alignment command reproduces the source AST and requires exact exported
node-type/edge agreement and DDG statement-type agreement before assigning an
AST counterpart. Its sidecar repairs 31,296 Python and 28,894 C# records in this
release. At the 256-node cap, projectable Python dependency edges rise from
3,280 to 680,349 and C# edges from zero to 351,742. Source parse failures or
syntax mismatches remain unaligned; external input nodes remain unmapped.
`coverage_by_language.json` describes the original export, while
`ddg_alignment_audit.json` records this repair. Upload the alignment sidecar
along with the rest of the bundle; the training runtime verifies its checksum.

## Kaggle's 12-hour limit

100 epochs is a ceiling, not a required training duration. It also does not
mean that the entire suite runs in one session: each arm/seed is a separate job.
The first configuration cell now exposes `MAX_EPOCHS`, `MIN_EPOCHS` and
`PATIENCE`. Do not lower a schedule merely to claim convergence: if a run reaches
the cap while validation is still improving, report a fixed-budget result.

There is useful historical timing evidence, but not a guarantee for this version:
`../outputs/kaggle/d2/Section 6.2 effect of training duration full 100.zip`
contains a CodeNet result configured for 100 epochs with 25,229.6 recorded runtime
seconds (about 7.0 hours) on a Tesla T4. The archived history has 100 epochs.
The revised data, DDG alignment, model and runtime differ, so measure them again.

Start with the full-model seed-42 pilot. Every saved status now includes a
`runtime_forecast`, also printed in the notebook log, with observed seconds per
validated epoch, estimated additional epochs that fit this session, and whether
the remaining epoch cap is likely to fit. Estimates include a 15% planning
margin on top of the output reserve; they never change the schedule automatically.
The first few epochs are provisional estimates. Use several epochs and pilot
the slowest comparison models before freezing a common schedule across main
arms/seeds. Choose it using timing and validation only; keep test evaluation off.
If that common schedule cannot fit, either continue checkpoints across sessions
or use and explicitly report a smaller fixed budget. Timing alone cannot prove
that a shorter run has converged.

- Every graph notebook trains one arm and one seed.
- Default session budget: 10 hours, with 45 minutes reserved, leaving 9h15 for
  imports, input preparation, training and validation.
- Checkpoints are written every 10 minutes at optimizer boundaries, after
  training an epoch, after checkpoint selection, and on a graceful time stop.
- The checkpoint contains weights, optimizer, AMP scaler, RNG states, history,
  epoch and batch position. A continuation resumes the optimization trajectory.
- The maximum main schedule is 100 epochs, with validation patience 10 after
  at least 10 epochs. This can span several Save & Run sessions.
- A time stop is recorded as `incomplete_session_budget`. It is not a completed
  experiment or a final paper score. Reaching the maximum epoch limit is
  distinguished from validation-based stopping too.

To continue: save the output, attach it as an input to the next run, and set
`RESUME_CHECKPOINT` to its `progress.pt`. Keep the same arm, seed, dataset,
source snapshot and scientific configuration. The runtime rejects mismatches.
An abrupt external termination can lose updates since the most recent periodic
checkpoint; continuation restores that checkpoint consistently. The time guard
is preventive, not a promise about failures outside the kernel's control.

When the protocol and model choices are fixed, set `FINAL_TEST = True`. Test
metrics are released only after training is complete. Reusing the same completed
output returns its saved result. Do not use test scores to choose arms, losses,
training lengths, prompts, token limits or hyperparameters.

## What the controls change

| Arm | Intervention |
| --- | --- |
| full | Learned adjacency and density/heat/graph-signal spectral readout |
| mean_max | Same latent encoder; mean/max pooled states projected to 152 features |
| attention | Same latent encoder; attention-pooled states projected to 152 features |
| projected | Spectral pipeline on normalized projected observed adjacency, without learned attention affinities |
| projected_ast / projected_ddg | Vary the projected adjacency view while keeping encoder AST+DDG inputs and reconstruction targets fixed |
| density_heat | Eigenvalue density and heat traces only |
| energy_only | Attributed graph-signal spectral energies only |
| classification_only | Full architecture, classification objective only |
| mean_max_classification_only / attention_classification_only | Non-spectral readout and classification only; complement controls retaining graph losses |
| auxiliary_half / auxiliary_double | Joint sensitivity of all auxiliary weights at 0.5x and 2x |
| without_spectral_and_hard_negative | Joint removal to examine this loss interaction |
| without_* | Full architecture with the named auxiliary loss weight set to zero |

The non-spectral readout controls retain the common graph regularizers. They
isolate the classifier readout, not every use of spectral information in the
objective. Readout parameter counts differ and are reported explicitly. The
same symmetric block-comparison recipe and primary metric are used throughout.

The revision model also corrects masked slot normalization, computes occupancy
using the declared node cap, removes the unused embedding projection, and uses
uniform angular quadrature for Chebyshev coefficients. These changes are an
explicit experiment version change; old and new metrics are not interchangeable.

## Matched-budget transfer

`transfer_plan.csv` enumerates the existing 166 configurations. By default,
three continuation-capable notebooks are generated for the first length-1 pilot
`table8|J->P`. Internal table keys retain the repository's earlier numbering.
Generate selected configurations or the whole matrix with:

```powershell
python scripts/build_revision_notebooks.py --transfer-key 'table8|J->P' --transfer-key 'table8|P->J'
python scripts/build_revision_notebooks.py --all-transfer
```

Each condition receives exactly 6,000 training pairs, balanced within its
language/label buckets, and 30 epochs: 1,410 optimizer steps at batch 32 and
effective batch 128. Early stopping is disabled for this fixed-step control;
validation selects the best checkpoint within the shared optimization budget.
Runs fail rather than silently reduce the pair budget. Target-language exposure
is reported separately, because equal total data does not equalize that exposure.
These experiments hold out a pair configuration and permit target-language code
and labels in training. They do not test a completely unseen target language.

`06_transfer_baselines` now supplies the same data/optimizer-step schedule for
ASTNN, RtvNN, CDLH, DeepSim, FA-AST+GGNN/GMN, CodeBERT, the C4 adaptation and
Qwen LoRA. All 166 configurations passed the 6,000-pair availability check on
the prepared common CodeNet support. Only a starter configuration is generated
by default. Generate the other conditions before reporting the complete matrix.
Physical batches differ by model; gradient accumulation does not create the
same in-batch negatives as a larger physical contrastive batch.

## Modern encoder baseline

`04_modern_baseline/qwen3_frozen_encoder_and_probe.ipynb` uses the publisher's
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) loading and
last-token pooling approach. Extraction is frozen, uses the same code pairs,
records truncation at 1,024 tokens and pins the resolved model commit. Its cache
resumes across sessions. Reuse those embeddings across probe seeds.

Run `PHASE="encode"`, then reattach the completed cache and use `PHASE="probe"`.
The supervised logistic probe scales features using training data only and
selects thresholds/checkpoints on validation. It also reports a cosine baseline
at final test release. Probe checkpoints resume across sessions. Requires
`transformers>=4.51` and access to the model weights; graph notebooks need no
network or additional Hugging Face packages.

`05_baselines` additionally includes a fully fine-tuned CodeBERT Siamese model,
a task-masked C4-inspired contrastive CodeBERT adaptation, and a trainable
Qwen3-Embedding-0.6B LoRA model with classification/pair-contrastive learning.
Their tokenizer caches and training checkpoints resume. Main runs use the same
validation/test rules, and actual CUDA memory/runtime remain to be measured.
Optional dependencies are in `requirements-transformers.txt`; notebooks install
them when missing, leaving Kaggle's CUDA PyTorch installation in place.

C4 is explicitly an adaptation: it masks false negatives, uses known training
problem labels and both endpoint directions, and follows this project's
selection and optimization protocol. Its extra training problem-label access
must be reported. See the [authors' implementation](https://github.com/Chenning-Tao/C4).
Qwen pooling follows its [publisher model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
and adapters use [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora).
No SOTA result, exact ZC3/FSD-CLCD reproduction, or GraphCodeBERT DFG-model result
is claimed. C4 jobs are omitted for BigCloneBench because its problem IDs are
unavailable; do not fabricate them from test clone labels.

`07_cross_benchmark` provides new SPECTRA and baseline jobs for the audited
AtCoder and BigCloneBench supports. The latter remains a Java-only control.

## Reporting

Completed graph runs report precision, recall, F1, accuracy, balanced accuracy,
macro-F1, AUROC, average precision (`AUPRC_AP`), MCC, confusion counts, per-language
scores, parameter counts, timing and provenance. Average precision is identified
explicitly rather than conflated with trapezoidal PR area.

```powershell
python -m spectral_code.revision.report <downloaded-completed-output-root> --output output/revision_results
python scripts/inventory_revision_results.py
```

The aggregator excludes incomplete runs and incomplete seed groups from the
summary, rejects duplicate identities and mixed implementation versions, and
reports Student-t intervals across seeds. These intervals do not make shared
transfer paths statistically independent. `08_analysis` now also supplies
validation-only spectral/slot diagnostics and paired uncertainty analysis
sharing problem weights across methods, seeds and transfer paths. The latter
is an exploratory multi-membership bootstrap, not an independent-path t-test
or a guarantee of generalization to a population of programming languages.

Training jobs optionally produce `benchmark.json` with compute timing on the
same seeded 256 training pairs, inference and forward/backward costs, memory,
parameter counts and physical batches. It excludes loading/transfers/optimizer
steps; end-to-end training/validation wall time is reported separately.

`spectral_code.revision.clone_types.report_clone_types` reports clone-type
metrics from a supplied CSV containing `pair_id,clone_type,annotation_source`.
The local BigCloneBench source has no authoritative type annotations. Type-IV
evaluation therefore remains unavailable until that metadata is supplied.

Parser success cannot be reconstructed from graph exports alone. The coverage
audit reports available AST backbones, exported spectral status, unknown node
types, truncation and DDG projection retention by language. Empty DDGs are not
automatically labeled extraction failures.

GPU runs, complete transfer coverage, authoritative clone-type metadata,
exact specialized-method reproductions, and the final manuscript rewrite remain.
See `MANUSCRIPT_CORRECTIONS.md` for corrections already justified by the audit.
