# SPECTRA reviewer-revision master plan

Current execution: [the six-condition CodeNet core is verified complete for seed 42](../results/CORE_CONTROLS_REVIEW.md).
The five new controls passed 175 checks. Bring the already planned
[12 CodeNet core repetitions](../queues/RUN_NEXT_CORE_REPEATS.md) forward:
same six conditions, seeds 123 and 231. The remaining 15 seed-42 controls stay
in the roadmap. Reuse the corrected full models and completed external
baselines; old full-model ZIPs remain archived. Earlier execution updates
below predate these results.

Execution update, 2026-11-09: [the combined validation review](../results/BATCH_2026-11-09.md)
verifies 17 completed baseline jobs across all three datasets; four CodeNet jobs
are paused and eight jobs lack submitted results. The original planning scope
below is retained. SPECTRA trails several baselines; prioritize input diagnostics
and central readout/objective controls before committing to all repeated seeds
and transfer conditions. These are validation findings with final tests off.

Execution update, 2026-11-07: current folders and the 24-job launch plan are in
[START_HERE_FA.md](../START_HERE_FA.md). The user reported CodeNet/full/seed42
complete at epoch 43 on T4 with test evaluation off; its full status file is
pending. Reuse that run. The original planning checklist below is retained for
scope and rationale; implemented items are detailed in the execution guide and
protocol rather than inferred from historical unchecked boxes.

Execution update, 2026-11-06: scope/default roster and initial settings are now
written in `revision/00_plan/EXPERIMENT_PROTOCOL.md` and its JSON.
The source-only control, explicit focused train/validation quotas, distinct
transfer readout IDs, validation-only collector and training dependency hash
have been implemented and locally checked. The 29-job seed-42 queue is prepared.
Start with `revision/START_HERE_FA.md`; most Kaggle hardware checks
and research results remain outstanding. The checklist below records the
original roadmap, not a claim that every reviewer issue is already resolved.

Planning document, 2026-11-06. This turn is planning only; no experiments are
started while preparing this roadmap. The agreed scope is **CodeNet,
AtCoder and BigCloneBench**, with a separate 12-hour maximum for each Kaggle
session. This roadmap supersedes earlier scheduling suggestions. Existing
notebooks, dataset archives and the submitted PDF were not changed for this plan.

The goal is to provide evidence for defensible revised claims. Code readiness,
completed experiments and corrected manuscript claims are different milestones.
An experiment that contradicts the original claim is a finding to report, not
a reviewer objection that can be removed by running more favorable settings.

## Agreed execution strategy

1. Finish protocol and code preparation before spending heavily on GPU runs.
2. Screen the full agreed baseline roster on all three datasets with **seed 42**.
3. Run the architecture/loss study on CodeNet with seed 42.
4. Add seeds 123 and 231 only to the comparisons supporting central conclusions.
5. Run a small, prespecified matched-transfer study with selected models.
6. Release final tests after design/model/baseline/path choices are frozen.
7. Rewrite the manuscript and build the response-to-reviewers matrix.

Successful seed-42 runs are reused when their final settings/data/version are
unchanged. Adding two seeds does not mean repeating seed 42. Three seeds are a
practical initial replication budget; reviewer #2 requests multiple seeds for
key experiments, not exactly three seeds for every model in every table.

## Phase 0 — Freeze scope, questions and baseline roster

**Compute:** no research training. **Exit:** one written experiment specification.

- [ ] Retain three scientific questions: spectral contribution under controlled
  architecture; detection on disjoint benchmark support; and carefully defined
  transfer/adaptation. BigCloneBench is a Java-only semantic detection control.
- [ ] Decide whether the revised manuscript will present a focused transfer
  case study or retain the original broad 60-path empirical coverage. Default:
  focused study first; broader coverage is conditional on evidence and budget.
- [ ] Freeze the baseline roster before looking at new test scores. The current
  ready roster is six legacy graph adaptations, CodeBERT, C4-adapted and Qwen LoRA.
- [ ] Decide what happens to original GraphCodeBERT, plain GNN/SNN, Deckard and
  fixed-spectrum rows: implement/re-evaluate on this protocol if retaining them,
  or remove/replace them with an explanation. Their old values cannot be copied
  into a new comparison. Deckard/fixed-spectrum evaluation is a separate, mostly
  CPU-cost item; it is not included in the neural-run count below.
- [ ] Conduct a feasibility check for a faithful recent specialized CLCD method.
  ZC3 has public author code but different task/supervision assumptions. Exact
  FSD-CLCD is specifically requested by R3 and is not supplied in the current
  package. A C4 adaptation does not automatically satisfy that specific request.
  Either add a verifiable implementation under a declared protocol, or document
  the empirical limitation and narrow the claims. Do not invent method names.
- [ ] Fix the primary selection metric: CodeNet/AtCoder accuracy; BigCloneBench
  F1. Fix threshold selection on validation and report the broader metric set.
- [ ] Prespecify tuning effort, early stopping, resource ceilings, handling of
  incomplete runs, and which results will receive multi-seed confirmation.

**Reviewers:** R1.1–3; R2 baseline relevance/protocol; R3.1/6; R4.1/4; R5.4/5; R7.1.

## Phase 1 — Close preparation gaps and freeze a reproducible release

**Compute:** local checks and short hardware checks. **Exit:** a frozen run-ready
package with all planned conditions expressible before broad screening starts.

Already available:

- CodeNet/AtCoder problem-, endpoint-, submission- and exact-source-disjoint
  derived supports; BigCloneBench endpoint/submission/source-disjoint support.
- Common graph support, unordered-pair deduplication, per-language/split coverage,
  DDG alignment sidecars, explicit selection metrics, checkpoints and resume.
- Symmetric comparison functions and local numerical tests; broader metrics;
  architecture/loss controls and baseline training adapters.

Still to prepare or verify before large runs:

- [ ] Add a **source-only A-A training / B-B test** condition. The existing
  `table7_cross|A-B` condition trains on A-A and B-B and tests A-B; it is a
  different question and is not this source-only control.
- [ ] Add explicit per-bucket train and validation quotas for the exposure-matched
  transfer design in Phase 7. Current matching balances buckets equally; it
  matches total training pairs/steps but not target-language label exposure
  across different path lengths, and validation budgets currently grow with
  the number of buckets.
- [ ] Allow the selected non-spectral SPECTRA sibling in transfer with its own
  model/arm ID and output directory. Current transfer templates identify their
  SPECTRA arm as full; relabeling a different arm as full is not acceptable.
- [ ] Check learned/projected AST+DDG, AST and DDG controls against the exact
  attribution being claimed. Attribute-matched CFG/CPG controls require their
  real graph views; the next-token channel must not be called CFG. Implement
  these if retaining a claim covering all graph families, otherwise narrow it.
- [ ] Provide a validation-only screening summary and a run ledger. It must not
  require final-test release to rank the first-stage candidates.
- [ ] Make tuning/output IDs distinguish configurations; keep one authoritative
  final run per model/dataset/seed. Do not aggregate tuning candidates as seeds.
- [ ] Resolve any intended additional baselines and annotation-dependent work.
- [ ] Make analysis-only updates independent of training-version fingerprints,
  or finish all such utilities before freezing. Current whole-package source
  hashing is conservative: changes can invalidate checkpoint continuation.
  Model/data/training changes must still invalidate continuation.
- [ ] Verify a test-only evaluation workflow that reuses completed checkpoints
  and token caches. Optional batching of several completed models can reduce
  the number of separate evaluation sessions without mixing training runs.
- [ ] Check actual GPU forward/backward, memory, dependency/model downloads,
  checkpoint save/reload and the time guard on Kaggle. CPU tests do not verify
  real pretrained GPU execution.
- [ ] Freeze dataset ZIP checksums, training implementation, model commits and
  configuration schema. Rebuild the notebook manifest after preparation changes.

Data qualifications to record:

- User IDs and historical parser success logs are unavailable. AST availability
  and empty DDG exports are not equivalent to original parser failure rates.
  If needed, perform a separately labeled logged re-extraction audit on a seeded
  source sample; do not silently replace training graphs with a new parser version.
- BigCloneBench has no authoritative problem/clone-type labels in the local
  source. Its derived test pool has 7,566 pairs but only 128 distinct programs;
  validation has 7,129 pairs and 127 programs. Report this limited coverage.
- Current common-support pair counts are CodeNet 69,538/14,896/14,906;
  AtCoder 70,000/15,000/15,000; BigCloneBench 70,000/7,129/7,566 (train/valid/test).
  Preserve each original split's label prevalence, including BCB's imbalanced
  validation/test sets. Do not describe all three datasets as problem-disjoint.

**Reviewers:** R2 leakage, reproducibility, symmetry, fairness; R3.5; R6.2/4; R7.2/3.

## Phase 2 — First Kaggle pilot

**Compute:** start with one model, not the whole queue. **Exit:** a realistic
runtime/memory estimate and a proven continuation path.

- [ ] Start the CodeNet full-model seed-42 job on the final prepared release.
- [ ] Keep `FINAL_TEST=False`. Observe several completed validation epochs,
  runtime forecasts, finite losses and checkpoint behavior.
- [ ] Check one tree baseline and one pretrained baseline for memory/runtime.
- [ ] Freeze physical batch sizes and pretrained model commits before broad
  screening. Accumulation does not produce the same in-batch negatives as a
  larger physical batch.
- [ ] Treat successful research-protocol pilot runs as part of Phase 3 if their
  settings remain unchanged. Tiny smoke tests are not research scores.

The 100-epoch value is a ceiling. Main comparisons use validation stopping;
models still improving at a resource cap are reported as capped, not converged.
Each session is configured to stop around 9h15 for work and preserve output
before the user's 12-hour limit. A training job can require several sessions.

## Phase 3 — Broad baseline screening: one seed first

**Seed:** 42 only. **Test:** off. **Exit:** completed validation results for the
agreed roster and prespecified selection of finalists.

| Dataset | Ready external baselines | SPECTRA full | Initial research jobs |
| --- | ---: | ---: | ---: |
| CodeNet | 9 | 1 | 10 |
| AtCoder | 9 | 1 | 10 |
| BigCloneBench | 8 | 1 | 9 |
| Total | 26 | 3 | **29** |

The six legacy families are ASTNN, RtvNN, CDLH, DeepSim, FA-AST+GGNN and
FA-AST+GMN. The text baselines are CodeBERT, C4-adapted and Qwen LoRA. C4-adapted
is omitted on BigCloneBench because its task-label objective needs unavailable
problem IDs. A separately cached frozen Qwen comparison is optional.

- [ ] Finish or explicitly mark each training job incomplete. No four-epoch
  truncation or arbitrary test-pair exclusion is a completed new comparison.
- [ ] Inspect validation trajectories; fix a broken implementation before using
  its weak result as a baseline. Record failures rather than disguising them.
- [ ] Use the prespecified validation criterion to select strong finalists.
  Include a relevant modern/specialized comparator, not just convenient weak
  methods. A main-benchmark winner is not automatically the transfer winner;
  retain relevant candidates such as RtvNN/DeepSim in that decision.
- [ ] Publish the agreed one-seed baseline table after final evaluation, clearly
  labeled with seed count. Report multi-seed evidence separately for key models.

The 29-job count excludes added roster items, failed runs, tuning candidates,
continuation sessions and final test-only evaluation.

**Reviewers:** R1.2; R2 convergence/recent baselines; R3.6; R4.4; R5.3/5; R7.1/3.

## Phase 4 — Isolate the spectral contribution on CodeNet

**First pass:** seed 42; validation only. **Exit:** evidence about what spectral
readout, topology and auxiliary objectives contribute separately.

Run the core six-way comparison:

| Architecture/readout | Full auxiliary objective | Classification only |
| --- | --- | --- |
| Spectral | `full` | `classification_only` |
| Mean/max | `mean_max` | `mean_max_classification_only` |
| Attention | `attention` | `attention_classification_only` |

Then run projected AST+DDG adjacency, projected AST, projected DDG, density/heat
only, and graph-signal energy only. The full model is reused from Phase 3.
Record exact parameter counts and remaining differences; matching descriptor
dimensions alone does not prove identical model capacity.

For a contribution claimed beyond CodeNet, replicate the selected non-spectral
comparison on AtCoder and the Java-only BCB control. These are not all a claim
of cross-language generalization: state each dataset's setting.

**Decision gate:** if the spectral readout has no reproducible advantage in the
relevant setting, revise the contribution or model. Do not proceed with a large
transfer matrix merely to search for favorable examples.

**Reviewers:** R2 spectral attribution; R3.5; R6.2; R7.2.

## Phase 5 — Justify or simplify the loss

**First pass:** seed 42 on CodeNet; use validation for weight/model choices.

- [ ] Compare full objective with classification only (reuse Phase 4).
- [ ] Remove each of the seven auxiliary terms in turn: spectral contrastive,
  AUC ranking, hard negatives, representation reconstruction, topology
  reconstruction, spectral variance and graph regularization.
- [ ] Check the joint removal of spectral contrastive and hard-negative terms.
- [ ] Check half/double global auxiliary weighting. Describe this as joint
  sensitivity, not exhaustive sensitivity for every individual coefficient.
- [ ] Repeat the losses supporting a central necessity claim across seeds;
  initially designate spectral contrastive, ranking and hard-negative removal
  as key comparisons. Label the remaining one-seed analyses exploratory.
- [ ] If auxiliary terms are unnecessary or harmful, simplify the model and
  update the mathematical description. Prefer final settings already supported
  by the frozen implementation; substantive code changes require a new version.

Across Phases 4–5 the current library contains 21 CodeNet arms. Running all of
them once is **20 additional jobs**, because full/seed42 is already available.
This is not a requirement to run all 21 arms with all three seeds. If a reduced
ablation scope is chosen before testing, narrow the corresponding claims.

**Reviewers:** R2 objective attribution; R6.5; R7.2.

## Phase 6 — Confirm the comparisons that support the paper

**Seeds:** add 123 and 231 to successful seed-42 jobs. **Exit:** consistent main
comparisons with uncertainty and an explicitly frozen final configuration.

- [ ] Repeat the six core architecture/objective arms on CodeNet.
- [ ] Repeat SPECTRA and one or two selected external baselines per dataset.
- [ ] Repeat the predesignated key loss comparisons.
- [ ] Repeat the selected cross-benchmark non-spectral check if making the
  contribution claim across datasets.
- [ ] Keep architecture, hyperparameters, source snapshot, data and pretrained
  model commits fixed across seeds. Keep model-selection decisions on validation.
- [ ] Assess uncertainty before claiming robust superiority. Three seeds are
  a starting point; close/unstable comparisons may require more or weaker claims.

With two external finalists per dataset, repeating the six core CodeNet arms
and SPECTRA/finalists on all three datasets costs **28 new jobs after deduplication**:
12 for the six CodeNet arms and 16 remaining model/dataset repetitions. Key
loss replications and cross-benchmark pooling replications are additional.
This count assumes the selected final SPECTRA configuration is one of the six
core arms; otherwise its additional seed runs must also be included.

**Reviewers:** R2 training randomness; R6.1/2/3/5.

## Phase 7 — Freeze a smaller, interpretable transfer study

**Default:** one prespecified source A and target B, with intermediate languages
C and D chosen before test scores. All conditions evaluate the SAME B-B test
pairs. This is a focused case study, not a claim about every language/path.

| Condition | Training pairs (illustrative exact quotas) | Target task supervision |
| --- | --- | --- |
| Source-only | A-A: 6,000 | None during task training |
| Direct/one hop | A-A: 4,500; A-B: 1,500 | Labeled cross-language target pairs |
| Two hops | A-A: 2,250; A-C: 2,250; C-B: 1,500 | Same number of target-containing training pairs |
| Three hops | A-A: 1,500; A-C: 1,500; C-D: 1,500; D-B: 1,500 | Same number of target-containing training pairs |
| Within-target reference | B-B: 6,000 | Target within-language labels |

Balance positive/negative labels within each training bucket. Match all five
conditions at 6,000 training pairs and, for the fixed-step control, 30 epochs
and 1,410 optimizer updates at effective batch 128. Validation selects within
that budget; this does not claim that every model has converged.

Also fix validation at an illustrative **1,200 pairs** per condition. Across
the three bridge conditions allocate 300 target-containing validation pairs:
direct 900/300; two-hop 450/450/300; three-hop 300/300/300/300. Source-only and
within-target references use 1,200 of their own permitted bucket. Confirm
availability before freezing; preserve balance and separation from training.
The source-only validation set contains no target-task labels.

Source-only and within-target references deliberately differ in supervision;
they are contextual comparisons, not exposure-matched bridge contrasts.
Matching total/target-exposure budgets still varies language/data composition.
Report that intervention rather than attributing every difference causally
to abstract path structure. Pretraining may already expose transformer models
to target-language syntax: say no target-task supervision, not never-seen language.

Choose a compact model roster:

- SPECTRA and its strongest matched non-spectral sibling.
- One strong relevant external baseline for a cost-limited focused study.
- Add a second external baseline, including a relevant modern/specialized
  comparator when needed, for stronger breadth. Do not imply omitted methods
  were beaten on transfer.

**Reviewers:** R2 bridge confounds/target supervision; R3 attribution/recent CLCD;
R6.2/3/4; R7.2.

## Phase 8 — Transfer pilot, replication and optional breadth

- [ ] Run the five conditions with seed 42 first: **15 jobs for three models**,
  or **20 for four models**. Review validity, validation, supervision budgets,
  runtime and coverage without choosing favorable target-test paths.
- [ ] Add seeds 123/231 for this frozen study: 30 or 40 additional jobs.
  Thus totals are **45** or **60** independent training jobs, including seed 42.
- [ ] If retaining intermediate-language reinforcement claims, add a prespecified
  reinforced condition while preserving total/target training and validation
  exposure. That adds 9 or 12 jobs across three seeds, respectively.
- [ ] If retaining the original held-out mixed-pair question (A-A+B-B train,
  A-B test), run its prespecified configurations separately. It is not the
  B-B bridge target and cannot be substituted for the source-only row above.
- [ ] Expand to further source-target directions/path choices only under a
  prespecified extension rule. A one-direction result supports that direction.
- [ ] Repeat all 60 original paths only if retaining broad empirical coverage.
  A broad one-seed descriptive table can coexist with a prespecified multi-seed
  core, but cannot support a claim of multi-seed stability across all 60 paths.
  Full inferential claims require the appropriate completed replication/support.
- [ ] The 166-configuration reinforcement matrix is an optional extension,
  not the default prerequisite for a narrower revised paper.

These are training-job counts, not guaranteed Save & Run counts or GPU hours.
Continuation and test-only sessions are additional; use measured pilot time.

## Phase 9 — Final evaluation and uncertainty

**Entry:** final methods, losses, hyperparameters, baseline shortlist, paths,
threshold procedure and analysis rules are frozen. **Exit:** traceable results.

- [ ] Release final test predictions from completed checkpoints; cache results.
  Do not retrain merely to switch `FINAL_TEST` on. Reuse token/embedding caches.
- [ ] Report the agreed broad baseline set even when it defeats SPECTRA. Clearly
  distinguish one-seed point estimates from multi-seed means and intervals.
- [ ] Report P/R/F1, accuracy, balanced accuracy, macro-F1, AUROC, AP and MCC,
  confusion counts, support counts and language-pair breakdowns as applicable.
- [ ] Call AP average precision explicitly. Do not silently call it trapezoidal
  AUPRC. Keep the same validation threshold rule across small balance changes.
- [ ] Use paired shared-problem resampling for CodeNet/AtCoder comparisons;
  share weights across methods/paths. Shared paths are not independent samples.
- [ ] If adding BCB clustered uncertainty, use program/function clusters explicitly;
  do not fabricate problem IDs. Current problem-based analysis rejects missing IDs.
- [ ] Obtain authoritative clone-type annotations if possible and run type-wise
  BCB evaluation. If unavailable, disclose that Type-IV-specific evidence remains
  absent. Do not infer semantic clone types from lexical similarity.
- [ ] Distinguish absolute accuracy from degradation relative to a within-target
  reference. Compare only compatible targets/supports/budgets and state the
  reference formula. The reference is not a theoretical upper bound.
- [ ] Treat old repeated test-set analyses as historical exploration. Do not
  describe them as having used an untouched test set; the new freeze is prospective.

## Phase 10 — Diagnostics, cost and explanation

**Compute:** mostly existing checkpoints and validation data, not retraining.
Validation diagnostics may begin during Phase 4. If they influence model or
protocol choices, complete that decision process before Phase 9's test freeze.
Post-test diagnostics can explain findings, but cannot reopen test-driven tuning.

- [ ] Generate equivalent/non-equivalent pair distance distributions with the
  precise distance formula, graph-size/density strata and support counts.
- [ ] Regenerate Figure 1 with all three spectra, including C, and descriptive
  graph-size/density information. Do not infer universal chance performance
  from one example.
- [ ] Export validation latent spectra and marginal cross-language slot profiles;
  use population summaries and clearly labeled examples, including limitations.
- [ ] If claiming semantic slot alignment, add a direct quantitative analysis;
  marginal language profiles alone do not establish semantic correspondence.
- [ ] Resolve Figure 7's .128/.328 discrepancy from its source or regenerate it.
- [ ] Measure inference and forward/backward cost, memory and parameters on
  matched hardware/support; report physical batch sizes and token/node limits.
  Current 256-pair timing excludes loading, transfers and optimizer steps.
  Report setup and training/validation wall time separately.
- [ ] Add algorithm pseudocode, GRU/residual/MLP slot updates, notation and
  implementation-appropriate time/memory complexity.
- [ ] Explain permutation invariance separately from language invariance. Add
  a valid rationale/conditional stability argument if defensible; do not claim
  a theorem of functional/language generalization from the CCD loss alone.

**Reviewers:** R2 reproducibility; R3.1–5/7/complexity; R4.6; R5.1–3.

## Phase 11 — Rewrite and verify the manuscript

**Dependency:** recover the editable LaTeX/Word source; currently the workspace
has the submitted PDF and correction notes, not editable manuscript source.

- [ ] Reframe motivation around a precise testable contribution and limitations
  of prior directions. A latent graph is learned from parser-derived inputs;
  do not describe the method as discarding AST/DDG information.
- [ ] Position existing pooling/attention/spectral ingredients accurately and
  identify the incremental contribution supported by the ablations.
- [ ] Verify and incorporate relevant requested/recent related work; explain
  which baselines were reproduced/adapted and which could not be evaluated.
- [ ] Replace every affected old result with compatible new evidence, or label
  historical protocols separately. Do not mix old/new pipeline scores in a ranking.
- [ ] Correct the 465-feature symmetric head, loss similarity definitions,
  conflicting representation dimensions and the actual PSS formula.
- [ ] Replace unsupported AtCoder FA-AST numbers unless their exact source is
  recovered. Correct the claim that .69 beats .79 and the attribution of .058.
- [ ] Address R6 directly: DeepSim's wins, stability versus absolute detection
  accuracy, actual target supervision, matched budgets and loss necessity.
- [ ] Correct the Section 7 roadmap omission; add missing symbol definitions.
- [ ] Explain BCB's monolingual purpose and the new derived-support boundaries.
- [ ] Move dense transfer matrices/nonessential material to supplementary files;
  place tables near their discussion; shorten and edit the English.
- [ ] Add strengths, weaknesses, threats and future work. State unavailable user
  IDs, parser logs, clone types, limited BCB coverage and limited transfer directions.
- [ ] Render the final PDF and audit every equation, figure, table, caption,
  number, cross-reference and conclusion against the authoritative run ledger.

**Reviewers:** all seven; especially R1.1/3–5, R3.7/roadmap, R4.1–6, R5.1–5, R6.1–5.

## Phase 12 — Assemble the review/resubmission package

- [ ] Complete a response matrix: reviewer comment -> change -> exact section,
  experiment/figure -> evidence -> remaining limitation.
- [ ] Publish no superiority/invariance claim stronger than the new evidence.
- [ ] Archive dataset/model/source checksums, immutable configurations, seeds,
  model commits, software/hardware versions, histories and prediction files.
- [ ] Keep main results, exploratory runs, tuning candidates and failed runs
  distinguishable. Provide a reproducible minimal run path.
- [ ] Reassess the destination journal after the revised evidence and scope are
  known. Planning experiments does not imply acceptance or authorize submission.

## Run ledger and step-by-step handoff

For every research run retain: dataset and support checksum; model/arm; seed;
training implementation and pretrained commit; LR; physical/effective batches;
caps/patience; pair counts; best epoch and validation metric; stop reason;
optimizer steps; wall time/GPU; checkpoint/output path; and test-release state.

After each stage, review that stage's outputs before expanding the queue.
An incomplete checkpoint should be continued, not entered as a finished score.
Changes to the scientific protocol trigger a new documented version; analysis
and presentation work should reuse valid completed training whenever possible.

The first execution action after this planning turn is to close Phase 1's
readiness items and freeze the release. Then run the single CodeNet full-model
pilot, followed by the one-seed, three-dataset baseline screening. No Kaggle
training or code edits were performed as part of creating this plan.
