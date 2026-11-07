# Reviewer concerns and implementation coverage

This is a code-readiness map, not a claim that the reviewer concerns have been
empirically resolved. All new performance conclusions await Kaggle execution.

| Concern | Supplied code/artifact | Still required |
| --- | --- | --- |
| R2: CodeNet leakage | Problem/submission/source-disjoint revision; common-support bundle; duplicate-pair checks | Rerun comparisons; disclose unavailable user IDs |
| R2: AtCoder split clarity | Authoritative problem IDs recovered from source archive; zero-overlap audit | New results and benchmark description |
| R2: BigCloneBench leakage/imbalance | Derived endpoint/source-disjoint support; unordered-pair deduplication; original split prevalence retained | Report limited surviving program pool and distinguish derived vs official scores |
| R2/R6.4: target supervision | Explicit transfer metadata and supervision description | Correct the manuscript's unseen-language claims |
| R2/R6.4: extra-data bridge confound | Exact matched 6,000-pair selection, 30 epochs/1,410 steps for SPECTRA and nine baseline adaptations | Run sufficient endpoint/bridge/path-length controls; report differing physical batches |
| R2/R3.5/R6.2/R7.2: spectral attribution | Same-encoder mean/max and attention, classification-only variants, projected AST+DDG/AST/DDG adjacency, density/heat and energy controls | Results determine which attribution is defensible; CFG/CPG attribute-matched controls are not supplied |
| R6.5: losses | Leave-one-out losses, classification only, combined spectral/hard-negative removal, half/double auxiliary weights | Run and interpret; joint scaling is not exhaustive individual-weight or interaction search |
| R2/R7.3: four epochs | New revision baseline jobs share validation stopping, explicit metric, configurable ceiling, checkpoint/resume | Check validation trajectories; resource-limited runs are not claimed converged |
| R1/R2/R3/R4/R5/R7: modern baselines | Fine-tuned CodeBERT, C4-inspired CLCD adaptation, frozen Qwen and Qwen LoRA | Actual GPU/model-access validation; exact FSD-CLCD and ZC3 reproductions are absent |
| R2: repeated test use | Validation-only training selection and diagnostics; final test opt-in and cache | Freeze design before releasing tests; old exploratory test curves remain historical |
| R2: pair order | Symmetric SPECTRA/baseline heads; numerical swap tests | Correct the stale concatenation formula in the manuscript |
| R2: uncertainty/dependent paths | Three-seed jobs; seed intervals; paired bootstrap with problem weights shared across methods and paths | Complete shared seed/condition sets; report exploratory assumptions and limited languages |
| R2: broader metrics | P/R/F1/accuracy/balanced accuracy/macro-F1/AUROC/AP/MCC, confusion counts | Final results; identify AP explicitly instead of silently calling it trapezoidal PR area |
| R2: Type-IV clones | Reporting function requiring authoritative pair/type/source annotations | Local source lacks official clone types; do not infer semantic type from lexical similarity |
| R2: parser/DDG/truncation | Per-language/split coverage, unknown nodes, truncation, raw/repaired DDG projection counts | Historical parser success/failure logs are unavailable; empty DDG is not labeled a failure |
| R3.2: Figure 1 anecdote | Validation clone/nonclone distance distributions and size/density-stratified summaries, missing-view counts | Replace/qualify figure claims from measured data; no universal fixed-spectrum chance claim |
| R3.3: cross-language slot/spectrum evidence | Validation descriptor/eigenvalue exports and per-language marginal slot profiles | Interpret cautiously; marginal profiles do not establish semantic slot equivalence |
| R3.1/4: theory/novelty | Joint latent-node permutation-invariance test; narrower empirical diagnostic framework | An actual language-generalization theorem is not supplied; do not claim one |
| R3/R4/R5: complexity/algorithm | Existing MANUSCRIPT_CORRECTIONS.md pseudocode and complexity discussion; measured cost hooks | Incorporate into editable manuscript and verify final layout |
| R5.3: comparative cost | Same seeded 256 training pairs; measured inference/forward-backward time, GPU memory and parameters; total training wall time | Run on stated hardware; not a true GraphCodeBERT DFG-pipeline benchmark |
| R6.1/3: overstated superiority | Existing manuscript corrections distinguish point estimates, degradation and absolute accuracy | Rewrite conclusions using new completed results; no superiority is presumed |
| R1/R4/R5: motivation, related work, language, length/layout | Audit and correction documents identify required editorial changes | Rewrite and render the manuscript after results; submitted PDF remains unchanged |

The new experiments are a different implementation/data version. They cannot
be inserted piecemeal into old tables while leaving the old comparison rows.

## Baseline provenance

- The six legacy graph families reuse this repository's documented portable
  adaptations (`spectral_code/faithful_graph_baselines`). They are not exact
  executions of all upstream preprocessing pipelines.
- [C4 authors' code](https://github.com/Chenning-Tao/C4): the new adaptation uses
  task-masked supervised contrastive learning on CodeBERT CLS vectors. It masks
  false negatives rather than repeating other tasks, uses both endpoint
  directions, and follows the shared revision optimizer/selection protocol.
  It has access to known training problem IDs. Do not label it an exact C4
  reproduction or a zero-shot ZC3 model.
- [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): last-token
  normalized pooling; the fine-tuned variant adds LoRA and a symmetric head.
  The frozen extraction uses 1,024 tokens; fine-tuned text models use 512 by
  default. Report their separate truncation/compute budgets.
- [ZC3 authors' implementation](https://github.com/lairikeqiA/ZC3) and
  [FSD-CLCD article](https://doi.org/10.1016/j.engappai.2024.108199) are distinct
  methods. Their names are not applied to substitute models in this package.
