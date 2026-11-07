# Manuscript corrections supported before new experiments

The workspace contains the submitted PDF but no editable manuscript source.
These are replacement specifications and draft passages for the source document.
New experiment outcomes are intentionally left unclaimed.

## Prediction and similarity definitions

For the submitted implementation, replace Equation 22 and Figure 2's projected-
embedding concatenation with the actual symmetric descriptor comparison.
The three descriptor blocks have sizes 32 (density), 24 (heat), and 96 (energy).
For each side and block, let `u = log(1 + max(s_block, 0))` and `v = u / ||u||_2`.
Concatenate `|v_left-v_right|`, `v_left * v_right`, and `|u_left-u_right|`, plus
cosine, mean absolute difference and RMS difference for that block. This gives
`3*152 + 3*3 = 465` pair features. The classifier widths are 465 -> 384 -> 224 -> 1.
The pair probability is sigmoid of the classifier logit. Threshold that
probability at the validation-selected threshold. The projected fragment
embedding is not an input to the submitted classifier.

Spectral contrast and hard-negative mining use the arithmetic mean of normalized
block cosines. Equation 24 should not describe a single cosine over the raw
concatenated descriptor; Equation 26 should not describe embedding cosine.
Describe revised readout arms separately, including their actual dimensions and
parameter counts.

## Transfer claims and degradation

Suggested statement for the existing Table 6:

> SPECTRA-Siam obtains point-estimate accuracy above .50 on all six tested
> cross-language configurations. DeepSim obtains higher accuracy on three
> configurations and approximately chance or lower performance on three others.
> These results distinguish consistency across configurations from absolute
> target accuracy; uncertainty and disjoint-split controls are required for
> stronger generalization conclusions.

The .058 figure is `.685 - .626667`, comparing Table 6's within-language and
endpoint-only means. Attribute it to that experiment, not the 60 bridge paths.
Report the bridge-path accuracy comparisons separately. A smaller drop does not
establish better target accuracy. Correct the conclusion that .69 CodeNet
accuracy exceeds GraphCodeBERT+SNN, whose reported accuracy is .79.

Suggested supervision description:

> Endpoint-only transfer trains on same-language pairs from both endpoint
> languages. Bridge-assisted adaptation additionally uses labeled cross-language
> pairs involving the target language while withholding its target within-
> language pair configuration. These settings evaluate unseen pair
> configurations, not complete absence of target-language code or supervision.

## Original versus revised protocol

Disclose that the submitted CodeNet release used pair-level partitions. In the
released 100K pairs, 63.2133% of test pairs have at least one endpoint seen in
training and 23.2267% have both. The revised release instead groups problems,
shared submission IDs and exact duplicate source before splitting. Its verified
overlap is zero for these four identifiers. User-level independence remains
unverified. Problem identity remains a proxy for functional equivalence.

Existing saved SPECTRA-Siam CodeNet results used F1 selection after graph
filtering left 7,489 positive and 7,470 negative validation pairs. Some GNN runs
used accuracy on all 15K validation pairs. Do not relabel the old runs as
accuracy-selected. The revised implementation fixes selection explicitly.

Disclose the 50K training/15K test profiles for CDLH and FA-AST wherever used;
they are not the same budget/support as full-data methods. Resolve AtCoder
FA-AST .82/.61 manuscript accuracies against the inspected archives' .50/.51
values by finding the exact source runs or replacing the unsupported entries.

Use validation trajectories for model/epoch/hyperparameter selection. State the
exploratory exception for old per-epoch test analyses and do not use a peak test
epoch as the new selected result. Revised training continues across Kaggle
sessions; a session limit is not a four-epoch scientific stopping criterion.

## Baseline representations

The RQ1 implementation uses 128 eigenvalues plus eight statistics: 136-D code
vectors and 274-D pair features. Correct the conflicting 64/72/146 dimensions.
The no-training comparator is PSS: remove padding, interpolate spectrum lengths,
L2-normalize, then apply the exponential distance kernel. It is not
`1/(1 + distance)` over statistics plus eigenvalues. Generate numerical
configuration/parameter tables from the actual run manifests.

## Slot-update algorithm and complexity

At each assignment iteration, normalize input/slot states; compute learned keys,
values and queries; softmax query-key scores over slots; mask invalid nodes;
normalize each slot's contributions over valid input nodes; aggregate values;
update slots with a GRU; and add the residual MLP update. Reconstruct input
states with the node-to-slot assignments and latent states. The revised model
excludes padding from both assignment normalization and reconstruction.

Using dense input relations, n input nodes, m slots, hidden width d, R relations,
I assignment iterations, and K Chebyshev order, describe the principal costs:

- Structural encoding: per layer approximately `O(R*n*d^2 + R*n^2*d)`.
- Assignment: approximately `O(I*((n+m)*d^2 + n*m*d))`.
- Dense projected prior: approximately `O(n^2*m + n*m^2)`.
- Latent affinity/refinement: approximately `O(m*d^2 + m^2*d)` per layer.
- Dense symmetric eigensolver: `O(m^3)`.
- Graph-signal filtering: approximately `O(K*m^2*N_sig)` plus band aggregation.
- Dense input adjacency memory: `O(R*n^2)` per example; batching multiplies it
  by the physical batch size.

Identify these as dense-implementation bounds, not a proof of generalization.
Measure actual latency/throughput against baselines on the same hardware.

## Editorial and evidential corrections

- Include Section 7 in the roadmap.
- Resolve Figure 7's .128 versus .328 clone-distance discrepancy from its source
  artifact; do not guess the replacement.
- Explain BigCloneBench as a single-language control, not cross-language evidence.
- Move large transfer matrices to supplementary material.
- State that existing component techniques are established; identify the new
  contribution precisely and support it with matched controls.
- Add relevant modern/specialized comparisons. A frozen Qwen3 encoder is now
  runnable but does not replace every requested fine-tuned/specialized baseline.
- Do not claim an invariance guarantee solely from CCD supervision. State the
  property tested empirically and its observed limits after new results exist.
- Add graph coverage, truncation, unknown-type, and available extraction-status
  statistics by language, distinguishing empty dependencies from failures.
