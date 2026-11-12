# Stage 2: architecture and loss screening, seed 42

**The five central controls are now verified complete**; together with the
input-v2 full model they form the completed seed-42 six-cell core. Reuse them.
[Results and interpretation](../results/CORE_CONTROLS_REVIEW.md).
All 12 core repeats are verified complete, including the seed-123 fresh restart.
**All five additional graph/descriptor controls are [verified complete](../results/ARCHITECTURE_CONTROLS_REVIEW.md)**:
projected AST+DDG, projected AST, projected DDG, energy-only and density/heat-only.
**All 20 CodeNet seed42 controls are now complete.** The latest `without_graph`, `auxiliary_half` and `auxiliary_double` outputs passed all 108 checks. No further control launch or continuation is needed. Consult [the current results](../results/README.md) and `../results/control_progress.csv`. The [independent queue](../queues/RUN_NEXT_INDEPENDENT_BATCH.md) now has zero fresh launches; all full-repeat identities also have completed evidence.

Use the **updated input-v2 notebooks** in [the current run list](../queues/RUN_NOW_INPUT_V2.md).
CodeNet/full/42 with that corrected input is now verified complete; reuse it.
The older instruction below to reuse the old full model is superseded by
[the input fix](../00_plan/INPUT_V2_FIX.md); all old results remain historical evidence.

2026-11-09 priority: [the new validation review](../results/BATCH_2026-11-09.md)
identifies five central CodeNet controls to run first: mean/max, attention,
classification-only full, classification-only mean/max and classification-only
attention. These separate readout and auxiliary objectives using the same
encoder settings. This ordering retains the full planned control set below.

After the baseline stage, run the 20 CodeNet controls in
[codenet/architecture](codenet/architecture) and [codenet/losses](codenet/losses).
Together with the corrected input-v2 full model, they form the 21-arm study.

These comparisons address reviewer 6's request to distinguish spectral readout
from the shared encoder and auxiliary objectives. Classification-only siblings,
projected graphs, descriptor components and loss/weight controls separate those
questions. See [the protocol](../00_plan/EXPERIMENT_PROTOCOL.md).

The four files under `optional_cross_benchmark/` are candidate mean/max and
attention controls on AtCoder/BigCloneBench. Select the planned sibling before
running its cross-benchmark confirmation; these four are not additional baseline
rows and are not all mandatory runs.
