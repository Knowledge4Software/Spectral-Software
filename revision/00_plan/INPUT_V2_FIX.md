# Java callee lexical-input fix — 2026-11-09

The existing graph exports already contain callee names such as `write`,
`readLine` and `openConnection` in `node_types`. Their `node_labels` often
contain line numbers. Java canonicalization recognizes these names as calls,
but the old lexical fallback retained raw text only for canonical unknowns.
Consequently these recognized calls lost their names in the lexical channel.

The loader now retains that callee text as fixed hashed lexical features,
alongside the same canonical `Call_Expr` category. Exact and parser-pattern
structural tags remain excluded; numeric line labels remain excluded. Actual
source-level node labels still take precedence. No train-fitted vocabulary or
source-text feature channel was added.

This is a loader fix, not a dataset-label or split repair. Existing dataset
ZIPs, program IDs, labels and train/validation/test partitions are unchanged.
Do not rebuild or reupload the datasets. The tiny independent support of the
derived BigCloneBench split remains a separate limitation: 127 validation
programs and 128 test programs.

On all 127 BigCloneBench validation graphs, nodes with nonempty lexical
features increased from **3 to 1,927 out of 17,111**. The 1,924 changed nodes
are recovered Java callee features. Read-only validation-input samples also
confirm the path affects Java in AtCoder and CodeNet. This verifies feature
retention, not a measured improvement in F1/Accuracy.

## Runs and provenance

The original input-fix batch comprised corrected SPECTRA full on all three
datasets plus five central CodeNet controls. The three full runs are now
[verified complete](../results/SPECTRA_INPUT_V2_REVIEW.md); reuse their outputs.
[The current run list](../queues/RUN_NOW_INPUT_V2.md) contains the five controls.
New controls start fresh with the same original training settings. The new
SPECTRA output folder names end in `_input_v2`. An existing v1 SPECTRA
checkpoint cannot resume into this representation; retain its old results
separately and do not combine old full models with corrected matched controls.

The old implementation hash is
`793a3d2debc4cac7c045fb34df66f4f12f0694538e3da7a046c1b0623cee2de4`;
the new hash is
`8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26`.
The previous protocol and all 29 original baseline notebook bytes are
[archived](../archive/input_v1/README.md). The original 29-job baseline roster
is retained; the remaining expensive baseline runs are deferred for now.

External baseline inputs use their own preparation code. None calls the
corrected SPECTRA feature path; their encoders, packing, objectives and stopping
rules are unchanged. Completed external baseline results remain usable.
Continue their old checkpoints with the archived original notebooks, not the
new generated runtime. Checkpoint identity enforcement remains strict.

The validation collector and paired-analysis reporting allow exactly the old
external-baseline hash to compare with the new hash; original hashes are kept
in validation exports. This exception never accepts old SPECTRA inputs,
different dataset support, conflicting candidates or arbitrary runtime changes.
Attach only the authoritative full-model version when collecting results.

## Local checks

Regression checks cover Java callees, structural tags, line-number exclusion,
explicit lexical labels, Python fallback, graph-model backward passes,
checkpoint continuation, notebook compilation/embedding and reporting
compatibility. `scripts/audit_revision_lexical_inputs.py` compares old/new
features on existing local validation inputs without training or test scoring.

Keep every historical result, including unfavorable ones. This dated amendment
was made after inspecting validation behavior and before releasing final tests.
The first corrected full-model results are now recorded in the linked review;
only validation has been evaluated. Retain the unfavorable CodeNet change too.
