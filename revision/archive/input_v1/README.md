# Preserved input-v1 notebooks: continuation only

These 29 original baseline notebooks are byte-for-byte snapshots from before
the 2026-11-09 Java callee lexical-input fix. Their protocol, notebook manifest
and manual result tracker are retained alongside them.

Use the original CodeNet CodeBERT, C4-adapted, CDLH or Qwen notebook here when
resuming its old checkpoint. Attach the same dataset and that method's latest
saved output; retain all original settings and the method's own token cache.
The old training implementation hash is
`793a3d2debc4cac7c045fb34df66f4f12f0694538e3da7a046c1b0623cee2de4`.

The same rule applies to any other already-started v1 external baseline.
Existing completed external baselines can be reused because their inputs do
not use the corrected SPECTRA canonical/lexical loader.

Do not start new SPECTRA runs from these notebooks. Historical v1 SPECTRA full
results remain evidence, but new full models and matched controls must all
use input v2 and fresh training. Follow the
[current eight-run list](../../queues/RUN_NOW_INPUT_V2.md).
