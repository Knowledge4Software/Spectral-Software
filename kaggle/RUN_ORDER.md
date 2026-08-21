# Kaggle run order

Use the same accelerator, dataset release, seed policy, and run profile for all
methods included in one table. Every completed run writes metrics, runtime,
trainable-parameter count, environment details, and pair counts under
`/kaggle/working/`.

For each established V3 benchmark, run the ordinary full-split comparison in
this order:

1. `baselines/deckard_baseline.ipynb`
2. `baselines/cdlh_baseline.ipynb`
3. `baselines/rtvnn_baseline.ipynb`
4. `baselines/astnn_baseline.ipynb`
5. `baselines/snn_baselines.ipynb`
6. `baselines/deepsim_baseline.ipynb`
7. `baselines/fa_ast_ggnn_baseline.ipynb`
8. `baselines/fa_ast_gmn_baseline.ipynb`
9. `baselines/gnn_baselines.ipynb`
10. `method/spectra_siam_lex.ipynb`

For the input-ablation table, additionally run
`method/spectra_siam_topo.ipynb` and `method/spectra_siam_label.ipynb`.
They share the exact same spectral pair head as `lex`; only encoder inputs are
removed.

For the **CodeNet 4L 50k main table**, attach the final `codenet_4l` clean-data
release (not the older 12k scope-study ZIP), then use the identical sequence
under `codenet_4l/`:

1. `baselines/deckard_baseline.ipynb`
2. `baselines/cdlh_baseline.ipynb`
3. `baselines/rtvnn_baseline.ipynb`
4. `baselines/astnn_baseline.ipynb`
5. `baselines/snn_baselines.ipynb`
6. `baselines/deepsim_baseline.ipynb`
7. `baselines/fa_ast_ggnn_baseline.ipynb`
8. `baselines/fa_ast_gmn_baseline.ipynb`
9. `baselines/gnn_baselines.ipynb`
10. `method/spectra_siam_lex.ipynb`

For the CodeNet input ablations additionally run
`method/spectra_siam_topo.ipynb` and `method/spectra_siam_label.ipynb`.

For the current CodeNet scope study, first attach the shared 12k clean-data
export and then run these notebooks in order:

1. `experiments/06_codenet_nonclone_scopes/01_clone_vs_aw.ipynb`
2. `experiments/06_codenet_nonclone_scopes/02_clone_vs_diff_problem.ipynb`
3. `experiments/06_codenet_nonclone_scopes/03_clone_vs_mixed_aw_diff.ipynb`

The attachment must contain `codes.jsonl(.gz)`, `pairs.csv(.gz)`, and
`graph_spectra.jsonl(.gz)`. All three runs use the same clone population and
write separate result filenames.
