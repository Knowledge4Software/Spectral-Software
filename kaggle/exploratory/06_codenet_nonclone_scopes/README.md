# CodeNet non-clone scope study

These twelve notebooks share one graph package and the same 4,000 positive clone
pairs. Attach
`C:\PyProjects\spectrals\outputs\kaggle_datasets\codenet_4l_clean_data.zip`
to each Kaggle notebook, enable a T4-class GPU, and use **Run All**.

| Scope | SPECTRA-Siam | ASTNN | DeepSim | RtVNN | Positive pairs | Negative pairs |
| --- | --- | --- | --- | --- | ---: | ---: |
| Clone vs A/WA | `01_clone_vs_aw.ipynb` | `01_clone_vs_aw_astnn.ipynb` | `01_clone_vs_aw_deepsim.ipynb` | `01_clone_vs_aw_rtvnn.ipynb` | 4,000 clone | 4,000 `hard_nonclone` |
| Clone vs different problem | `02_clone_vs_diff_problem.ipynb` | `02_clone_vs_diff_problem_astnn.ipynb` | `02_clone_vs_diff_problem_deepsim.ipynb` | `02_clone_vs_diff_problem_rtvnn.ipynb` | 4,000 clone | 4,000 `nonclone_diff_problem` |
| Clone vs mixed | `03_clone_vs_mixed_aw_diff.ipynb` | `03_clone_vs_mixed_aw_diff_astnn.ipynb` | `03_clone_vs_mixed_aw_diff_deepsim.ipynb` | `03_clone_vs_mixed_aw_diff_rtvnn.ipynb` | 4,000 clone | 2,000 A/WA + 2,000 different-problem |

Every selected kind is uniform over the ten language configurations. Each run
contains 5,600 train, 1,200 validation, and 1,200 test pairs. Selection occurs
before graph loading, is deterministic, and is recorded in `PairScopeAudit`.
Every notebook stops if any language bucket or selected pair is lost during the
graph join. The baseline results and manifests include the pair-scope audit.

All runs use seed 42. The SPECTRA-Siam notebooks allow up to 8 epochs (matching
the maximum budget used by ASTNN and RtVNN) with validation ROC-AUC early stopping;
ASTNN and RtVNN allow 8 epochs, while DeepSim allows 5 under their faithful schedules.

All three use redesigned method 2 (`graph_signal_spectral`): attributed graph
signals pass through the learned latent Laplacian and Chebyshev filters, with no
direct pooled-node or raw-source lexical bypass. Mutation pairs are excluded
from this study; the old mutation notebooks remain only as legacy artifacts.

All three SPECTRA scope notebooks use the comparison-focused revision over the
original joint latent spectrum. Their symmetric classifiers receive only
block-normalized differences, products, log-distances, and summary distances
from the density, heat-trace, and Chebyshev-energy descriptor blocks. Individual
descriptors and code embeddings never enter the final head. Seed 42, eight
epochs, pair IDs, splits, and all other experiment-budget settings remain
unchanged. The descriptor remains 152-dimensional and the revised model has
3,016,855 trainable parameters versus 3,013,429 originally (+0.11%), keeping
capacity effectively fixed.

The shared package exports AST, CFG, DDG, and CPG so graph baselines can be run
on the exact same pair IDs. The three SPECTRA notebooks consume AST+DDG and derive their sequential
`next_token` relation directly from AST node order. Both endpoints of every
sampled pair are restricted to 20--50 physical source lines, inclusive.

ASTNN and RtVNN consume AST. DeepSim consumes CFG; 21 of the 20,902 pair-referenced
programs have an empty exported CFG. For those programs only, the DeepSim
notebooks substitute the package's complete CPG adjacency instead of deleting
their pairs. Every fallback code ID and count is written to the run manifest,
so all methods still evaluate the exact 8,000-pair scope.

Build the shared local graph package from the repository root:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/nonclone_scope_study/run_pipeline.py `
  --stop-after 01 --force-prepare

.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/nonclone_scope_study/run_pipeline.py `
  --start-at 02
```

Download each notebook's `*_results.csv` and `*_run_metadata.json` outputs.
