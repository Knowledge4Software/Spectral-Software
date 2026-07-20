# Kaggle Notebooks

These notebooks consume a portable clean export containing code, labelled
pairs, sparse graph matrices, and spectral values. They do not run Joern.

| Notebook | Method family |
| --- | --- |
| `gnn_baselines.ipynb` | One graph type at a time: AST, CFG, PDG, DDG, CPG. |
| `snn_baselines.ipynb` | Siamese network over spectral representations. |
| `deckard_baseline.ipynb` | Deckard-style structural baseline. |
| `rtvnn_baseline.ipynb` | Token-sequence Siamese baseline. |
| `cdlh_baseline.ipynb` | Code representation baseline. |
| `deepsim_baseline.ipynb` | Deep similarity baseline. |
| `astnn_baseline.ipynb` | ASTNN-style baseline. |
| `fa_ast_ggnn_baseline.ipynb` | FA-AST GGNN-style baseline. |
| `fa_ast_gmn_baseline.ipynb` | FA-AST graph-matching baseline. |

Attach only the XGLUE dataset at this path:
`/kaggle/input/datasets/koushamoeini/xglue4`. Then run all cells once. Each
notebook writes XGLUE-only outputs such as `xglue4_*_results.csv` to
`/kaggle/working/` and displays the result table at the end.
