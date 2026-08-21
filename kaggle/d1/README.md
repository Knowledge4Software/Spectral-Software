# Section 6.1: hyperparameter sensitivity

Four one-factor sweeps on CodeNet, five epochs per arm. Everything except the
swept value is identical to the canonical lexical run, so each curve isolates
one hyperparameter.

| Notebook | Sweeps |
| --- | --- |
| `01_latent_graph_size.ipynb` | latent graph size `m` |
| `02_assignment_iterations.ipynb` | latent-assignment iterations |
| `03_chebyshev_order.ipynb` | Chebyshev order |
| `04_batch_size.ipynb` | batch size |

Each writes `sensitivity_<sweep>.csv` to `/kaggle/working`. Collect them under
`outputs/kaggle/d1/`, then build the figure and table:

```bash
python research/discussions/build_d1_latex.py
```
