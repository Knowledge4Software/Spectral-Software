# RQ1: latent-graph export

Runs the trained lexical method over a benchmark and exports every pair's
learned latent graph and its spectrum, so RQ1 can compare the learned spectral
representation against spectra of fixed program graphs without retraining.

| Notebook | Benchmark |
| --- | --- |
| `01_atcoder_export_latent_graphs.ipynb` | AtCoder |
| `02_xglue_export_latent_graphs.ipynb` | BigCloneBench (XGLUE) |

These carry the same model as the other method notebooks plus detached tensors
returned only outside training, so the exported run is the reference run.

Collect the archives under `outputs/kaggle/RQ1/`, then:

```bash
python research/rq1/04_run_all.py            # the RQ1 table
python research/rq1/export_figure_pdfs.py    # the score-distribution figure
```

Regenerate these notebooks from the current method with
`python research/rq1/build_notebooks.py`.
