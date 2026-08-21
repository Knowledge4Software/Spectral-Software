# Section 6.4: what the latent graph captures

Inference only: loads the finished CodeNet lexical checkpoint and aggregates
the slot-specialization matrix, the mean latent adjacency, and one held-out
triplet. Nothing is retrained.

Attach the CodeNet clean-data dataset and the finished SPECTRA-Siam lexical run
(for `spectra_siam_*_final.pt`), then Run All.

It can also be run locally, without Kaggle, in about 20 minutes on CPU:

```bash
python research/discussions/run_d4_local.py
```

That writes the figure and `d4_triplet_eigenvalues.csv` straight into
`kaggle/latex/d4/figures/`. Keep the CSV: restyling the eigenvalue strips from
it takes seconds, whereas regenerating it needs the full pass.
