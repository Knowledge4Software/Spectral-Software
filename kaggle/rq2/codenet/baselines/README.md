# CodeNet 4L 50k baselines

These are the same canonical baseline families used for AtCoder, BigCloneBench,
GPTCloneBench, and SemanticCloneBench. Attach the final CodeNet clean-data
dataset containing `pairs.csv.gz`, `codes.jsonl.gz`, and
`graph_spectra.jsonl.gz`, enable **2x T4 GPU**, and choose **Run All**.

The notebooks reject the retired 12k non-clone-scope package. The required
CodeNet 50k release has 70,000 train, 15,000 validation, and 15,000 test pairs
(50,000 clones and 50,000 different-problem non-clones overall). All neural
baselines use four epochs and validation selects Accuracy when validation is
exactly balanced, otherwise F1.

Run the following files:

1. `deckard_baseline.ipynb`
2. `cdlh_baseline.ipynb`
3. `rtvnn_baseline.ipynb`
4. `astnn_baseline.ipynb`
5. `snn_baselines.ipynb` — AST/CFG/DDG/CPG spectrum + SNN
6. `deepsim_baseline.ipynb`
7. `fa_ast_ggnn_baseline.ipynb`
8. `fa_ast_gmn_baseline.ipynb`
9. `gnn_baselines.ipynb` — GNN-AST/CFG/DDG/CPG

The main method and its three input ablations remain in `../method/`.
Each notebook writes result CSVs, a manifest, and language-level test results
to `/kaggle/working` for download.
