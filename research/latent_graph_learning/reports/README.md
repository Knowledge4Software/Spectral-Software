# Final paper benchmark report

Run `01_final_paper_benchmark_report.ipynb` after placing Kaggle result ZIPs in
`outputs/kaggle` and completing the local spectral-representation runs. The
notebook displays the article-style tables directly and regenerates their CSV
and PNG exports, plus the coverage check, in `reports/artifacts/`.

It reports CodeXGLUE in the BigCloneBench column, as chosen for this project.
Missing artifacts are marked as missing; no metric is estimated or fabricated.

The collaborator-provided pretrained metrics are archived separately in
`artifacts/pretrained_collaborator_results.csv`. Run
`generate_paper_results_tables.ps1` to regenerate the source-separated result
tables in both paper drafts from that file and `artifacts/wide_test_metrics.csv`.
