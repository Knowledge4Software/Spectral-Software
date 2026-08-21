# Cross-language transfer

This experiment trains on mono-language pairs from one SemanticCloneBench V3
language, selects its threshold on validation pairs from that **same source
language**, freezes the threshold, then evaluates every target language.  The
diagonal is in-language performance; every off-diagonal cell is zero-shot
transfer.

For SemanticCloneBench, attach `semanticclonebench_v3_clean_data.zip` and run:

1. `01_spectra_siam_semanticclonebench_v3.ipynb`
2. `02_astnn_semanticclonebench_v3.ipynb`
3. `03_rtvnn_semanticclonebench_v3.ipynb`
4. `04_deepsim_semanticclonebench_v3.ipynb`

For the additional SPECTRA-Siam GPTCloneBench transfer matrix, attach
`gptclonebench_v3_clean_data.zip` and run
`05_spectra_siam_gptclonebench_v3.ipynb`.

Each notebook writes a `*_cross_language_results.csv` matrix and a matching
`*_cross_language_f1.png` heatmap under `/kaggle/working`.

The three comparison baselines and SPECTRA-Siam use `comparison_50k`.  Since
SemanticCloneBench has fewer pairs per mono-language split, all available
eligible pairs are used.  The SPECTRA notebook keeps `USE_SOURCE_LEXICAL=False`
so source-code tokens are not an input to the method under test.
