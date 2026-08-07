# SemanticCloneBench V3 — Kaggle experiments

Attach `semanticclonebench_v3_clean_data.zip` as the only Kaggle input. Each notebook discovers the clean-data folder/ZIP automatically.

Shared experimental protocol: `MAX_TRAIN_PAIRS=100_000`, `MAX_VALID_PAIRS=20_000`, `MAX_TEST_PAIRS=20_000`. Smaller official splits are used in full.

- `baselines/`: nine self-contained comparison notebooks. Graph baselines use `ast`, the graph layer consistently available across all languages.
- `method/spectra_siam.ipynb`: canonical SPECTRA-Siam, 32 latent slots, four epochs, canonical node types plus hashed lexical fallback.
