# ATCoder cross-language pipeline

This pipeline mirrors the portable XGLUE export, but runs Joern separately for
Java and Python before merging the sparse graphs and spectral features into a
single clean Kaggle dataset.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe notebooks\datasets\atcoder\run_pipeline\run_all.py
```

The default input is `../data/archive.zip`; final files are written under
`../outputs/atcoder/clean_data/` and `../outputs/atcoder/atcoder_clean_data.zip`.

By default, 4,578 officially generated but invalid negative labels are excluded.
Set `ATCODER_INCLUDE_INVALID_GENERATED_NEGATIVES=1` before running stage 01 only
if every official row must be retained.
