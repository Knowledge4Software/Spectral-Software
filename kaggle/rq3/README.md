# RQ3: Bridge-Assisted Cross-Language Generalization (Project CodeNet)

One notebook per (table, model, target language). Attach the CodeNet 4L
clean-data dataset, select a T4 GPU, and Run All. Each notebook appends to
its own CSV in `/kaggle/working` after every configuration, so a notebook
that hits the session limit can simply be rerun and will continue.

Language codes: J=Java, P=Python, C=C++, S=C#.

## Notebooks

| Table | Model | Target | Configs | Notebook |
|---|---|---|---|---|
| table10 | ASTNN | J | 24 | `table10/rq3_table10_astnn_J.ipynb` |
| table10 | ASTNN | P | 24 | `table10/rq3_table10_astnn_P.ipynb` |
| table10 | ASTNN | C | 24 | `table10/rq3_table10_astnn_C.ipynb` |
| table10 | ASTNN | S | 24 | `table10/rq3_table10_astnn_S.ipynb` |
| table10 | RtvNN | J | 24 | `table10/rq3_table10_rtvnn_J.ipynb` |
| table10 | RtvNN | P | 24 | `table10/rq3_table10_rtvnn_P.ipynb` |
| table10 | RtvNN | C | 24 | `table10/rq3_table10_rtvnn_C.ipynb` |
| table10 | RtvNN | S | 24 | `table10/rq3_table10_rtvnn_S.ipynb` |
| table10 | DeepSim | J | 24 | `table10/rq3_table10_deepsim_J.ipynb` |
| table10 | DeepSim | P | 24 | `table10/rq3_table10_deepsim_P.ipynb` |
| table10 | DeepSim | C | 24 | `table10/rq3_table10_deepsim_C.ipynb` |
| table10 | DeepSim | S | 24 | `table10/rq3_table10_deepsim_S.ipynb` |
| table10 | SPECTRA-Siam | J | 24 | `table10/rq3_table10_spectra_siam_J.ipynb` |
| table10 | SPECTRA-Siam | P | 24 | `table10/rq3_table10_spectra_siam_P.ipynb` |
| table10 | SPECTRA-Siam | C | 24 | `table10/rq3_table10_spectra_siam_C.ipynb` |
| table10 | SPECTRA-Siam | S | 24 | `table10/rq3_table10_spectra_siam_S.ipynb` |

## What each table measures

| Table | Setting |
|---|---|
| table7 | Within-language reference, and cross-language with no cross-language supervision |
| table8 | Length-1 bridge `X1 -> Y` |
| table9 | Length-2 bridge `X1 -> X2 -> Y`, with and without `X2-X2` |
| table10 | Length-3 bridge `X1 -> X2 -> X3 -> Y`, four reinforcement modes |

## After the runs

Download every `rq3_*.csv` into one folder, then:

```bash
python research/rq3/aggregate.py --results-dir <folder> --output-dir outputs/rq3
```

That writes Tables 7-10 (LaTeX + PNG) and `section_5_3_text.md` with every
statistic the Section 5.3 placeholders need.

## Pretrained column

The paper's tables include a Pretrained column. These notebooks do not
produce it; the aggregator leaves those cells empty.
