# Table 10 run checklist

Each notebook covers two length-3 bridge paths (8 configurations).
Tick a row once its notebook has finished and its CSV is downloaded.

Results are written to `rq3_table10_<model>_<target>_part<N>.csv`.

| Done | Model | Target | Part | Paths | Configs | Est. hours | Notebook |
|---|---|---|---|---|---|---|---|
| [ ] | SPECTRA-Siam | Java (J) | 1 | `C->P->S->J`, `C->S->P->J` | 8 | 1.0 | `spectra_siam/J/part1_CPSJ__CSPJ.ipynb` |
| [ ] | SPECTRA-Siam | Java (J) | 2 | `P->C->S->J`, `P->S->C->J` | 8 | 1.0 | `spectra_siam/J/part2_PCSJ__PSCJ.ipynb` |
| [ ] | SPECTRA-Siam | Java (J) | 3 | `S->C->P->J`, `S->P->C->J` | 8 | 1.0 | `spectra_siam/J/part3_SCPJ__SPCJ.ipynb` |
| [ ] | SPECTRA-Siam | Python (P) | 1 | `C->J->S->P`, `C->S->J->P` | 8 | 1.0 | `spectra_siam/P/part1_CJSP__CSJP.ipynb` |
| [ ] | SPECTRA-Siam | Python (P) | 2 | `J->C->S->P`, `J->S->C->P` | 8 | 1.0 | `spectra_siam/P/part2_JCSP__JSCP.ipynb` |
| [ ] | SPECTRA-Siam | Python (P) | 3 | `S->C->J->P`, `S->J->C->P` | 8 | 1.0 | `spectra_siam/P/part3_SCJP__SJCP.ipynb` |
| [ ] | SPECTRA-Siam | C++ (C) | 1 | `J->P->S->C`, `J->S->P->C` | 8 | 1.0 | `spectra_siam/C/part1_JPSC__JSPC.ipynb` |
| [ ] | SPECTRA-Siam | C++ (C) | 2 | `P->J->S->C`, `P->S->J->C` | 8 | 1.0 | `spectra_siam/C/part2_PJSC__PSJC.ipynb` |
| [ ] | SPECTRA-Siam | C++ (C) | 3 | `S->J->P->C`, `S->P->J->C` | 8 | 1.0 | `spectra_siam/C/part3_SJPC__SPJC.ipynb` |
| [ ] | SPECTRA-Siam | C# (S) | 1 | `C->J->P->S`, `C->P->J->S` | 8 | 1.0 | `spectra_siam/S/part1_CJPS__CPJS.ipynb` |
| [ ] | SPECTRA-Siam | C# (S) | 2 | `J->C->P->S`, `J->P->C->S` | 8 | 1.0 | `spectra_siam/S/part2_JCPS__JPCS.ipynb` |
| [ ] | SPECTRA-Siam | C# (S) | 3 | `P->C->J->S`, `P->J->C->S` | 8 | 1.0 | `spectra_siam/S/part3_PCJS__PJCS.ipynb` |
| [ ] | ASTNN | Java (J) | 1 | `C->P->S->J`, `C->S->P->J` | 8 | 4.2 | `astnn/J/part1_CPSJ__CSPJ.ipynb` |
| [ ] | ASTNN | Java (J) | 2 | `P->C->S->J`, `P->S->C->J` | 8 | 4.2 | `astnn/J/part2_PCSJ__PSCJ.ipynb` |
| [ ] | ASTNN | Java (J) | 3 | `S->C->P->J`, `S->P->C->J` | 8 | 4.2 | `astnn/J/part3_SCPJ__SPCJ.ipynb` |
| [ ] | ASTNN | Python (P) | 1 | `C->J->S->P`, `C->S->J->P` | 8 | 4.2 | `astnn/P/part1_CJSP__CSJP.ipynb` |
| [ ] | ASTNN | Python (P) | 2 | `J->C->S->P`, `J->S->C->P` | 8 | 4.2 | `astnn/P/part2_JCSP__JSCP.ipynb` |
| [ ] | ASTNN | Python (P) | 3 | `S->C->J->P`, `S->J->C->P` | 8 | 4.2 | `astnn/P/part3_SCJP__SJCP.ipynb` |
| [ ] | ASTNN | C++ (C) | 1 | `J->P->S->C`, `J->S->P->C` | 8 | 4.2 | `astnn/C/part1_JPSC__JSPC.ipynb` |
| [ ] | ASTNN | C++ (C) | 2 | `P->J->S->C`, `P->S->J->C` | 8 | 4.2 | `astnn/C/part2_PJSC__PSJC.ipynb` |
| [ ] | ASTNN | C++ (C) | 3 | `S->J->P->C`, `S->P->J->C` | 8 | 4.2 | `astnn/C/part3_SJPC__SPJC.ipynb` |
| [ ] | ASTNN | C# (S) | 1 | `C->J->P->S`, `C->P->J->S` | 8 | 4.2 | `astnn/S/part1_CJPS__CPJS.ipynb` |
| [ ] | ASTNN | C# (S) | 2 | `J->C->P->S`, `J->P->C->S` | 8 | 4.2 | `astnn/S/part2_JCPS__JPCS.ipynb` |
| [ ] | ASTNN | C# (S) | 3 | `P->C->J->S`, `P->J->C->S` | 8 | 4.2 | `astnn/S/part3_PCJS__PJCS.ipynb` |
| [ ] | RtvNN | Java (J) | 1 | `C->P->S->J`, `C->S->P->J` | 8 | 4.0 | `rtvnn/J/part1_CPSJ__CSPJ.ipynb` |
| [ ] | RtvNN | Java (J) | 2 | `P->C->S->J`, `P->S->C->J` | 8 | 4.0 | `rtvnn/J/part2_PCSJ__PSCJ.ipynb` |
| [ ] | RtvNN | Java (J) | 3 | `S->C->P->J`, `S->P->C->J` | 8 | 4.0 | `rtvnn/J/part3_SCPJ__SPCJ.ipynb` |
| [ ] | RtvNN | Python (P) | 1 | `C->J->S->P`, `C->S->J->P` | 8 | 4.0 | `rtvnn/P/part1_CJSP__CSJP.ipynb` |
| [ ] | RtvNN | Python (P) | 2 | `J->C->S->P`, `J->S->C->P` | 8 | 4.0 | `rtvnn/P/part2_JCSP__JSCP.ipynb` |
| [ ] | RtvNN | Python (P) | 3 | `S->C->J->P`, `S->J->C->P` | 8 | 4.0 | `rtvnn/P/part3_SCJP__SJCP.ipynb` |
| [ ] | RtvNN | C++ (C) | 1 | `J->P->S->C`, `J->S->P->C` | 8 | 4.0 | `rtvnn/C/part1_JPSC__JSPC.ipynb` |
| [ ] | RtvNN | C++ (C) | 2 | `P->J->S->C`, `P->S->J->C` | 8 | 4.0 | `rtvnn/C/part2_PJSC__PSJC.ipynb` |
| [ ] | RtvNN | C++ (C) | 3 | `S->J->P->C`, `S->P->J->C` | 8 | 4.0 | `rtvnn/C/part3_SJPC__SPJC.ipynb` |
| [ ] | RtvNN | C# (S) | 1 | `C->J->P->S`, `C->P->J->S` | 8 | 4.0 | `rtvnn/S/part1_CJPS__CPJS.ipynb` |
| [ ] | RtvNN | C# (S) | 2 | `J->C->P->S`, `J->P->C->S` | 8 | 4.0 | `rtvnn/S/part2_JCPS__JPCS.ipynb` |
| [ ] | RtvNN | C# (S) | 3 | `P->C->J->S`, `P->J->C->S` | 8 | 4.0 | `rtvnn/S/part3_PCJS__PJCS.ipynb` |
| [ ] | DeepSim | Java (J) | 1 | `C->P->S->J`, `C->S->P->J` | 8 | 1.2 | `deepsim/J/part1_CPSJ__CSPJ.ipynb` |
| [ ] | DeepSim | Java (J) | 2 | `P->C->S->J`, `P->S->C->J` | 8 | 1.2 | `deepsim/J/part2_PCSJ__PSCJ.ipynb` |
| [ ] | DeepSim | Java (J) | 3 | `S->C->P->J`, `S->P->C->J` | 8 | 1.2 | `deepsim/J/part3_SCPJ__SPCJ.ipynb` |
| [ ] | DeepSim | Python (P) | 1 | `C->J->S->P`, `C->S->J->P` | 8 | 1.2 | `deepsim/P/part1_CJSP__CSJP.ipynb` |
| [ ] | DeepSim | Python (P) | 2 | `J->C->S->P`, `J->S->C->P` | 8 | 1.2 | `deepsim/P/part2_JCSP__JSCP.ipynb` |
| [ ] | DeepSim | Python (P) | 3 | `S->C->J->P`, `S->J->C->P` | 8 | 1.2 | `deepsim/P/part3_SCJP__SJCP.ipynb` |
| [ ] | DeepSim | C++ (C) | 1 | `J->P->S->C`, `J->S->P->C` | 8 | 1.2 | `deepsim/C/part1_JPSC__JSPC.ipynb` |
| [ ] | DeepSim | C++ (C) | 2 | `P->J->S->C`, `P->S->J->C` | 8 | 1.2 | `deepsim/C/part2_PJSC__PSJC.ipynb` |
| [ ] | DeepSim | C++ (C) | 3 | `S->J->P->C`, `S->P->J->C` | 8 | 1.2 | `deepsim/C/part3_SJPC__SPJC.ipynb` |
| [ ] | DeepSim | C# (S) | 1 | `C->J->P->S`, `C->P->J->S` | 8 | 1.2 | `deepsim/S/part1_CJPS__CPJS.ipynb` |
| [ ] | DeepSim | C# (S) | 2 | `J->C->P->S`, `J->P->C->S` | 8 | 1.2 | `deepsim/S/part2_JCPS__JPCS.ipynb` |
| [ ] | DeepSim | C# (S) | 3 | `P->C->J->S`, `P->J->C->S` | 8 | 1.2 | `deepsim/S/part3_PCJS__PJCS.ipynb` |

## Estimated total per model

| Model | Notebooks | Hours |
|---|---|---|
| SPECTRA-Siam | 12 | 12.0 |
| ASTNN | 12 | 50.4 |
| RtvNN | 12 | 48.0 |
| DeepSim | 12 | 14.4 |

**48 notebooks, ~125 GPU-hours in total.**
