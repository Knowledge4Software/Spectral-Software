# RQ3 tables provenance

Accuracies come from `outputs/kaggle/RQ3`. UniXcoder cells are a
collaborator's runs, copied verbatim from the paper source.

Every model is trained under the same 4-epoch budget.

## Runs that ended near single-class prediction

These accuracies are reported in the tables, but their recall shows the
run barely separated the classes at this epoch budget. Nearly all target
C\#, so the pattern is a property of that target rather than of one model.

- ASTNN / table10 / C->J->P->S|X2-X2: recall 0.985, accuracy 0.5060
- ASTNN / table10 / C->J->P->S|none: recall 0.996, accuracy 0.5087
- ASTNN / table10 / C->P->J->S|none: recall 0.983, accuracy 0.4980
- ASTNN / table10 / J->C->P->S|none: recall 0.999, accuracy 0.5007
- ASTNN / table10 / J->P->C->S|X2-X2+X3-X3: recall 0.992, accuracy 0.5053
- ASTNN / table10 / J->P->C->S|X2-X2: recall 0.999, accuracy 0.4993
- ASTNN / table10 / J->P->C->S|X3-X3: recall 0.993, accuracy 0.5007
- ASTNN / table10 / J->P->C->S|none: recall 0.989, accuracy 0.5020
- ASTNN / table10 / P->C->J->S|none: recall 0.991, accuracy 0.5087
- ASTNN / table10 / P->J->C->S|X2-X2+X3-X3: recall 0.989, accuracy 0.4953
- ASTNN / table10 / P->J->C->S|X2-X2: recall 0.992, accuracy 0.4973
- ASTNN / table10 / P->J->C->S|X3-X3: recall 0.995, accuracy 0.4973
- ASTNN / table8 / C->S: recall 0.991, accuracy 0.4953
- ASTNN / table8 / P->S: recall 0.983, accuracy 0.5040
- ASTNN / table9 / C->J->S|X2-X2: recall 0.988, accuracy 0.5033
- ASTNN / table9 / J->C->S|X2-X2: recall 0.985, accuracy 0.5007
- ASTNN / table9 / J->C->S|none: recall 0.997, accuracy 0.5007
- ASTNN / table9 / J->P->S|none: recall 0.997, accuracy 0.5040
- ASTNN / table9 / P->C->S|X2-X2: recall 0.985, accuracy 0.4980
- ASTNN / table9 / P->J->S|X2-X2: recall 1.000, accuracy 0.5020
- ASTNN / table9 / P->J->S|none: recall 0.989, accuracy 0.5093
- DeepSim / table10 / P->J->C->S|X2-X2: recall 0.985, accuracy 0.5580
- DeepSim / table10 / S->J->C->P|none: recall 0.985, accuracy 0.5913
- DeepSim / table8 / J->S: recall 1.000, accuracy 0.5000
- DeepSim / table9 / J->C->S|X2-X2: recall 0.999, accuracy 0.5227
- RtvNN / table9 / J->C->S|X2-X2: recall 0.983, accuracy 0.5220
- SPECTRA-Siam / table10 / C->J->S->P|X3-X3: recall 0.983, accuracy 0.6023
- SPECTRA-Siam / table10 / C->J->S->P|none: recall 0.993, accuracy 0.5734
- SPECTRA-Siam / table10 / C->P->J->S|X2-X2+X3-X3: recall 0.983, accuracy 0.5640
- SPECTRA-Siam / table10 / C->P->J->S|none: recall 0.980, accuracy 0.5487
- SPECTRA-Siam / table10 / C->S->J->P|X3-X3: recall 0.997, accuracy 0.5560
- SPECTRA-Siam / table10 / C->S->J->P|none: recall 0.985, accuracy 0.6144
- SPECTRA-Siam / table10 / J->C->P->S|none: recall 0.988, accuracy 0.5313
- SPECTRA-Siam / table10 / J->S->C->P|X3-X3: recall 0.996, accuracy 0.5547
- SPECTRA-Siam / table10 / J->S->C->P|none: recall 0.996, accuracy 0.5211
- SPECTRA-Siam / table10 / J->S->P->C|X3-X3: recall 0.983, accuracy 0.5727
- SPECTRA-Siam / table10 / P->C->J->S|X3-X3: recall 0.993, accuracy 0.5540
- SPECTRA-Siam / table10 / P->C->S->J|none: recall 0.988, accuracy 0.5757
- SPECTRA-Siam / table10 / P->J->C->S|X2-X2: recall 0.989, accuracy 0.5320
- SPECTRA-Siam / table10 / P->J->C->S|X3-X3: recall 1.000, accuracy 0.5160
- SPECTRA-Siam / table10 / P->J->C->S|none: recall 1.000, accuracy 0.5127
- SPECTRA-Siam / table10 / P->J->S->C|none: recall 0.985, accuracy 0.5527
- SPECTRA-Siam / table10 / P->S->J->C|none: recall 0.983, accuracy 0.5433
- SPECTRA-Siam / table10 / S->C->J->P|none: recall 0.981, accuracy 0.6318
- SPECTRA-Siam / table10 / S->C->P->J|X3-X3: recall 0.981, accuracy 0.6157
- SPECTRA-Siam / table10 / S->J->C->P|X3-X3: recall 0.993, accuracy 0.5755
- SPECTRA-Siam / table8 / J->S: recall 1.000, accuracy 0.5007
- SPECTRA-Siam / table8 / S->P: recall 0.992, accuracy 0.5446
