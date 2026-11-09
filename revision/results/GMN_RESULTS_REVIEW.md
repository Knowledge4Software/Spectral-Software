# Additional GMN results — 2026-11-09

**AtCoder FA-AST-GMN is newly verified complete; all 32 checks passed. The BigCloneBench upload is a duplicate of the previously verified GMN run. Neither needs continuation.**

| Dataset | Accuracy | F1 | AUROC | Epochs / best epoch | Recorded hours | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| atcoder | 80.32% | 80.76% | 0.8757 | 38 / 28 | 5.87 | New full checkpoint |
| bigclonebench | 99.42% | 97.77% | 0.9944 | 30 / 20 | 5.08 | Existing verified run; duplicate re-upload |

The AtCoder result is selected by validation Accuracy at epoch 28 and stops at epoch 38 after ten non-improving validations. It covers the same 15,000 validation pairs and all 41,762 required dataset endpoints. Saved/current-best weights loaded strictly into the planned 1,204,610-parameter GMN; weights, optimizer and history were finite. Dataset/source identity, original legacy encoder hash, settings, threshold selection, metric counts, update cursor and benchmark checks passed. Test evaluation remains off.

Its original external runtime (`793a3d2...`) remains compatible with corrected SPECTRA: the input-v2 change affects only the SPECTRA canonical/lexical feature path. The current legacy encoder and preparation code match the saved sources. This is a portable project adaptation, not an exact original-paper reproduction.

The BigCloneBench ZIP has a different container hash, but **all 56 member files are byte-for-byte identical** to the existing reviewed archive, including the checkpoint and status. It is the same run and does not provide a new seed or independent replication. Its existing 97.77% F1 remains unchanged.

AtCoder GMN ranks third among six completed methods by Accuracy: RtvNN 82.91%, ASTNN 81.29%, GMN 80.32%, GGNN 80.26%, corrected SPECTRA 78.34%, DeepSim 71.27%. The GMN/GGNN difference is only 0.06 percentage points; no statistical significance is established. BigCloneBench GMN remains at 97.77% F1 versus corrected SPECTRA 97.15%. These are seed-42 validation results, not held-out final-test scores.

## File organization

- New active AtCoder ZIP: `atcoder_fa_ast_gmn_seed42_complete.zip`.
- Existing active BigCloneBench ZIP retained: `bigclonebench_fa_ast_gmn_seed42_complete.zip`.
- Incoming duplicate moved to `_archive/duplicate_uploads/bigclonebench_fa_ast_gmn_seed42_complete_duplicate01.zip`.
- `_audit/gmn_additions_audit.json` records the integrity checks, archive hashes and duplicate-content comparison. All moved/renamed ZIP hashes were preserved.

The baseline roster now has **18 verified completions, four paused CodeNet jobs and seven jobs without submitted results**. There are 22 current report sets and three separately retained historical SPECTRA full records; the duplicate upload does not increase the run count.

The next priority remains [five CodeNet controls](../queues/RUN_NOW_INPUT_V2.md). Keep their existing running jobs if already started. Expensive unreported external baselines can remain deferred.

Limits: counts-based metrics were recomputed; independent AUROC/AP reproduction needs validation predictions. Recorded hours exclude the final benchmark and Kaggle startup/export overhead. No model, notebook, training code or dataset was modified during this review.
