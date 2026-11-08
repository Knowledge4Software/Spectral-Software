# Review large training runs without downloading checkpoints

CodeBERT, C4-adapted and Qwen checkpoints can be several gigabytes because the
training outputs retain model weights and continuation state. This is expected;
file size does not establish that training completed successfully.

For validation screening, only `status.json` is essential. `benchmark.json`
adds measured compute costs. Keep each run's parent directory or another clear
dataset/model/seed label so identical JSON filenames are not mixed together.

Use [validation_screening.ipynb](../05_analysis/validation_screening.ipynb) to
collect small reports without downloading training outputs to your laptop:

1. Import this updated notebook into Kaggle. Select CPU; Internet can be off.
2. Add the saved outputs of CodeNet/CodeBERT, CodeNet/C4-adapted,
   CodeNet/Qwen3-LoRA, and CodeNet/SPECTRA full as inputs. Only attach outputs
   that the current Kaggle account can access. If the outputs are on different
   private accounts, run the collector separately on each account instead.
3. Do not attach the training dataset. Keep `QUEUE_CSV=None` unless a planned
   run ledger is wanted, then Save & Run All. This job reads reports; it does
   not start training or download pretrained models.
4. Download **only** `revision_small_reports.zip` from its saved output and put
   it into `C:\PyProjects\spectrals\outputs\revesion\01`. The ZIP size is printed
   at the end. It excludes `progress.pt`, NumPy token arrays, model downloads
   and final test artifacts.

Alternatively, download each original run's `status.json` and `benchmark.json`
individually and retain the run's name in the folder or filename. A budget pause
can have a status report but no benchmark yet.

Do not delete the original saved training outputs. Checkpoints are needed for
continuation, later final-test evaluation and weight-level integrity checks.
The report ZIP permits a summary review; it does not prove that all saved
weights are finite. No rerun is needed simply to obtain these reports.

Current priorities after the local batch-01 review:

1. Collect the three text-run reports and the missing SPECTRA/full report.
2. Resume CodeNet/CDLH/42 from its saved checkpoint, with the original settings.
3. If not already started, upload the corrected AtCoder/BigCloneBench v2 datasets
   and run their 10 + 9 seed-42 baseline jobs in parallel. Their training does
   not depend on downloading CodeNet checkpoints. Check the tracker first so
   already-running jobs are not duplicated.
4. Keep final tests off. Review the screening results before selecting repeated
   seeds and starting the next planned comparisons.
