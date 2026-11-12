# Continue the revision: checkpoint diagnostics and F1 review

[Persian guide](README_FA.md)

**Yes: attach the saved outputs of the previous training runs.** Each required
saved output must contain its original `progress.pt`. The old `.ipynb` files
alone do not contain the trained weights and are not sufficient inputs.
Attaching an output lets the new notebook read its files; it does not rerun
that training notebook.

## What to run and attach

The first four notebooks are independent and can run in parallel.

| New notebook | Existing dataset to attach | Previous training outputs to attach | Hardware |
| --- | --- | --- | --- |
| [01_atcoder_diagnosis.ipynb](01_atcoder_diagnosis.ipynb) | `atcoder_revision_common_support_v2.zip` | AtCoder / full, seeds **42, 123, 231**: three outputs | T4 GPU |
| [02_codenet_f1_seed42.ipynb](02_codenet_f1_seed42.ipynb) | `codenet_revision_common_support.zip` | All six CodeNet conditions at **seed42** | T4 GPU |
| [02_codenet_f1_seed123.ipynb](02_codenet_f1_seed123.ipynb) | The same CodeNet dataset | All six CodeNet conditions at **seed123** | T4 GPU |
| [02_codenet_f1_seed231.ipynb](02_codenet_f1_seed231.ipynb) | The same CodeNet dataset | All six CodeNet conditions at **seed231** | T4 GPU |
| [03_collect_review_reports.ipynb](03_collect_review_reports.ipynb), after those four finish | No training dataset | The saved outputs of the **four new analysis notebooks above** | CPU |

The six CodeNet conditions are `full`, `mean_max`, `attention`,
`classification_only`, `mean_max_classification_only`, and
`attention_classification_only`. Attach the six matching outputs for that
notebook's seed. The new analyses use 21 existing checkpoints in total:
three AtCoder checkpoints and 18 CodeNet checkpoints.

**Reuse the current dataset uploads. You do not need to upload the original
2 GB datasets again.** No BigCloneBench input or text-baseline output is needed
for this batch.

## Kaggle steps

1. Import the **new analysis `.ipynb`** from this folder.
2. For the first four jobs, enable a T4 GPU and turn Internet off.
3. Attach the matching existing dataset from the table.
4. Attach the required **saved training output versions**, containing
   `progress.pt`. Choose the same versions that produced the reviewed ZIPs
   listed below. You do not need to edit or run the original training notebooks.
5. Keep the default settings in the new analysis notebook:

   ```python
   CLEAN_DATA_DIR = None
   CHECKPOINT_PATHS = {}
   RESUME_REPORTS_DIR = None
   FINAL_TEST = False
   ```

6. Save and run. The notebook discovers checkpoints automatically and checks
   their SHA-256 hashes. If `missing_inputs` appears or a checksum check fails,
   check the attached output versions before trying again.
7. Download only the generated `_small_reports.zip`. Preserve the saved Kaggle
   output so it can be attached to the collector or used to resume analysis.

**Alternative if you cannot attach a previous saved output:** upload the
matching local result ZIPs listed below into a Kaggle Dataset and attach that
dataset. The analysis accepts extracted `progress.pt` files or the original,
exactly named ZIPs. Preserve the original checkpoint bytes; do not resave,
convert, or strip the checkpoint. The training dataset from the table is still
required alongside these checkpoint inputs.

## Exact previous result archives

The ZIP filenames below are our renamed local downloads; the corresponding
Kaggle notebook titles may differ. Match the dataset, condition, seed and saved
output version. These paths are relative to:

```text
C:\PyProjects\spectrals\outputs\revesion
```

The authoritative machine-readable list is
[checkpoint_review_inputs.csv](../../queues/checkpoint_review_inputs.csv).
Checkpoint hashes and identities are pinned in
[checkpoint_review_manifest.json](../../queues/checkpoint_review_manifest.json).

### AtCoder diagnosis: three previous outputs

```text
01/atcoder_full_seed42_input_v2_complete.zip
03/atcoder_full_seed123_input_v2_validation_fix_complete.zip
03/atcoder_full_seed231_input_v2_validation_fix_complete.zip
```

### CodeNet seed42: six previous outputs

```text
01/codenet_full_seed42_input_v2_complete.zip
02/codenet_mean_max_seed42_input_v2_complete.zip
02/codenet_attention_seed42_input_v2_complete.zip
02/codenet_classification_only_seed42_input_v2_complete.zip
02/codenet_mean_max_classification_only_seed42_input_v2_complete.zip
02/codenet_attention_classification_only_seed42_input_v2_complete.zip
```

### CodeNet seed123: six previous outputs

```text
03/codenet_full_seed123_input_v2_complete.zip
03/codenet_mean_max_seed123_input_v2_complete.zip
03/codenet_attention_seed123_input_v2_complete.zip
03/codenet_classification_only_seed123_input_v2_validation_fix_restart_complete.zip
03/codenet_mean_max_classification_only_seed123_input_v2_complete.zip
03/codenet_attention_classification_only_seed123_input_v2_complete.zip
```

### CodeNet seed231: six previous outputs

```text
03/codenet_full_seed231_input_v2_complete.zip
03/codenet_mean_max_seed231_input_v2_complete.zip
03/codenet_attention_seed231_input_v2_complete.zip
03/codenet_classification_only_seed231_input_v2_complete.zip
03/codenet_mean_max_classification_only_seed231_input_v2_complete.zip
03/codenet_attention_classification_only_seed231_input_v2_complete.zip
```

For CodeNet classification-only / seed123, use the **completed validation-fix
restart** listed above. The earlier failed attempt is not accepted.

## Collect the four new analysis outputs

After the four GPU analyses finish, import `03_collect_review_reports.ipynb`,
select CPU, keep Internet off, and attach their four saved analysis outputs.
The collector needs their report directories, including `analysis_result.json`
and the exported prediction files. It does not need the old training weights
or a training dataset. It verifies report hashes and requires all six CodeNet
conditions across three seeds before producing the complete core summary.

Put downloaded reports here, keeping their generated filenames:

```text
C:\PyProjects\spectrals\outputs\revesion\04_checkpoint_review
```

Expected downloads:

- `01_atcoder_diagnosis_small_reports.zip`
- `02_codenet_f1_seed42_small_reports.zip`
- `02_codenet_f1_seed123_small_reports.zip`
- `02_codenet_f1_seed231_small_reports.zip`
- After the collector: `checkpoint_review_collection_small_reports.zip`

You do not need to download the large training outputs again.

## Purpose and limits

**AtCoder:** compare best and last saved weights across three seeds. The job
measures full-validation performance, performance on a fixed 2,048-pair
training sample, descriptor diversity, slot occupancy, graph density, loss
components and one diagnostic gradient batch. It also checks precision
sensitivity on a fixed sample. Gradient probes do not update weights. A failed
probe is recorded explicitly and does not silently discard the other seeds.

**CodeNet:** export validation probabilities for every pair and fit the F1
threshold using the same procedure for all 18 existing models. Keep the metrics
at the original threshold too. The collector reports means, sample SD and
matched-seed differences across the six readout/objective conditions.

The saved checkpoint epochs were selected by **Accuracy**. Recalibrating a
validation threshold for F1 cannot recover the best F1 epoch or change the old
early-stopping decisions. These are validation analyses, not final-test results.
On a balanced set, predicting every pair as a clone already gives 66.67% F1;
a higher calibrated F1 alone does not resolve near-chance AUROC.

The [registered analysis plan](../../00_plan/F1_CHECKPOINT_REASSESSMENT.md)
defines the comparison rules. The [continuation status](../../00_plan/CONTINUATION_STATUS.md)
tracks the remaining baseline, transfer, final-test and manuscript work.

## Runtime and resuming

The initial planning estimate is **30-90 minutes per GPU analysis**, not a
measured Kaggle runtime. Input reading and GPU conditions can change this.
An estimate of remaining evaluation time is printed after the first model.

Each GPU notebook has a 10-hour internal session budget with a 45-minute
reserve, allowing approximately 9h15 of work before output handling, below
the intended 12-hour Kaggle session limit.

If a run reports `incomplete_session_budget`, keep its output, attach that
previous **analysis output** to the next analysis session, and set
`RESUME_REPORTS_DIR` to its directory under `/kaggle/input`. Keep the original
training checkpoints and dataset attached too. Completed analysis views are
reused after hash verification; an unfinished view is recomputed.

Both accidental BCB/seed231 executions remain preserved as attempts at one
seed. They are outside these four analyses and require no new run now.
