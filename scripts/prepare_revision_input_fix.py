"""Record the input-v2 amendment and the eight immediately useful Kaggle jobs."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_revision_stage1 import DATASETS
from spectral_code.revision.versioning import training_implementation_hash


def prepare():
    base=ROOT/'revision'
    with (base/'queues/notebook_manifest.csv').open(encoding='utf-8',newline='') as f:
        manifest={r['notebook']:r for r in csv.DictReader(f)}
    with (base/'results/run_progress.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    old={r['job_id']:r for r in csv.DictReader((base/'archive/input_v1/run_progress.csv').open(encoding='utf-8'))}
    for r in rows:
        if r['job_id'] in {f'{d}_full_seed42' for d in DATASETS} and r['status']=='complete_verified':
            previous=old[r['job_id']]
            r.update(status='rerun_required_input_fix',evidence='Input-v2 amendment; historical v1 archive remains verified: '+previous['evidence'],
                completed_epochs='',test_evaluated='False',updated='2026-11-09',
                notes=f"Fresh v2 run required; old v1 completed {previous['completed_epochs']} epochs. {previous['notes']} (do-not-rerun instruction superseded by input fix)")
    with (base/'results/run_progress.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    selected=[(d,'full',f'01_baselines_seed42/{d}/{d}_full_seed42.ipynb') for d in DATASETS]
    selected += [('codenet',arm,f'02_controls_seed42/codenet/{group}/codenet_{arm}_seed42.ipynb') for arm,group in (
        ('mean_max','architecture'),('attention','architecture'),('classification_only','losses'),
        ('mean_max_classification_only','architecture'),('attention_classification_only','architecture'))]
    jobs=[]
    for slot,(dataset,arm,path) in enumerate(selected,1):
        bundle,digest,_=DATASETS[dataset]
        jobs.append(dict(slot=slot,job_id=f'{dataset}_{arm}_seed42',dataset=dataset,arm=arm,seed=42,
            notebook=path,notebook_sha256=manifest[path]['sha256'],dataset_zip=bundle+'.zip',dataset_sha256=digest,
            implementation_hash=training_implementation_hash(),gpu=True,internet=False,
            resume_checkpoint='None',final_test=False))
    with (base/'queues/input_v2_run_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
    lines=['# Run now: input v2, seed 42','','Run these **eight** jobs in parallel if GPU slots are available. Import the updated local files. External baselines can wait for now.','',
        '**Use existing dataset uploads. No new dataset upload or preprocessing job is required.**','',
        '| Slot | Dataset | Notebook | Dataset ZIP already uploaded |','| ---: | --- | --- | --- |']
    for j in jobs:
        lines.append(f"| {j['slot']} | {j['dataset']} | [{j['arm']}](../{j['notebook']}) | `{j['dataset_zip']}` |")
    lines += ['', 'Settings: GPU on (T4 is supported), Internet off, seed 42, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Keep the default 100-epoch ceiling / minimum 10 / patience 10. Each session has a 10-hour budget with 45 minutes reserved for outputs, below the Kaggle 12-hour cap. A checkpointed budget pause may still need continuation.',
        '', 'Use only the matching training dataset as input. These are fresh runs: attach no old SPECTRA output. New SPECTRA output folders end in `_input_v2`; use that output when continuing a v2 run. Preserve every v1 result separately.',
        '', 'The three full-model runs must be repeated because their Java node features changed. All five controls use that same corrected representation. Existing completed external baselines retain their results because their feature preparation is unchanged.',
        '', 'For a previously paused CodeNet CodeBERT/C4/CDLH/Qwen job, use its original notebook under [the v1 archive](../archive/input_v1/README.md) and its own checkpoint/token cache. The new generated runtime cannot resume an old checkpoint. Do not enable numerical-upgrade overrides.',
        '', 'Observed input-v1 full-model times were about 3.1h CodeNet, 4.1h AtCoder and 1.3h BigCloneBench on T4. These are context, not predictions: v2 convergence and control runtimes are not yet measured. Preserve partial outputs if the session guard stops a run.',
        '', 'See [the input-fix amendment](../00_plan/INPUT_V2_FIX.md) for the evidence, compatibility rules and limits. The machine-readable list is `input_v2_run_queue.csv`.','']
    (base/'queues/RUN_NOW_INPUT_V2.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Prepared 8 input-v2 jobs; 3 historical full runs marked for fresh corrected-input runs.')


if __name__=='__main__':
    prepare()
