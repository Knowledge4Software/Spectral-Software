"""Record the input-v2 amendment and queue only unfinished priority jobs."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_revision_stage1 import DATASETS
from spectral_code.revision.versioning import training_implementation_hash


def prepare(root=None):
    base=Path(root) if root is not None else ROOT/'revision'
    with (base/'queues/notebook_manifest.csv').open(encoding='utf-8',newline='') as f:
        manifest={r['notebook']:r for r in csv.DictReader(f)}
    with (base/'results/run_progress.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    if 'implementation_hash' not in fields:
        fields.append('implementation_hash')
    current_hash=training_implementation_hash()
    old={r['job_id']:r for r in csv.DictReader((base/'archive/input_v1/run_progress.csv').open(encoding='utf-8'))}
    for r in rows:
        if (r['job_id'] in {f'{d}_full_seed42' for d in DATASETS} and r['status']=='complete_verified'
                and r.get('implementation_hash') != current_hash):
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
    progress={r['job_id']:r for r in rows}
    completed=[]
    for dataset,arm,path in selected:
        job_id=f'{dataset}_{arm}_seed42'
        known=progress.get(job_id,{})
        if known.get('status')=='complete_verified' and known.get('implementation_hash')==current_hash:
            completed.append(job_id)
            continue
        bundle,digest,_=DATASETS[dataset]
        jobs.append(dict(slot=len(jobs)+1,job_id=job_id,dataset=dataset,arm=arm,seed=42,
            notebook=path,notebook_sha256=manifest[path]['sha256'],dataset_zip=bundle+'.zip',dataset_sha256=digest,
            implementation_hash=training_implementation_hash(),gpu=True,internet=False,
            resume_checkpoint='None',final_test=False))
    with (base/'queues/input_v2_run_queue.csv').open('w',encoding='utf-8',newline='') as f:
        job_fields=('slot','job_id','dataset','arm','seed','notebook','notebook_sha256','dataset_zip','dataset_sha256','implementation_hash','gpu','internet','resume_checkpoint','final_test')
        w=csv.DictWriter(f,fieldnames=job_fields);w.writeheader();w.writerows(jobs)
    lines=['# Run now: input v2, seed 42','',f'Run these **{len(jobs)}** unfinished priority jobs in parallel if GPU slots are available. Import the updated local files. External baselines can wait for now.','',
        '**Use existing dataset uploads. No new dataset upload or preprocessing job is required.**','',
        '| Slot | Dataset | Notebook | Dataset ZIP already uploaded |','| ---: | --- | --- | --- |']
    for j in jobs:
        lines.append(f"| {j['slot']} | {j['dataset']} | [{j['arm']}](../{j['notebook']}) | `{j['dataset_zip']}` |")
    lines += ['', 'Settings: GPU on (T4 is supported), Internet off, seed 42, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Keep the default 100-epoch ceiling / minimum 10 / patience 10. Each session has a 10-hour budget with 45 minutes reserved for outputs, below the Kaggle 12-hour cap. A checkpointed budget pause may still need continuation.',
        '', 'Use only the matching training dataset as input. These are fresh runs: attach no old SPECTRA output. New SPECTRA output folders end in `_input_v2`; use that output when continuing a v2 run. Preserve every v1 result separately.',
        '', 'The input-v2 full models and all five controls use the corrected Java node features. Verified current full-model outputs are excluded from this list: reuse them. Existing completed external baselines also retain their results because their feature preparation is unchanged.',
        '', 'For a previously paused CodeNet CodeBERT/C4/CDLH/Qwen job, use its original notebook under [the v1 archive](../archive/input_v1/README.md) and its own checkpoint/token cache. The new generated runtime cannot resume an old checkpoint. Do not enable numerical-upgrade overrides.',
        '', 'Control runtimes are not yet measured. Preserve partial outputs if the session guard stops a run.',
        '', 'Completed input-v2 priority jobs: '+(', '.join(completed) if completed else 'none reported yet')+'.',
        '', 'See [the current full-model review](../results/SPECTRA_INPUT_V2_REVIEW.md) and [input-fix amendment](../00_plan/INPUT_V2_FIX.md). The machine-readable list is `input_v2_run_queue.csv`.','']
    (base/'queues/RUN_NOW_INPUT_V2.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Prepared {len(jobs)} unfinished input-v2 jobs; {len(completed)} verified current jobs excluded.')


if __name__=='__main__':
    prepare()
