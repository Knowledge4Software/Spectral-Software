"""Queue the preplanned six CodeNet conditions with two additional seeds."""
from __future__ import annotations
import ast,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_code.revision.versioning import TRAINING_SOURCES
from spectral_code.revision.screening import INPUT_V2_HASH,comparison_implementation_hash
from spectral_code.revision.protocol import fingerprint
from scripts.prepare_revision_stage1 import DATASETS

CORE=(('full','baselines'),('mean_max','architecture'),('attention','architecture'),
      ('classification_only','losses'),('mean_max_classification_only','architecture'),('attention_classification_only','architecture'))

def prepare(root=None):
    base=Path(root) if root is not None else ROOT/'revision'
    with (base/'queues/notebook_manifest.csv').open(encoding='utf-8',newline='') as f:
        manifest={r['notebook']:r for r in csv.DictReader(f)}
    progress={}
    for file in ('run_progress.csv','control_progress.csv'):
        with (base/'results'/file).open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                if r['job_id'] in progress:raise ValueError('Duplicate progress job')
                progress[r['job_id']]=r
    # Frozen experiment notebooks remain valid after a numerical-only recovery.
    code=INPUT_V2_HASH
    for arm,_ in CORE:
        row=progress[f'codenet_{arm}_seed42']
        if row['status']!='complete_verified' or comparison_implementation_hash(dict(row,arm=arm))!=code:
            raise ValueError(f'Finish and verify the current seed-42 core before repeats: {arm}')
    jobs=[]
    for arm,group in CORE:
        for seed in (123,231):
            relative=f'03_selected_repeats/codenet/{group}/codenet_{arm}_seed{seed}.ipynb'
            path=base/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=manifest[relative]['sha256']:raise ValueError(f'Notebook hash differs: {relative}')
            nb=json.loads(path.read_text(encoding='utf-8'))
            assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
            if assignments['ARM']!=arm or assignments['SEED']!=seed or assignments['RESUME_CHECKPOINT'] is not None or assignments['FINAL_TEST'] is not False:
                raise ValueError(f'Unexpected run settings: {relative}')
            if tuple(assignments[k] for k in ('MAX_EPOCHS','MIN_EPOCHS','PATIENCE','SESSION_HOURS'))!=(100,10,10,10.0):
                raise ValueError(f'Training schedule differs from seed 42: {relative}')
            tree=ast.parse(''.join(nb['cells'][2]['source']))
            node=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_sources')
            packed=ast.literal_eval(node.value)
            if fingerprint({n:packed[n] for n in TRAINING_SOURCES})!=code:raise ValueError(f'Runtime differs from seed 42: {relative}')
            jobs.append(dict(slot=len(jobs)+1,job_id=f'codenet_{arm}_seed{seed}',dataset='codenet',arm=arm,seed=seed,
                notebook=relative,notebook_sha256=digest,dataset_zip=DATASETS['codenet'][0]+'.zip',dataset_sha256=DATASETS['codenet'][1],
                implementation_hash=code,gpu=True,internet=False,resume_checkpoint='None',final_test=False))
    fields=list(jobs[0])
    with (base/'queues/core_repeat_roster.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(jobs)
    reported={}
    tracker=base/'results/core_repeat_progress.csv'
    if tracker.exists():
        with tracker.open(encoding='utf-8',newline='') as f:
            reported={r['job_id']:r for r in csv.DictReader(f)}
        if reported.keys()-{r['job_id'] for r in jobs}:raise ValueError('Repeat tracker contains jobs outside the core roster')
    fresh=[];attention=[];completed=[]
    for job in jobs:
        known=reported.get(job['job_id'],{})
        status=known.get('status','not_reported')
        if status=='complete_verified' and comparison_implementation_hash(dict(known,arm=job['arm']))==code:
            completed.append(job['job_id'])
        elif status not in ('not_reported','not_started',''):
            attention.append(dict(job,saved_status=status,action='Check original Kaggle run state/error; continue its own checkpoint only after diagnosing the stop'))
        else:fresh.append(job)
    recovery_path=base/'queues/validation_recovery_manifest.csv'
    recovery={}
    if recovery_path.exists():
        with recovery_path.open(encoding='utf-8',newline='') as f:
            recovery={r['job_id']:r for r in csv.DictReader(f)}
    for job in fresh+attention:
        fixed=recovery.get(job['job_id'])
        if fixed:
            if hashlib.sha256((base/fixed['notebook']).read_bytes()).hexdigest()!=fixed['notebook_sha256']:
                raise ValueError('Recovery notebook hash differs')
            for key in ('notebook','notebook_sha256','implementation_hash','resume_checkpoint'):job[key]=fixed[key]
            if 'action' in job:
                job['action']=('Restart from epoch 1 with the fixed notebook and CodeNet data only; failed checkpoint unavailable.'
                               if fixed['resume_checkpoint']=='None' else
                               'Use the recovery notebook with its own failed output; retry epoch 43 validation before further training.')
    for slot,job in enumerate(fresh,1):job['slot']=slot
    with (base/'queues/core_repeat_run_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(fresh)
    with (base/'queues/core_repeat_resume_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+['saved_status','action']);w.writeheader();w.writerows(attention)
    lines=['# Remaining CodeNet core repeats','',
        f'Original roster: 12 jobs. **{len(completed)} verified complete, {len(attention)} incomplete/needs attention, {len(fresh)} without submitted results.** Reuse completed outputs. No seed-42 job is repeated.','',
        '**Every notebook uses the existing `codenet_revision_common_support.zip` dataset.** These are the seed-123/231 repetitions, separate from the remaining seed-42 ablations.','',
        '## No output submitted','',
        '| Slot | Condition | Seed | Notebook |','| ---: | --- | ---: | --- |']
    for j in fresh:lines.append(f"| {j['slot']} | {j['arm']} | {j['seed']} | [{Path(j['notebook']).name}](../{j['notebook']}) |")
    lines += ['', 'If one of these is already running on Kaggle, let that existing job finish. No local output does not establish that a job has never started.','',
        '## Incomplete / needs attention','', '| Condition | Seed | Saved status | Action |','| --- | ---: | --- | --- |']
    for j in attention:lines.append(f"| [{j['arm']}](../{j['notebook']}) | {j['seed']} | {j['saved_status']} | {j['action']} |")
    lines += ['', 'For the failed seed 123 follow [validation recovery instructions](../00_plan/VALIDATION_NUMERICAL_RECOVERY.md). The current queue uses a fresh restart because its saved Kaggle checkpoint is unavailable. Attach only CodeNet data. The separately preserved resume notebook is optional if that checkpoint becomes available.',
        '', 'For fresh jobs only: use GPU (T4 supported), Internet off, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Start each new seed from scratch without old model outputs attached. Keep the encoded seed and all model/loss/training settings unchanged. The 100-epoch ceiling, minimum 10 and patience 10 stay fixed; each session uses the existing 10-hour budget and 45-minute output reserve. A budget pause may still require continuation with that run\'s own checkpoint.',
        '', 'Seed-42 recorded times for the six conditions ranged from about 1.7h to 6.0h on T4, including setup and training/validation. Other seeds can stop at different epochs. Timing is an estimate; preserve incomplete outputs.',
        '', 'The purpose is to estimate the readout/objective interaction across seeds. Mean/max classification-only exceeded full by only 10 net correct validation pairs at seed 42; attention classification-only exceeded it by two. Do not infer significance or paired error overlap from those aggregate counts.',
        '', 'The other 15 seed-42 controls, external finalists and transfer study remain in the roadmap. This batch brings the already planned core replication forward before those more costly extensions. Final tests remain off.',
        '', 'See [the current repeat progress](../results/CORE_REPEATS_PROGRESS.md) and [seed-42 results](../results/CORE_CONTROLS_REVIEW.md). `core_repeat_run_queue.csv` contains only jobs without submitted results; `core_repeat_resume_queue.csv` separates incomplete jobs; `core_repeat_roster.csv` retains all 12 planned jobs.','']
    (base/'queues/RUN_NEXT_CORE_REPEATS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Core repeats: {len(completed)} complete, {len(attention)} need attention, {len(fresh)} without output. Frozen roster verified; recovery notebooks have separate provenance.')
    return fresh

if __name__=='__main__':prepare()
