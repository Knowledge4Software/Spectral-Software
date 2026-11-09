"""Queue the preplanned six CodeNet conditions with two additional seeds."""
from __future__ import annotations
import ast,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_code.revision.versioning import TRAINING_SOURCES,training_implementation_hash
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
    code=training_implementation_hash()
    for arm,_ in CORE:
        row=progress[f'codenet_{arm}_seed42']
        if row['status']!='complete_verified' or row['implementation_hash']!=code:
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
    with (base/'queues/core_repeat_run_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
    lines=['# Next: 12 CodeNet core repeats','',
        'The six seed-42 conditions are verified complete. This is the 12-run CodeNet core portion of the preplanned 28 selected-repeat jobs; it does not expand the original scope. Run the same six conditions with seeds 123 and 231. No seed-42 run is repeated.','',
        '**Every notebook uses the existing `codenet_revision_common_support.zip` dataset.** All 12 can run in parallel with enough GPU slots.','',
        '| Slot | Condition | Seed | Notebook |','| ---: | --- | ---: | --- |']
    for j in jobs:lines.append(f"| {j['slot']} | {j['arm']} | {j['seed']} | [{Path(j['notebook']).name}](../{j['notebook']}) |")
    lines += ['', 'Use GPU (T4 supported), Internet off, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Start each new seed from scratch without old model outputs attached. Keep the encoded seed and all model/loss/training settings unchanged. The 100-epoch ceiling, minimum 10 and patience 10 stay fixed; each session uses the existing 10-hour budget and 45-minute output reserve. A budget pause may still require continuation with that run’s own checkpoint.',
        '', 'Seed-42 recorded times for the six conditions ranged from about 1.7h to 6.0h on T4, including setup and training/validation. Other seeds can stop at different epochs. Timing is an estimate; preserve incomplete outputs.',
        '', 'The purpose is to estimate the readout/objective interaction across seeds. Mean/max classification-only exceeded full by only 10 net correct validation pairs at seed 42; attention classification-only exceeded it by two. Do not infer significance or paired error overlap from those aggregate counts.',
        '', 'The other 15 seed-42 controls, external finalists and transfer study remain in the roadmap. This batch brings the already planned core replication forward before those more costly extensions. Final tests remain off.',
        '', 'See [the audited seed-42 results](../results/CORE_CONTROLS_REVIEW.md). The machine-readable list is `core_repeat_run_queue.csv`.','']
    (base/'queues/RUN_NEXT_CORE_REPEATS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Prepared 12 preplanned CodeNet core repeats; all notebooks match the seed-42 runtime.')
    return jobs

if __name__=='__main__':prepare()
