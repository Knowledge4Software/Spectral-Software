"""A refreshed run list must never reset corrected full-model completions."""
import csv
from pathlib import Path
import shutil
import pytest
from scripts.prepare_revision_input_fix import prepare
from spectral_code.revision.versioning import training_implementation_hash


def test_input_fix_queue_keeps_current_full_completions_and_is_idempotent(tmp_path):
    repo=Path(__file__).resolve().parents[1]/'revision'
    for directory in ('queues','results','archive/input_v1'):
        (tmp_path/directory).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(repo/'queues/notebook_manifest.csv',tmp_path/'queues/notebook_manifest.csv')
    fields=['job_id','status','evidence','completed_epochs','test_evaluated','updated','notes','implementation_hash']
    rows=[dict(job_id=f'{d}_full_seed42',status='complete_verified',evidence='new verified archive',completed_epochs='24',
        test_evaluated='False',updated='2026-11-09',notes='current input-v2',implementation_hash=training_implementation_hash())
        for d in ('codenet','atcoder','bigclonebench')]
    for target in ('results/run_progress.csv','archive/input_v1/run_progress.csv'):
        with (tmp_path/target).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    prepare(tmp_path)
    first=(tmp_path/'results/run_progress.csv').read_bytes()
    prepare(tmp_path)
    assert (tmp_path/'results/run_progress.csv').read_bytes()==first
    saved=list(csv.DictReader((tmp_path/'results/run_progress.csv').open(encoding='utf-8')))
    assert all(r['status']=='complete_verified' for r in saved)
    jobs=list(csv.DictReader((tmp_path/'queues/input_v2_run_queue.csv').open(encoding='utf-8')))
    assert len(jobs)==5 and all(r['dataset']=='codenet' and r['arm']!='full' for r in jobs)
    controls=[]
    for arm in ('mean_max','attention','classification_only','mean_max_classification_only','attention_classification_only'):
        controls.append(dict(rows[0],job_id=f'codenet_{arm}_seed42'))
    with (tmp_path/'results/control_progress.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(controls)
    prepare(tmp_path)
    jobs=list(csv.DictReader((tmp_path/'queues/input_v2_run_queue.csv').open(encoding='utf-8')))
    assert jobs==[]
    assert 'RUN_NEXT_CORE_REPEATS.md' in (tmp_path/'queues/RUN_NOW_INPUT_V2.md').read_text()


def test_core_repeats_use_only_two_new_seeds_and_require_completed_core(tmp_path):
    from scripts.prepare_revision_core_repeats import CORE,prepare as prepare_repeats
    repo=Path(__file__).resolve().parents[1]/'revision'
    (tmp_path/'queues').mkdir();(tmp_path/'results').mkdir()
    manifest=list(csv.DictReader((repo/'queues/notebook_manifest.csv').open(encoding='utf-8')))
    paths={f'03_selected_repeats/codenet/{group}/codenet_{arm}_seed{seed}.ipynb' for arm,group in CORE for seed in (123,231)}
    for path in paths:
        (tmp_path/path).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/path,tmp_path/path)
    with (tmp_path/'queues/notebook_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows([r for r in manifest if r['notebook'] in paths])
    fields=['job_id','status','implementation_hash']
    controls=[dict(job_id=f'codenet_{arm}_seed42',status='complete_verified',implementation_hash=training_implementation_hash()) for arm,_ in CORE]
    for file,rows in (('run_progress.csv',controls[:1]),('control_progress.csv',controls[1:])):
        with (tmp_path/'results'/file).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    jobs=prepare_repeats(tmp_path)
    assert len(jobs)==12 and {r['seed'] for r in jobs}=={123,231}
    assert all(sum(r['arm']==arm for r in jobs)==2 for arm,_ in CORE)
    controls[1]['status']='incomplete_session_budget'
    with (tmp_path/'results/control_progress.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(controls[1:])
    with pytest.raises(ValueError,match='Finish and verify'):
        prepare_repeats(tmp_path)
