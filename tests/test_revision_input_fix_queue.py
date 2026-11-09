"""A refreshed run list must never reset corrected full-model completions."""
import csv
from pathlib import Path
import shutil
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
