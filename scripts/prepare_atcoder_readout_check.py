"""Confirm the prespecified strongest simple readout on AtCoder; reuse spectral runs."""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
from pathlib import Path

from spectral_code.revision.protocol import fingerprint
from spectral_code.revision.screening import VALIDATION_RECOVERY_HASH
from spectral_code.revision.versioning import TRAINING_SOURCES, training_implementation_hash

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    root = Path(root)
    assert training_implementation_hash() == VALIDATION_RECOVERY_HASH
    oldplan = json.loads((root/'revision/queues/atcoder_f1_confirmation_v1.json').read_text(encoding='utf-8'))
    jobs = []
    for seed in (42, 123, 231):
        old = next(j for j in oldplan['jobs'] if j['seed']==seed and j['arm']=='classification_only')
        source = root/'revision'/old['notebook']; before=source.read_bytes()
        assert hashlib.sha256(before).hexdigest()==old['sha256']
        nb = json.loads(before)
        config = ''.join(nb['cells'][1]['source'])
        assert "ARM = 'classification_only'" in config
        config=config.replace("ARM = 'classification_only'", "ARM = 'mean_max_classification_only'")
        nb['cells'][1]['source']=config.splitlines(keepends=True)
        run=''.join(nb['cells'][3]['source'])
        assert 'selection_metric="F1"' in run
        run=run.replace('_f1_confirmation_v1"','_readout_check_v1"')
        nb['cells'][3]['source']=run.splitlines(keepends=True)
        nb['cells'][0]['source']=(f'# AtCoder simple-readout confirmation: mean/max classification-only, seed{seed}\n\n'
            'Fresh control experiment, not a replacement proposed method. Attach only the existing AtCoder v2 dataset. '
            'GPU T4, Internet off, FINAL_TEST=False. Compare against the already completed spectral '
            f'classification-only seed{seed} F1 confirmation. Same encoder/input/objective/LR/batch/selection budget. '
            'Readout heads differ in parameter count; benchmarks disclose that difference. '
            '100-epoch ceiling, minimum10, patience10, 10h session including45min reserve. '
            'Do not attach old checkpoints for the initial run. Preserve all weak outcomes.\n').splitlines(keepends=True)
        tree=ast.parse(''.join(nb['cells'][2]['source']))
        packed=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_sources'))
        assert fingerprint({n:packed[n] for n in TRAINING_SOURCES})==VALIDATION_RECOVERY_HASH
        for c in nb['cells']:
            c['id']=fingerprint(c['source'])[:12]
            if c['cell_type']=='code':compile(''.join(c['source']),'readout_check','exec')
        rel=f'07_atcoder_readout_check_v1/atcoder_mean_max_classification_only_seed{seed}_readout_check_v1.ipynb'
        path=root/'revision'/rel;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
        jobs.append(dict(job_id=f'atcoder_mean_max_classification_only_seed{seed}_readout_check_v1',seed=seed,
            arm='mean_max_classification_only',notebook=rel,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reference_job_id=old['job_id'],metric='F1',dataset_zip='atcoder_revision_common_support_v2.zip',
            max_epochs=100,min_epochs=10,patience=10,physical_batch=32,effective_batch=128,amp=True,status='ready_not_run'))
        assert source.read_bytes()==before
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(jobs[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(jobs)
    (root/'revision/queues/atcoder_readout_check_v1.csv').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    plan=dict(version='atcoder_readout_check_v1',jobs=jobs,status='registered_not_run',source_hash=VALIDATION_RECOVERY_HASH,
        selection_basis='Mean/max classification-only had the highest three-seed mean validation F1 among simple CodeNet readouts; selected before new AtCoder readout results.',
        scope='Three-seed confirmation under classification-only objective; original full-objective failures retained; not universal spectral attribution.',
        reuse='All three completed AtCoder spectral classification-only F1 runs; no spectral retraining requested.',
        estimates='Approximately2-6h per T4 notebook if patience stops early; extrapolated cap around8h; budget stop requires continuation.')
    collector_source=root/'revision'/oldplan['collector']['notebook']
    assert hashlib.sha256(collector_source.read_bytes()).hexdigest()==oldplan['collector']['sha256']
    collector=json.loads(collector_source.read_bytes())
    collector['cells'][0]['source']=[
        '# AtCoder readout confirmation: report collector\n\n',
        'CPU, Internet off. Attach the three new mean/max classification-only outputs and the three completed spectral classification-only F1 confirmation outputs. No dataset or checkpoint download required. Download atcoder_readout_check_v1_small_reports.zip. Incomplete runs remain unranked.']
    run=''.join(collector['cells'][3]['source'])
    a=run.index('    if (_training.get("dataset")');b=run.index('\nfrom spectral_code.revision.screening import collect_validation',a)
    guard='''    _allowed_phase = (
        _state["arm"] == "classification_only" and _status_path.parent.name.endswith("_f1_confirmation_v1")
        or _state["arm"] == "mean_max_classification_only" and _status_path.parent.name.endswith("_readout_check_v1"))
    if (_training.get("dataset") != "atcoder" or _training.get("selection_metric") != "F1"
        or _training.get("seed") not in {42, 123, 231} or not _allowed_phase):
        raise ValueError("Attach only the three spectral and three mean/max classification-only outputs: " + str(_status_path))
'''
    run=run[:a]+guard+run[b:]
    run=run.replace('atcoder_confirmation_v1_small_reports.zip','atcoder_readout_check_v1_small_reports.zip')
    collector['cells'][3]['source']=run.splitlines(keepends=True)
    for c in collector['cells']:
        c['id']=fingerprint(c['source'])[:12]
        if c['cell_type']=='code':compile(''.join(c['source']),'readout_collector','exec')
    target=root/'revision/07_atcoder_readout_check_v1/collect_atcoder_readout_check_v1.ipynb'
    target.write_text(json.dumps(collector,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
    plan['collector']=dict(notebook=target.relative_to(root/'revision').as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),gpu=False)
    (root/'revision/queues/atcoder_readout_check_v1.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Prepared three mean/max controls; reuse all three completed spectral references. No training launched.')
    return jobs


if __name__=='__main__':prepare()
