"""Prepare a prospective F1-selected objective comparison, separate from old runs."""
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
    relative = '03_selected_repeats/atcoder/validation_recovery/atcoder_full_seed123_validation_recovery.ipynb'
    original = root / 'revision' / relative
    template = original.read_bytes()
    rows = list(csv.DictReader((root / 'revision/queues/next_independent_roster.csv').open(encoding='utf-8')))
    expected = next(r['notebook_sha256'] for r in rows if r['notebook'] == relative)
    assert hashlib.sha256(template).hexdigest() == expected
    jobs = []
    for seed in (42, 123, 231):
        for arm in ('full', 'classification_only'):
            nb = json.loads(template)
            config = ''.join(nb['cells'][1]['source'])
            assert "ARM = 'full'" in config and 'SEED = 123' in config
            config = config.replace("ARM = 'full'", f'ARM = {arm!r}').replace('SEED = 123', f'SEED = {seed}')
            nb['cells'][1]['source'] = config.splitlines(keepends=True)
            run = ''.join(nb['cells'][3]['source'])
            assert run.count('selection_metric="Accuracy"') == 1
            run = run.replace('selection_metric="Accuracy"', 'selection_metric="F1"')
            run = run.replace('_input_v2_validation_fix"', '_f1_confirmation_v1"')
            nb['cells'][3]['source'] = run.splitlines(keepends=True)
            nb['cells'][0]['source'] = (f'# AtCoder F1 confirmation v1: {arm}, seed{seed}\n\n'
                'Fresh full-data training. Attach only the existing `atcoder_revision_common_support_v2` dataset. '
                'No previous notebook output or checkpoint is required. GPU T4 on; Internet off; FINAL_TEST=False. '
                'This prospective comparison uses F1 for epoch selection, threshold choice and patience for BOTH objectives. '
                'Original Accuracy-selected runs remain separate evidence; never resume them into this protocol. '
                'Shared original architecture, input representation, learning rate, AMP policy and physical/effective batch32/128. '
                '100-epoch ceiling, minimum10, patience10; 10h session with45min output reserve. '
                'A budget pause is incomplete and requires continuation using only this exact job/protocol checkpoint. '
                'Keep all three seeds, including weak outcomes. F1 near66.67% can be an all-positive baseline; inspect AUROC/AP too.\n').splitlines(keepends=True)
            packed = ast.literal_eval(next(n.value for n in ast.parse(''.join(nb['cells'][2]['source'])).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_sources'))
            assert fingerprint({name: packed[name] for name in TRAINING_SOURCES}) == VALIDATION_RECOVERY_HASH
            for c in nb['cells']:
                c['id'] = fingerprint(c['source'])[:12]
                if c['cell_type'] == 'code':compile(''.join(c['source']), 'atcoder_confirmation', 'exec')
            relative_new = f'06_atcoder_confirmation_v1/atcoder_{arm}_seed{seed}_f1_confirmation_v1.ipynb'
            target = root / 'revision' / relative_new
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8', newline='\n')
            jobs.append(dict(job_id=f'atcoder_{arm}_seed{seed}_f1_confirmation_v1', dataset='atcoder', arm=arm, seed=seed,
                notebook=relative_new, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), metric='F1', amp=True,
                max_epochs=100, min_epochs=10, patience=10, physical_batch=32, effective_batch=128,
                dataset_zip='atcoder_revision_common_support_v2.zip', session_hours=10, final_test=False,
                status='ready_not_run', estimates='Approximately 1-6h if patience stops early; about 7.7h extrapolated 100-epoch budget with margin; continuation if necessary'))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(jobs[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(jobs)
    (root / 'revision/queues/atcoder_f1_confirmation_v1.csv').write_text(stream.getvalue(), encoding='utf-8', newline='\n')
    plan = dict(version='atcoder_f1_confirmation_v1', status='registered_not_run', jobs=jobs,
        source_hash=VALIDATION_RECOVERY_HASH, hypothesis='Does classification-only improve full-data learning/stability relative to full losses under matched F1 selection?',
        interpretation='Prospective objective confirmation; not a declared repair, not a spectral-vs-mean/max test; preserve old failed results and all new seeds')
    (root / 'revision/queues/atcoder_f1_confirmation_v1.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8', newline='\n')
    collector = json.loads((ROOT / 'revision/05_analysis/validation_screening.ipynb').read_text(encoding='utf-8'))
    bootstrap = ''.join(collector['cells'][2]['source'])
    assignment = next(n for n in ast.parse(bootstrap).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_sources')
    reporting_sources = ast.literal_eval(assignment.value)
    reporting_sources['screening.py'] = (ROOT / 'spectral_code/revision/screening.py').read_text(encoding='utf-8')
    bootstrap = bootstrap.replace(ast.get_source_segment(bootstrap, assignment), '_sources = ' + repr(reporting_sources), 1)
    collector['cells'][2]['source'] = bootstrap.splitlines(keepends=True)
    collector['cells'][0]['source'] = (
        '# AtCoder F1 confirmation v1: small-report collector\n\n'
        'CPU, Internet off. Attach saved outputs of only the six new confirmation jobs. '
        'No training dataset or checkpoint download is needed. Incomplete outputs are listed, not ranked. '
        'Download `atcoder_confirmation_v1_small_reports.zip`; keep weights saved on Kaggle. '
        'This reports training status; held-out tests remain off.').splitlines(keepends=True)
    run = ''.join(collector['cells'][3]['source'])
    guard = '''# Reject old Accuracy-selected runs and other experiment phases.
import json
for _status_path in INPUT_ROOT.rglob("status.json"):
    _state = json.loads(_status_path.read_text(encoding="utf-8"))
    if not {"identity", "training", "arm"} <= _state.keys():
        continue
    _training = _state["training"]
    if (_training.get("dataset") != "atcoder" or _training.get("selection_metric") != "F1"
        or _training.get("seed") not in {42, 123, 231}
        or _state["arm"] not in {"full", "classification_only"}
        or not _status_path.parent.name.endswith("_f1_confirmation_v1")):
        raise ValueError("Attach only the new six AtCoder F1 confirmation outputs: " + str(_status_path))
'''
    marker = 'from spectral_code.revision.screening import collect_validation'
    assert marker in run
    run = run.replace(marker, guard + '\n' + marker).replace('revision_small_reports.zip', 'atcoder_confirmation_v1_small_reports.zip')
    collector['cells'][3]['source'] = run.splitlines(keepends=True)
    for c in collector['cells']:
        c['id'] = fingerprint(c['source'])[:12]
        if c['cell_type'] == 'code':compile(''.join(c['source']), 'atcoder_collector', 'exec')
    target = root / 'revision/06_atcoder_confirmation_v1/collect_atcoder_f1_confirmation_v1.ipynb'
    target.write_text(json.dumps(collector, ensure_ascii=False, indent=1)+'\n', encoding='utf-8', newline='\n')
    plan['collector'] = dict(notebook=target.relative_to(root / 'revision').as_posix(), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), gpu=False)
    (root / 'revision/queues/atcoder_f1_confirmation_v1.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8', newline='\n')
    assert original.read_bytes() == template
    print('Prepared six fresh F1-selected AtCoder jobs; no training launched and original notebooks unchanged.')
    return jobs


if __name__ == '__main__':
    prepare()
