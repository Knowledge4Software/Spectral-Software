import ast
import hashlib
import json
import shutil

from scripts.prepare_atcoder_confirmation import ROOT, prepare


def test_f1_confirmation_is_separate_matched_and_frozen_inputs_are_preserved(tmp_path):
    for rel in ['queues/next_independent_roster.csv',
                '03_selected_repeats/atcoder/validation_recovery/atcoder_full_seed123_validation_recovery.ipynb']:
        target = tmp_path / 'revision' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'revision' / rel, target)
    source = tmp_path / 'revision/03_selected_repeats/atcoder/validation_recovery/atcoder_full_seed123_validation_recovery.ipynb'
    before = source.read_bytes()
    jobs = prepare(tmp_path)
    assert len(jobs) == 6 and {(j['arm'], j['seed']) for j in jobs} == {
        (a, s) for a in ('full', 'classification_only') for s in (42, 123, 231)}
    for j in jobs:
        path = tmp_path / 'revision' / j['notebook']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == j['sha256']
        n = json.loads(path.read_text(encoding='utf-8'))
        config, run = ''.join(n['cells'][1]['source']), ''.join(n['cells'][3]['source'])
        assert 'RESUME_CHECKPOINT = None' in config and 'FINAL_TEST = False' in config
        assert f'ARM = {j["arm"]!r}' in config and f'SEED = {j["seed"]}' in config
        assert 'selection_metric="F1"' in run and '_f1_confirmation_v1' in run
        assert 'selection_metric="Accuracy"' not in run and j['amp'] is True
    assert source.read_bytes() == before
    collector = json.loads((tmp_path / 'revision/06_atcoder_confirmation_v1/collect_atcoder_f1_confirmation_v1.ipynb').read_text(encoding='utf-8'))
    bootstrap = ast.parse(''.join(collector['cells'][2]['source']))
    packed = ast.literal_eval(next(n.value for n in bootstrap.body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_sources'))
    assert packed['screening.py'] == (ROOT / 'spectral_code/revision/screening.py').read_text(encoding='utf-8')
    assert 'selection_metric' in ''.join(collector['cells'][3]['source'])
