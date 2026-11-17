import hashlib,json,shutil
from scripts.prepare_atcoder_readout_check import ROOT,prepare


def test_only_new_readout_is_trained_with_matched_f1_settings(tmp_path):
    manifest=json.loads((ROOT/'revision/queues/atcoder_f1_confirmation_v1.json').read_text())
    paths=['queues/atcoder_f1_confirmation_v1.json',manifest['collector']['notebook']]+[j['notebook'] for j in manifest['jobs'] if j['arm']=='classification_only']
    before={}
    for rel in paths:
        dest=tmp_path/'revision'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'revision'/rel,dest);before[rel]=dest.read_bytes()
    jobs=prepare(tmp_path)
    assert len(jobs)==3 and {j['seed'] for j in jobs}=={42,123,231}
    for j in jobs:
        p=tmp_path/'revision'/j['notebook'];assert hashlib.sha256(p.read_bytes()).hexdigest()==j['sha256']
        nb=json.loads(p.read_text());config=''.join(nb['cells'][1]['source']);run=''.join(nb['cells'][3]['source'])
        assert "ARM = 'mean_max_classification_only'" in config
        assert 'RESUME_CHECKPOINT = None' in config and 'FINAL_TEST = False' in config
        assert 'selection_metric="F1"' in run and '_readout_check_v1' in run
        for c in nb['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),p.name,'exec')
    assert all((tmp_path/'revision'/rel).read_bytes()==data for rel,data in before.items())
