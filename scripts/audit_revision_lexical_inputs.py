"""Compare old/new node features on existing validation data without training."""
from __future__ import annotations
import argparse
import ast
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_code.revision import data
from spectral_code.revision.training import load_pair_data
from spectral_code.revision.versioning import training_implementation_hash


def audit(output):
    notebook = json.loads((ROOT/'revision/archive/input_v1/01_baselines_seed42/codenet/codenet_full_seed42.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(''.join(notebook['cells'][2]['source']))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_sources')
    old_sources=ast.literal_eval(assignment.value)
    fn=next(n for n in ast.parse(old_sources['data.py']).body if isinstance(n,ast.FunctionDef) and n.name=='lexical_source')
    # Only our preserved local function; never execute external ZIP sources.
    namespace=dict(vars(data))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'preserved_v1_lexical_source','exec'),namespace)
    old=namespace['lexical_source']
    reports={}
    for dataset in ('bigclonebench','atcoder','codenet'):
        bundle=dataset+'_revision_common_support'+('' if dataset=='codenet' else '_v2')
        corpus,frames,digest=load_pair_data(ROOT.parent/'data/working/revision'/bundle,require_disjoint=dataset=='codenet')
        valid=frames['valid']; ids=set(valid.left_id.astype(str))|set(valid.right_id.astype(str))
        if dataset!='bigclonebench':
            ids={i for i in ids if corpus.code_languages[i]=='java'}
        counts=Counter(); examples=[]
        with data._open_text(corpus.graphs_path) as stream:
            for line in stream:
                row=json.loads(line); key=str(row.get('code_id',row.get('id','')))
                if key not in ids: continue
                layer=row.get('graphs',{}).get('ast',{}); layer=layer.get('adjacency',layer)
                types=layer.get('node_types',[])[:256]; labels=layer.get('node_labels',[])
                language=corpus.code_languages[key]
                if not types: continue
                counts['programs']+=1
                for i,t in enumerate(types):
                    label=labels[i] if i<len(labels) else ''
                    canonical=data.get_canonical_id(str(t),language=language)
                    before=data.lexical_feature_ids(old(t,label,canonical))
                    after=data.lexical_feature_ids(data.lexical_source(t,label,canonical,language=language))
                    counts['nodes']+=1
                    counts['before_nonempty_nodes']+=int(np.any(before))
                    counts['after_nonempty_nodes']+=int(np.any(after))
                    changed=not np.array_equal(before,after)
                    counts['changed_nodes']+=changed
                    if changed and len(examples)<5: examples.append(str(t))
                if dataset!='bigclonebench' and counts['programs']>=32: break
        assert counts['changed_nodes']>0
        assert counts['after_nonempty_nodes']>counts['before_nonempty_nodes']
        if dataset=='bigclonebench':
            assert counts['programs']==len(ids)==127
            assert counts['before_nonempty_nodes']==3 and counts['after_nonempty_nodes']==1927
        reports[dataset]=dict(data_hash=digest,scope='all validation programs' if dataset=='bigclonebench' else 'first 32 Java validation graphs in export order; diagnostic sample only',counts=dict(counts),examples=examples)
        print(dataset,json.dumps(reports[dataset]),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(training_implementation_hash=training_implementation_hash(),datasets=reports),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/revision/input_v2/lexical_audit.json')
    audit(parser.parse_args().output)
