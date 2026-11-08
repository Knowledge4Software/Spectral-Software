"""Regression coverage for Joern callee text lost by canonicalization."""
import numpy as np
import pytest
from spectral_code.revision.canonical_types import get_canonical_id
from spectral_code.revision.data import canonical_graph_from_record, lexical_source, lexical_feature_ids


@pytest.mark.parametrize('callee', ['write', 'readLine', 'openConnection', 'println'])
def test_java_callee_survives_numeric_export_label(callee):
    record = {'code_id':'sample','graphs':{'ast':{'adjacency':{
        'node_types':['METHOD',callee,'IDENTIFIER'], 'node_labels':['1','2','3'],
        'row':[0,1], 'col':[1,2]}}}}
    graph = canonical_graph_from_record(record,language='java')
    assert graph.canonical_ids[1] == get_canonical_id('CALL')+1
    np.testing.assert_array_equal(graph.lexical_ids[1],lexical_feature_ids(callee))
    assert np.any(graph.lexical_ids[1])
    assert not np.any(graph.lexical_ids[[0,2]])


@pytest.mark.parametrize('tag',['CALL','IDENTIFIER','METHOD','RETURN','<operator>.assignment','method_invocation'])
def test_parser_structural_tags_and_line_numbers_do_not_become_lexical(tag):
    assert lexical_source(tag,'12',get_canonical_id(tag,language='java'),language='java') == ''


def test_source_label_takes_precedence_and_python_fallback_is_preserved():
    assert lexical_source('println','userValue',get_canonical_id('println',language='java'),language='java') == 'userValue'
    assert lexical_source('Name','Name,total',get_canonical_id('Name',language='python'),language='python') == 'total'
    assert lexical_source('unmapped_custom','9',get_canonical_id('unmapped_custom',language='python'),language='python') == 'unmapped_custom'


def test_diagnostics_reject_old_checkpoint_before_reading_new_inputs(monkeypatch):
    from spectral_code.revision import diagnostics
    monkeypatch.setattr(diagnostics,'_load_checkpoint',lambda *a: {'implementation_hash':'old_input'})
    with pytest.raises(ValueError,match='input/runtime differs'):
        diagnostics.validation_diagnostics('unused','unused','unused')
