from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

from spectral_code.faithful_graph_baselines import notebook_runtime as runtime


def test_faithful_baselines_read_clean_data_from_a_kaggle_zip(tmp_path, monkeypatch):
    """The faithful notebooks must accept the same ZIP attachment as SPECTRA."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive_path = input_root / "atcoder_clean_data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "clean_data/pairs.csv.gz",
            gzip.compress(b"left_id,right_id,split,label\na,b,train,1\n"),
        )
        archive.writestr(
            "clean_data/graph_spectra.jsonl.gz",
            gzip.compress(b'{"code_id": "a"}\n'),
        )

    monkeypatch.setitem(runtime.__dict__, "KAGGLE_INPUT_OVERRIDE", str(input_root))
    monkeypatch.setitem(runtime.__dict__, "WORK_DIR_OVERRIDE", str(tmp_path / "working"))

    pairs = runtime._find_input("atcoder_v3", "pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp")
    graphs = runtime._find_input(
        "atcoder_v3",
        "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp",
    )

    assert pairs.is_file()
    assert graphs.is_file()
    with runtime._open_text(pairs) as handle:
        assert "left_id,right_id" in handle.read()
    with runtime._open_text(graphs) as handle:
        assert '"code_id"' in handle.read()
