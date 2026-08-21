from __future__ import annotations

import csv
import gzip

import pytest

from spectral_code.evaluation.clean_data_export import write_clean_pairs


def test_clean_pair_export_preserves_aligned_provenance(tmp_path):
    splits = {
        "train": [(1, 2, 1), (3, 4, 0)],
        "valid": [(5, 6, 1), (7, 8, 0)],
        "test": [(9, 10, 1), (11, 12, 0)],
    }
    provenance = {
        split: [
            {"pair_kind": "clone"},
            {"pair_kind": "mutation_operator_swap", "is_mutation": True},
        ]
        for split in splits
    }

    counts = write_clean_pairs(tmp_path, splits, pair_metadata=provenance)

    assert counts == {"train": 2, "valid": 2, "test": 2}
    with gzip.open(tmp_path / "pairs.csv.gz", "rt", encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))
    assert rows[0]["pair_kind"] == "clone"
    assert rows[1]["pair_kind"] == "mutation_operator_swap"
    assert rows[1]["is_mutation"] == "True"
    assert rows[1]["label"] == "0"


def test_clean_pair_export_rejects_misaligned_provenance(tmp_path):
    with pytest.raises(ValueError, match="expected 2"):
        write_clean_pairs(
            tmp_path,
            {"train": [(1, 2, 1), (3, 4, 0)]},
            pair_metadata={"train": [{"pair_kind": "clone"}]},
        )
