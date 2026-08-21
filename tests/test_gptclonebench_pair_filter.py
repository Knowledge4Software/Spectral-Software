from spectral_code.evaluation.v3_benchmark_preparation import (
    _prepare_gptclonebench_non_self_pairs,
)


def _rows(split: str, anchor: str):
    base = {
        "anchor_author_unit_id": anchor,
        "partner_author_unit_id": f"{anchor}-partner",
        "source_pool_id_1": f"{split}-{anchor}-pool-a",
        "source_pool_id_2": f"{split}-{anchor}-pool-b",
        "source_original_content_sha256_1": f"{split}-{anchor}-hash-a",
        "source_original_content_sha256_2": f"{split}-{anchor}-hash-b",
    }
    return [
        {**base, "pair_id": f"{anchor}-self-a", "pair_kind": "type1_self_g1", "code_id_1": f"{anchor}-a", "code_id_2": f"{anchor}-a", "label": 1},
        {**base, "pair_id": f"{anchor}-self-b", "pair_kind": "type1_self_g2", "code_id_1": f"{anchor}-b", "code_id_2": f"{anchor}-b", "label": 1},
        {**base, "pair_id": f"{anchor}-positive", "pair_kind": "type4_g1_g2", "code_id_1": f"{anchor}-a", "code_id_2": f"{anchor}-b", "label": 1},
        *[
            {**base, "pair_id": f"{anchor}-negative-{index}", "pair_kind": f"negative_{index}", "code_id_1": f"{anchor}-a", "code_id_2": f"{anchor}-n{index}", "label": 0}
            for index in range(3)
        ],
    ]


def test_gptclonebench_filter_is_balanced_self_free_and_deterministic():
    source = {
        split: _rows(split, f"{split}-anchor-1") + _rows(split, f"{split}-anchor-2")
        for split in ("train", "valid", "test")
    }
    first, audit = _prepare_gptclonebench_non_self_pairs(source)
    second, _ = _prepare_gptclonebench_non_self_pairs(source)

    assert first == second
    for split, rows in first.items():
        assert len(rows) == 4
        assert sum(row["label"] == 1 for row in rows) == 2
        assert sum(row["label"] == 0 for row in rows) == 2
        assert all(row["code_id_1"] != row["code_id_2"] for row in rows)
        assert audit["splits"][split]["removed_self_pairs"] == 4
        assert audit["splits"][split]["removed_excess_negatives"] == 4
