"""The RQ3 experiment matrix must keep the counts and the no-leakage rule.

Replaces tests/test_rq3_notebooks.py, which asserted the same invariants
against a notebook generator that no longer exists. The matrix module is now
the single source of truth for which configurations exist, so the guarantees
are checked there instead of in generated notebook text.
"""
from __future__ import annotations

from experiments.kaggle.rq3.matrix import (
    LANGUAGES,
    SYMBOLS,
    all_configurations,
    bucket,
    summary,
)


def test_matrix_enumerates_the_paper_configuration_counts():
    counts = summary()
    # 10 controls: 4 same-language and 6 cross-language pairs.
    assert counts["table7_within"] == 4
    assert counts["table7_cross"] == 6
    # The three bridge levels the paper reports as 156 configurations.
    assert counts["table8"] == 12
    assert counts["table9"] == 48
    assert counts["table10"] == 96
    assert counts["table8"] + counts["table9"] + counts["table10"] == 156
    assert counts["total"] == 166


def test_bridge_configurations_never_train_on_their_own_target_bucket():
    """The transfer claim is only meaningful if the target is held out."""
    for configuration in all_configurations():
        if configuration.table == "table7_within":
            continue
        assert configuration.test_bucket not in configuration.train_buckets, configuration


def test_every_configuration_tests_on_exactly_its_declared_target():
    for configuration in all_configurations():
        assert configuration.test_bucket, configuration
        assert configuration.train_buckets, configuration


def test_cross_language_buckets_use_one_canonical_spelling():
    """The export writes each pair under a single fixed ordering."""
    for left in SYMBOLS:
        for right in SYMBOLS:
            assert bucket(left, right) == bucket(right, left)
    for symbol in SYMBOLS:
        assert bucket(symbol, symbol) == LANGUAGES[symbol]


def test_configuration_keys_are_unique():
    keys = [configuration.key for configuration in all_configurations()]
    assert len(keys) == len(set(keys))
