from collections import Counter

from spectral_code.evaluation.codenet_preparation import (
    CONFIGURATIONS,
    DEFAULT_SAMPLE_SIZE,
    PAIR_KINDS,
    SPLITS,
    _source_line_count,
    _sample_targets,
    _capacity_aware_sample_targets,
)


def test_source_line_count_uses_inclusive_physical_lines():
    assert _source_line_count("one") == 1
    assert _source_line_count("one\ntwo") == 2
    assert _source_line_count("one\n\nthree\n") == 3


def test_capacity_aware_targets_preserve_language_totals_and_global_splits():
    configurations = ("python", "python_csharp")
    pair_kinds = ("clone",)
    capacities = {
        ("python", "clone", "train"): 1_000,
        ("python", "clone", "valid"): 1_000,
        ("python", "clone", "test"): 1_000,
        ("python_csharp", "clone", "train"): 1_000,
        ("python_csharp", "clone", "valid"): 20,
        ("python_csharp", "clone", "test"): 10,
    }
    targets = _capacity_aware_sample_targets(800, configurations, pair_kinds, capacities)

    for configuration in configurations:
        assert sum(targets[(configuration, "clone", split)] for split in SPLITS) == 400
    assert {split: sum(targets[(configuration, "clone", split)] for configuration in configurations) for split in SPLITS} == {
        "train": 560,
        "valid": 120,
        "test": 120,
    }
    assert targets[("python_csharp", "clone", "valid")] <= 20
    assert targets[("python_csharp", "clone", "test")] <= 10


def _counts(targets, configuration):
    by_kind = Counter()
    by_split = Counter()
    for (current_configuration, pair_kind, split), count in targets.items():
        if current_configuration != configuration:
            continue
        by_kind[pair_kind] += count
        by_split[split] += count
    return by_kind, by_split


def test_thirty_thousand_variant_is_uniform_across_all_release_buckets():
    targets = _sample_targets(30_000, CONFIGURATIONS, PAIR_KINDS)

    assert sum(targets.values()) == 30_000
    for configuration in CONFIGURATIONS:
        by_kind, by_split = _counts(targets, configuration)
        assert by_kind == {
            "clone": 750,
            "nonclone_diff_problem": 750,
            "hard_nonclone": 750,
            "nonclone_mutation": 750,
        }
        assert by_split == {"train": 2_100, "valid": 450, "test": 450}


def test_default_profile_retains_the_complete_presplit_release():
    assert DEFAULT_SAMPLE_SIZE is None


def test_eight_hundred_smoke_profile_preserves_every_bucket_and_split():
    targets = _sample_targets(800, CONFIGURATIONS, PAIR_KINDS)

    assert sum(targets.values()) == 800
    for configuration in CONFIGURATIONS:
        by_kind, by_split = _counts(targets, configuration)
        assert set(by_kind.values()) == {20}
        assert by_split == {"train": 56, "valid": 12, "test": 12}


def test_six_thousand_subset_has_exact_uniform_bucket_distribution():
    targets = _sample_targets(6_000, CONFIGURATIONS, PAIR_KINDS)

    assert sum(targets.values()) == 6_000
    for configuration in CONFIGURATIONS:
        by_kind, by_split = _counts(targets, configuration)
        assert by_kind == {
            "clone": 150,
            "nonclone_diff_problem": 150,
            "hard_nonclone": 150,
            "nonclone_mutation": 150,
        }
        assert by_split == {"train": 420, "valid": 90, "test": 90}


def test_twenty_thousand_variant_stays_equal_by_configuration_kind_and_split():
    targets = _sample_targets(20_000, CONFIGURATIONS, PAIR_KINDS)

    assert sum(targets.values()) == 20_000
    for configuration in CONFIGURATIONS:
        by_kind, by_split = _counts(targets, configuration)
        assert sum(by_kind.values()) == 2_000
        assert set(by_kind.values()) == {500}
        assert sum(by_split.values()) == 2_000
        assert by_split == {"train": 1_400, "valid": 300, "test": 300}
        assert tuple(by_split) == SPLITS
