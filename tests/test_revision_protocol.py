from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spectral_code.revision.prepare import assign_problem_splits, generate_pairs, pair_overlap
from spectral_code.revision.protocol import RunConfig, choose_threshold, matched_training_pairs


def _codes():
    return [{"code_id": f"{p}/{lang}/{i}", "source_code_id": f"submission/{p}/{lang}/{i}",
             "problem_id": str(p), "language": lang, "code": f"program {p} {lang} {i}"}
            for p in range(100) for lang in ("python", "java", "cpp", "csharp") for i in range(3)]


def test_new_pairs_are_balanced_and_problem_submission_hash_disjoint():
    codes = _codes()
    splits = assign_problem_splits(codes)
    pairs = generate_pairs(codes, splits, per_configuration=40)
    assert len(pairs) == 400
    assert pairs.groupby(["split", "configuration_id"]).label.mean().eq(.5).all()
    assert not pairs.pair_id.duplicated().any()
    assert all(n == 0 for values in pair_overlap(pairs, codes).values() for n in values.values())
    pd.testing.assert_frame_equal(pairs, generate_pairs(codes, splits, per_configuration=40))


def test_duplicate_source_or_submission_joins_problem_groups_before_split():
    codes = _codes()
    codes[12]["code"] = codes[0]["code"]
    codes[24]["source_code_id"] = codes[12]["source_code_id"]
    splits = assign_problem_splits(codes)
    assert splits["0"] == splits["1"] == splits["2"]


def test_metric_policy_survives_small_label_imbalance_and_matches_brute_force():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([.8, .2, .1, .7, .6])
    for primary in ("Accuracy", "F1"):
        threshold, result = choose_threshold(y, p, primary)
        from spectral_code.revision.protocol import metrics
        scores = [metrics(y, p, t)[primary] for t in np.r_[p, np.nextafter(p.max(), np.inf)]]
        assert result[primary] == max(scores)
        assert np.isfinite(threshold)


def test_bridge_training_budget_is_exact_and_fails_on_missing_capacity():
    frame = pd.DataFrame([{"configuration_id": bucket, "label": label, "id": f"{bucket}/{label}/{i}"}
                          for bucket in ("java", "java_cpp", "cpp_python") for label in (0, 1) for i in range(20)])
    a = matched_training_pairs(frame, ("java", "java_cpp"), 24, 42)
    b = matched_training_pairs(frame, ("java", "java_cpp", "cpp_python"), 24, 42)
    assert len(a) == len(b) == 24
    assert a.label.mean() == b.label.mean() == .5
    with pytest.raises(ValueError):
        matched_training_pairs(frame, ("java", "java_cpp"), 100, 42)


def test_kaggle_budget_requires_room_below_twelve_hours():
    with pytest.raises(ValueError):
        RunConfig(session_hours=12)
    with pytest.raises(ValueError):
        RunConfig(session_hours=1, reserve_minutes=65)
    with pytest.raises(ValueError):
        RunConfig(batch_size=30, effective_batch_size=128)


def test_runtime_forecast_preserves_margin_and_handles_no_history_or_expired_clock():
    from spectral_code.revision.protocol import runtime_forecast
    assert runtime_forecast(100, 0, 300, 3600)["status"] == "awaiting_first_validated_epoch"
    # Ten validated epochs averaged five minutes each; reserve another 15%.
    forecast = runtime_forecast(100, 10, 3000, 3600)
    assert forecast["seconds_per_epoch"] == 300
    assert forecast["estimated_more_epochs_this_session"] == 10
    assert not forecast["epoch_cap_likely_fits_this_session"]
    assert runtime_forecast(20, 10, 3000, 3600)["epoch_cap_likely_fits_this_session"]
    assert runtime_forecast(20, 10, 3000, -1)["estimated_more_epochs_this_session"] == 0
    assert "estimated_more_epochs_this_session" not in runtime_forecast(20, 10, 3000, None)
