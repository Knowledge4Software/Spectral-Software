"""Clone-type reporting from supplied authoritative positive-pair annotations."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .protocol import metrics


def report_clone_types(predictions, annotations, threshold, output):
    frame = pd.read_csv(predictions, dtype={"pair_id": str})
    types = pd.read_csv(annotations, dtype={"pair_id": str, "clone_type": str})
    if not {"pair_id", "clone_type", "annotation_source"}.issubset(types):
        raise ValueError("Annotations need pair_id, clone_type and annotation_source; do not infer Type-IV from token overlap")
    if types.pair_id.duplicated().any() or types[["clone_type", "annotation_source"]].isna().any().any():
        raise ValueError("Each pair needs one nonmissing authoritative annotation")
    merged = frame.merge(types, on="pair_id", how="left", validate="one_to_one")
    if merged.loc[merged.label.eq(0), "clone_type"].notna().any():
        raise ValueError("Clone-type labels apply to positive pairs only")
    negatives = merged[merged.label.eq(0)]
    rows = []
    for name, positives in merged[merged.label.eq(1) & merged.clone_type.notna()].groupby("clone_type"):
        subset = pd.concat((positives, negatives))
        rows.append({"clone_type": name, **metrics(subset.label, subset.probability, threshold),
                     "positive_pairs": len(positives), "shared_negative_pairs": len(negatives),
                     "annotation_sources": ";".join(sorted(positives.annotation_source.unique()))})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "clone_type_metrics.csv", index=False)
    status = {"annotated_positives": int((merged.label.eq(1) & merged.clone_type.notna()).sum()),
              "unannotated_positives": int((merged.label.eq(1) & merged.clone_type.isna()).sum()),
              "threshold_source": "fixed validation-selected threshold supplied by completed experiment",
              "interpretation": "Each type uses all shared negatives; report recall by type. F1/AP depend on this constructed prevalence; do not average types as independent tests."}
    (output / "clone_type_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
