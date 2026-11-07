"""Freeze audited common graph support without relabeling existing pairs.

AtCoder problem IDs come from the local source metadata, never inferred from
clone-pair connected components. BigCloneBench problem and clone-type labels
are not fabricated when the source export does not supply them.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _find_file, _open_text
from .prepare import pair_overlap
from .protocol import fingerprint


def _component_assignments(codes, provided=None):
    parent = {str(row["code_id"]): str(row["code_id"]) for row in codes}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for row in codes:
        keys = [("source", hashlib.sha256(row["code"].encode()).hexdigest())]
        if row.get("source_code_id"):
            keys.append(("submission", str(row["source_code_id"])))
        if row.get("problem_id") is not None:
            keys.append(("problem", str(row["problem_id"])))
        for key in keys:
            code_id = str(row["code_id"])
            if key in seen:
                a, b = find(code_id), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = code_id
    owners = {}
    if provided is not None:
        for row in codes:
            owners.setdefault(find(str(row["code_id"])), set()).add(provided.get(str(row.get("problem_id"))))
    assignments = {}
    for key in parent:
        component = find(key)
        if provided is not None:
            splits = owners[component]
            assignments[key] = next(iter(splits)) if len(splits) == 1 and None not in splits else "excluded_conflicting_component"
        else:
            value = int(hashlib.sha256(f"endpoint-reference-v1/42/{component}".encode()).hexdigest()[:16], 16) / 2**64
            assignments[key] = "train" if value < .7 else "valid" if value < .85 else "test"
    return assignments


def _cap_preserving_prevalence(frame, cap, reference_prevalence, seed=42):
    available = frame.label.value_counts().to_dict()
    fraction = reference_prevalence
    total = min(len(frame), cap) if cap else len(frame)
    total = min(total, int(available.get(1, 0) / fraction), int(available.get(0, 0) / (1 - fraction)))
    positives = min(available.get(1, 0), round(total * fraction))
    negatives = min(available.get(0, 0), total - positives)
    if not positives or not negatives:
        raise ValueError("Filtering left too few pairs in one class")
    return pd.concat((frame[frame.label.eq(1)].sample(positives, random_state=seed),
                      frame[frame.label.eq(0)].sample(negatives, random_state=seed))).sample(frac=1, random_state=seed)


def prepare_bundle(source, output, dataset, *, source_metadata_zip=None,
                   required_layers=("ast", "cfg", "ddg", "cpg"),
                   caps=(70000, 15000, 15000)):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty destination; existing datasets are preserved")
    output.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        for name in ("codes.jsonl.gz", "pairs.csv.gz", "graph_spectra.jsonl.gz", "metadata.json"):
            if archive:
                matches = [n for n in archive.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {name}")
                with archive.open(matches[0]) as stream, (output / name).open("wb") as target:
                    shutil.copyfileobj(stream, target)
            else:
                shutil.copyfile(_find_file(source, name), output / name)
        original_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if not archive and original_metadata.get("ddg_ast_alignment"):
            shutil.copyfile(_find_file(source, "ddg_ast_alignment.jsonl.gz"), output / "ddg_ast_alignment.jsonl.gz")
            for name in ("ddg_alignment_audit.json", "coverage_by_language.json", "problem_splits.csv"):
                if (source / name).exists():
                    shutil.copyfile(source / name, output / name)
        with _open_text(output / "codes.jsonl.gz") as stream:
            codes = [json.loads(line) for line in stream]
        for row in codes:
            row["code_id"] = str(row["code_id"])
        provided = None
        if dataset == "atcoder":
            if source_metadata_zip is None:
                raise ValueError("AtCoder requires the original problem metadata archive")
            with zipfile.ZipFile(source_metadata_zip) as metadata_archive:
                metadata = pd.read_parquet(io.BytesIO(metadata_archive.read("ATCoder/function_metadata.parquet")),
                                           columns=["function_id", "problem_id"])
                splits = pd.read_parquet(io.BytesIO(metadata_archive.read("ATCoder/splits_v3_problem_disjoint/problem_split_assignments.parquet")),
                                         columns=["problem_id", "split"])
            lookup = dict(zip(metadata.function_id.astype(str), metadata.problem_id.astype(str)))
            provided = dict(zip(splits.problem_id.astype(str), splits.split.replace({"validation": "valid"})))
            for row in codes:
                row["problem_id"] = lookup.get(str(row["source_code_id"]))
            if any(row["problem_id"] is None for row in codes):
                raise ValueError("Some AtCoder programs have no authoritative problem ID")
        pairs = pd.read_csv(output / "pairs.csv.gz", dtype={"left_id": str, "right_id": str})
        pairs["split"] = pairs.split.replace({"validation": "valid"})
        original_counts = pairs.groupby(["split", "label"]).size().to_dict()
        usable, coverage = set(), Counter()
        wanted = {row["code_id"] for row in codes}
        with _open_text(output / "graph_spectra.jsonl.gz") as stream:
            for line in stream:
                row = json.loads(line)
                if str(row.get("code_id")) not in wanted:
                    continue
                layers = row.get("graphs", {})
                good = True
                for layer in required_layers:
                    view = layers.get(layer, {})
                    nodes = view.get("adjacency", view).get("node_types", [])
                    coverage[f"{layer}_missing_or_empty_nodes"] += not bool(nodes)
                    good &= bool(nodes)
                if good:
                    usable.add(str(row["code_id"]))
        original_pairs = pairs.copy()
        if dataset == "codenet":
            # The existing CodeNet split was already problem/source disjoint.
            if original_metadata.get("protocol") != "revision_problem_disjoint_v1":
                raise ValueError("Start CodeNet common-support preparation from its audited revision directory")
            membership = None
        else:
            membership = _component_assignments(codes, provided)
            pairs = pairs[pairs.left_id.map(membership).eq(pairs.split) & pairs.right_id.map(membership).eq(pairs.split)]
        pairs = pairs[pairs.left_id.isin(usable) & pairs.right_id.isin(usable)].copy()
        pairs["_unordered_pair"] = ["/".join(sorted((str(a), str(b)))) for a, b in
                                    pairs[["left_id", "right_id"]].itertuples(index=False, name=None)]
        if pairs.groupby("_unordered_pair").label.nunique().gt(1).any():
            raise ValueError("Conflicting labels for the same unordered code pair")
        before_deduplication = len(pairs)
        pairs = pairs.drop_duplicates(["split", "_unordered_pair"]).drop(columns="_unordered_pair")
        duplicate_pairs_removed = before_deduplication - len(pairs)
        filtered_counts = pairs.groupby(["split", "label"]).size().to_dict()
        selected = []
        for split, cap in zip(("train", "valid", "test"), caps):
            reference = original_pairs[original_pairs.split.eq(split)]
            subset = pairs[pairs.split.eq(split)]
            selected.append(_cap_preserving_prevalence(subset, cap, float(reference.label.mean())))
        pairs = pd.concat(selected, ignore_index=True)
        pairs["configuration_id"] = pairs.get("configuration_id", pd.Series(index=pairs.index, dtype=str))
        index = {row["code_id"]: row for row in codes}
        missing = pairs.configuration_id.isna()
        pairs.loc[missing, "configuration_id"] = [
            a if a == b else "_".join(sorted((a, b))) for a, b in
            ((index[left]["language"], index[right]["language"]) for left, right in
             pairs.loc[missing, ["left_id", "right_id"]].itertuples(index=False, name=None))]
        if "pair_id" not in pairs:
            pairs["pair_id"] = [hashlib.sha256(f"{split}/{a}/{b}".encode()).hexdigest()[:24]
                                for split, a, b in pairs[["split", "left_id", "right_id"]].itertuples(index=False, name=None)]
        if pairs.pair_id.duplicated().any():
            raise ValueError("Duplicate pair IDs in retained support")
        problems_known = all(row.get("problem_id") is not None for row in codes)
        audit_codes = codes if problems_known else [{**row, "problem_id": row["code_id"]} for row in codes]
        overlap = pair_overlap(pairs, audit_codes)
        if not problems_known:
            for item in overlap.values():
                item.pop("problem")  # endpoint identity is not a functionality label
        if any(value for item in overlap.values() for value in item.values()):
            raise ValueError(f"Disjointness audit failed: {overlap}")
        pairs.to_csv(output / "pairs.csv.gz", index=False, compression="gzip")
        with gzip.open(output / "codes.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in codes:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {"format": "spectral_clean_data_v1", "dataset_key": dataset,
                    "dataset": f"{dataset}: audited common support; derived benchmark",
                    "protocol": "revision_problem_disjoint_v1" if dataset == "codenet" else "revision_reference_audited_v1",
                    "source_metadata": original_metadata, "required_graph_layers": required_layers,
                    "overlap": overlap, "problem_disjoint": problems_known,
                    "problem_id_status": "authoritative source metadata" if problems_known else "unavailable; not inferred",
                    "user_disjoint": "unverified: user metadata unavailable",
                    "clone_type_status": "unavailable unless authoritative annotations are supplied separately",
                    "selection_seed": 42, "caps": caps,
                    "pair_table_sha256": fingerprint(pairs.to_dict("records")),
                    "counts": pairs.groupby(["split", "label"]).size().rename("pairs").reset_index().to_dict("records"),
                    "graph_coverage": dict(coverage), "usable_graph_endpoints": len(usable),
                    "duplicate_unordered_pairs_removed": duplicate_pairs_removed,
                    "original_counts": {str(k): int(v) for k, v in original_counts.items()},
                    "after_support_and_disjointness": {str(k): int(v) for k, v in filtered_counts.items()},
                    "split_policy": "retain original pair labels and partition; remove endpoints/components incompatible with disjoint support",
                    "prevalence_policy": "subsample labels to preserve each original split's prevalence; no label rebalance to 50/50 for BigCloneBench",
                    "claim_boundary": "derived common-support endpoint/source-disjoint evaluation, not a replacement score on the unfiltered official release"}
        if original_metadata.get("ddg_ast_alignment"):
            manifest["ddg_ast_alignment"] = original_metadata["ddg_ast_alignment"]
        (output / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        if archive:
            archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("codenet", "atcoder", "bigclonebench"), required=True)
    parser.add_argument("--source-metadata-zip", type=Path)
    parser.add_argument("--required-layers", nargs="+", default=["ast", "cfg", "ddg", "cpg"])
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.source, args.output, args.dataset,
                                    source_metadata_zip=args.source_metadata_zip, required_layers=tuple(args.required_layers)), indent=2))
