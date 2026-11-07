"""Build disjoint, balanced pairs from existing exported CodeNet program graphs.

Problem identity remains a proxy for functionality, as in the original release.
No behavioral equivalence checking is claimed. Exact duplicate source and shared
submission IDs join problem groups before splitting. Graph extraction is reused.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import fingerprint

LANGUAGES = ("python", "java", "cpp", "csharp")
CONFIGURATIONS = tuple((a, b) for i, a in enumerate(LANGUAGES) for b in LANGUAGES[i:])


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_problem_splits(codes: list[dict], seed: int = 42) -> dict[str, str]:
    """Union duplicate sources/submissions across problems, then hash components."""
    parent = {str(c["problem_id"]): str(c["problem_id"]) for c in codes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for code in codes:
        problem = str(code["problem_id"])
        source = str(code.get("code", ""))
        keys = [("source", _sha(source))]
        if code.get("source_code_id"):
            keys.append(("submission", str(code["source_code_id"])))
        for key in keys:
            if key in seen:
                a, b = find(problem), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = problem
    result = {}
    for problem in parent:
        value = int(_sha(f"revision-problem-v1/{seed}/{find(problem)}")[:16], 16) / 2**64
        result[problem] = "train" if value < .70 else "valid" if value < .85 else "test"
    return result


def pair_overlap(pairs: pd.DataFrame, codes: list[dict]) -> dict:
    index = {str(c["code_id"]): c for c in codes}
    sets = {}
    for split in ("train", "valid", "test"):
        frame = pairs[pairs.split.eq(split)]
        ids = set(frame.left_id.astype(str)) | set(frame.right_id.astype(str))
        sets[split] = {
            "endpoint": ids,
            "problem": {str(index[c]["problem_id"]) for c in ids},
            "source_hash": {_sha(index[c]["code"]) for c in ids},
            "submission": {str(index[c].get("source_code_id", c)) for c in ids},
        }
    return {f"{a}_{b}": {key: len(sets[a][key] & sets[b][key]) for key in sets[a]}
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}


def generate_pairs(codes: list[dict], assignments: dict[str, str], *, per_configuration: int = 10000,
                   seed: int = 42) -> pd.DataFrame:
    if per_configuration <= 0 or per_configuration % 40:
        raise ValueError("Pairs per configuration must be a positive multiple of 40 for exact 70/15/15 balance")
    rng = np.random.default_rng(seed)
    groups = defaultdict(lambda: defaultdict(list))
    index = {str(c["code_id"]): c for c in codes}
    for code in codes:
        split = assignments[str(code["problem_id"])]
        groups[(split, code["language"])][str(code["problem_id"])].append(str(code["code_id"]))
    rows = []
    for split, fraction in (("train", .7), ("valid", .15), ("test", .15)):
        per_label = int(per_configuration * fraction / 2)
        for a, b in CONFIGURATIONS:
            left, right = groups[(split, a)], groups[(split, b)]
            shared = sorted(p for p in left.keys() & right.keys() if a != b or len(left[p]) >= 2)
            if not shared or len(set(left) | set(right)) < 2:
                raise ValueError(f"Insufficient problems for {split}/{a}/{b}; change seed or input pool")
            for label in (1, 0):
                used = set()
                problems_a, problems_b = sorted(left), sorted(right)
                attempts = 0
                while len(used) < per_label:
                    attempts += 1
                    if attempts > max(10000, per_label * 100):
                        raise ValueError(f"Insufficient distinct pairs for {split}/{a}/{b}/{label}; reduce quota")
                    p = str(rng.choice(shared if label else problems_a))
                    q = p if label else str(rng.choice(problems_b))
                    if not label and p == q:
                        continue
                    x, y = str(rng.choice(left[p])), str(rng.choice(right[q]))
                    if x == y or _sha(index[x]["code"]) == _sha(index[y]["code"]):
                        continue
                    key = tuple(sorted((x, y)))
                    if key in used:
                        continue
                    used.add(key)
                    configuration = a if a == b else f"{a}_{b}"
                    rows.append({"split": split, "left_id": x, "right_id": y, "label": label,
                                 "configuration_id": configuration,
                                 "pair_id": _sha(f"{split}/{x}/{y}")[:24],
                                 "pair_kind": "clone" if label else "nonclone_diff_problem"})
    return pd.DataFrame(rows)


def prepare(source_zip: Path, output: Path, *, per_configuration: int = 10000, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory; existing artifacts are preserved: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        code_member = next(n for n in names if n.endswith("/codes.jsonl.gz"))
        graph_member = next(n for n in names if n.endswith("/graph_spectra.jsonl.gz"))
        with archive.open(code_member) as raw, gzip.open(raw, "rt", encoding="utf-8") as stream:
            codes = [json.loads(line) for line in stream if line.strip()]
        codes = [c for c in codes if c.get("language") in LANGUAGES and c.get("problem_id")
                 and str(c.get("code", "")).strip() and not c.get("is_mutant", False)]
        wanted = {str(c["code_id"]) for c in codes}
        usable, coverage = set(), Counter()
        # Reuse the existing gzip byte stream; recompressing the multi-GB graph
        # corpus wastes substantial preparation time. Unused records are safe:
        # the corpus loader selects only IDs referenced by the new pair table.
        cache = output / "graph_spectra.jsonl.gz"
        with archive.open(graph_member) as raw, cache.open("wb") as dest:
            shutil.copyfileobj(raw, dest, length=1024 * 1024)
        with gzip.open(cache, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                code_id = str(row.get("code_id"))
                if code_id not in wanted:
                    continue
                ast = row.get("graphs", {}).get("ast", {})
                adjacency = ast.get("adjacency", ast)
                if not adjacency.get("node_types"):
                    coverage["missing_ast"] += 1
                    continue
                usable.add(code_id)
                coverage["truncated_ast_256"] += len(adjacency["node_types"]) > 256
                for layer in ("ast", "cfg", "ddg", "cpg"):
                    item = row.get("graphs", {}).get(layer, {})
                    coverage[f"{layer}_missing_spectra"] += not bool(item.get("eigenvalues"))
        codes = [c for c in codes if str(c["code_id"]) in usable]
        assignments = assign_problem_splits(codes, seed)
        pairs = generate_pairs(codes, assignments, per_configuration=per_configuration, seed=seed)
        overlap = pair_overlap(pairs, codes)
        if any(count for values in overlap.values() for count in values.values()):
            raise RuntimeError(f"Disjointness audit failed: {overlap}")
        with gzip.open(output / "codes.jsonl.gz", "wt", encoding="utf-8") as stream:
            for code in codes:
                stream.write(json.dumps(code, ensure_ascii=False) + "\n")
        pairs.to_csv(output / "pairs.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"problem_id": p, "split": s} for p, s in sorted(assignments.items())]).to_csv(output / "problem_splits.csv", index=False)
    manifest = {
        "format": "spectral_clean_data_v1", "protocol": "revision_problem_disjoint_v1",
        "dataset": "CodeNet problem/submission/exact-source-disjoint", "seed": seed,
        "input_archive": str(source_zip.resolve()), "source_codes_member_crc": archive.getinfo(code_member).CRC,
        "source_graphs_member_crc": archive.getinfo(graph_member).CRC,
        "codes": len(codes), "pairs": len(pairs), "overlap": overlap,
        "coverage": dict(coverage), "user_disjoint": "unverified: user IDs unavailable in this release",
        "label_definition": "same problem = positive, different problem = negative; no execution validation",
        "pair_table_sha256": fingerprint(pairs.to_dict("records")),
        "problem_assignments_sha256": fingerprint(assignments),
        "graph_support": "common valid AST backbone; other graph-layer failures remain reported",
    }
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-configuration", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_zip, args.output, per_configuration=args.per_configuration, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
