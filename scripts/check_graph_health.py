"""Fail fast when an exported clean dataset's graphs do not match their source.

Three independent extraction defects reached published results before this check
existed, and all three are invisible in per-record spot checks because each
individual graph looks plausible:

* ATCoder Java kept one arbitrary method per submission, so ~20% of records were
  reduced to an empty ``METHOD/BLOCK`` skeleton.
* SemanticCloneBench and GPTCloneBench C attached graphs to the wrong record
  entirely, because the C frontend emits ~5x more DOT files than source files
  and the index fell back to matching them by export order.

Both collapse the rank correlation between a record's source size and its graph
size, which is what this script measures per dataset and per language.

Usage::

    python scripts/check_graph_health.py
    python scripts/check_graph_health.py --dataset atcoder_v3 --min-correlation 0.6
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.dataset_limitations import limitation_for
OUTPUTS_ROOT = PROJECT_ROOT.parent / "outputs"
DEFAULT_DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")

# A METHOD node plus its signature scaffolding and an empty BLOCK: a body-less
# export, never a real program.
SKELETON_TYPES = frozenset({
    "METHOD", "PARAM", "BLOCK", "MODIFIER", "METHOD_RETURN", "TYPE_DECL", "MEMBER",
    "NAMESPACE_BLOCK", "FILE", "TYPE_REF", "ANNOTATION",
    "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT",
})


def _open_text(path: Path):
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if compressed else path.open("r", encoding="utf-8")


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation without a SciPy dependency; average ranks break ties."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        position = 0
        while position < n:
            end = position
            while end + 1 < n and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mean_x) ** 2 for a in rx) ** 0.5 * sum((b - mean_y) ** 2 for b in ry) ** 0.5
    )
    return numerator / denominator if denominator else float("nan")


def check_dataset(clean_dir: Path, min_correlation: float, max_skeleton: float, strict: bool = False) -> list[str]:
    codes_path = clean_dir / "codes.jsonl.gz"
    graphs_path = clean_dir / "graph_spectra.jsonl.gz"
    if not codes_path.is_file() or not graphs_path.is_file():
        return [f"{clean_dir}: missing codes.jsonl.gz or graph_spectra.jsonl.gz"]

    languages: dict[str, str] = {}
    sizes: dict[str, int] = {}
    with _open_text(codes_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            code_id = str(record.get("code_id"))
            languages[code_id] = str(record.get("language", "unknown"))
            sizes[code_id] = len(record.get("code", ""))

    per_layer: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"chars": [], "nodes": [], "skeleton": 0, "empty": 0, "total": 0, "structures": Counter()}
    )
    layers: list[str] = []
    with _open_text(graphs_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row.get("code_id"))
            language = languages.get(code_id, "unknown")
            graphs = row.get("graphs") or {}
            if not layers:
                layers = [name for name in ("ast", "cfg", "ddg", "cpg", "pdg") if name in graphs]
            for layer in layers:
                adjacency = (graphs.get(layer) or {}).get("adjacency") or {}
                node_types = adjacency.get("node_types") or []
                bucket = per_layer[(language, layer)]
                bucket["total"] += 1
                if not node_types:
                    bucket["empty"] += 1
                    continue
                if set(node_types) <= SKELETON_TYPES:
                    bucket["skeleton"] += 1
                bucket["chars"].append(float(sizes.get(code_id, 0)))
                bucket["nodes"].append(float(len(node_types)))
                bucket["structures"][(tuple(node_types), tuple(adjacency.get("row") or ()))] += 1

    failures = []
    known = []
    dataset = clean_dir.parent.name
    for language, layer in sorted(per_layer):
        bucket = per_layer[(language, layer)]
        total = bucket["total"]
        covered = total - bucket["empty"]
        correlation = _spearman(bucket["chars"], bucket["nodes"])
        skeleton_rate = bucket["skeleton"] / max(covered, 1)
        coverage = covered / max(total, 1)
        unique_rate = len(bucket["structures"]) / max(covered, 1)

        problems = []
        if covered == 0:
            problems.append("layer is empty for every record")
        elif correlation != correlation or correlation < min_correlation:
            problems.append(
                f"source-size vs graph-size correlation {correlation:.3f} < {min_correlation}"
                " - graphs do not match their records"
            )
        if skeleton_rate > max_skeleton:
            problems.append(
                f"{skeleton_rate:.1%} of graphs are body-less skeletons (limit {max_skeleton:.1%})"
            )

        # A gap we have already diagnosed and routed around is not a reason to
        # discard an otherwise-good build; only an undeclared one is.
        limitation = limitation_for(dataset, language, layer)
        if problems and limitation and not strict:
            status = "KNOWN"
            known.append(f"{dataset}/{language}/{layer}: {problems[0]}")
        elif problems:
            status = "FAIL"
            failures.extend(f"{dataset}/{language}/{layer}: {problem}" for problem in problems)
        else:
            status = "OK"

        print(
            f"  {language:8s} {layer:4s} n={total:6,} cover={coverage:6.1%} corr={correlation:6.3f} "
            f"skeleton={skeleton_rate:6.1%} unique={unique_rate:6.1%}  {status}"
        )

    if known:
        print(f"  ({len(known)} declared limitation(s); see spectral_code/evaluation/dataset_limitations.py)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--min-correlation", type=float, default=0.5)
    parser.add_argument("--max-skeleton", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true",
                        help="Also fail on declared limitations (use when auditing the toolchain itself).")
    args = parser.parse_args()

    all_failures = []
    for dataset in args.datasets:
        clean_dir = args.outputs_root / dataset / "clean_data"
        print(f"\n=== {dataset}")
        if not clean_dir.is_dir():
            print("  (not built)")
            continue
        all_failures.extend(check_dataset(clean_dir, args.min_correlation, args.max_skeleton, args.strict))

    print()
    if all_failures:
        print(f"[-] {len(all_failures)} graph-health failure(s):")
        for failure in all_failures:
            print(f"    {failure}")
        return 1
    print("[+] All checked graph layers track their source records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
