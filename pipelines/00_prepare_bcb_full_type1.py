import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUMP_PATH = Path(r"C:\Users\koush\PyProjects\bcb")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "bench_data" / "bcb_full_type1"


def _copy_unescape(value: str) -> str:
    if value == r"\N":
        return ""

    replacements = {
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\\": "\\",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _normalize_pair(left_id: int, right_id: int) -> tuple[int, int]:
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def _iter_copy_table(dump_path: Path, table_name: str, desc: str):
    dump_size = dump_path.stat().st_size
    target_prefix = f"COPY {table_name} "
    in_table = False

    with dump_path.open("rb") as f, tqdm(
        total=dump_size,
        unit="B",
        unit_scale=True,
        desc=desc,
    ) as pbar:
        for raw in f:
            pbar.update(len(raw))
            line = raw.decode("utf-8", errors="replace").rstrip("\n")

            if not in_table:
                if line.startswith(target_prefix):
                    in_table = True
                continue

            if line == r"\.":
                break

            yield line


def collect_type1_pairs(dump_path: Path) -> list[tuple[int, int, int]]:
    pairs: set[tuple[int, int]] = set()

    for line in _iter_copy_table(dump_path, "public.clones", "Pass 1/5: Type-1 clone pairs"):
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        if parts[4] != "1":
            continue

        left_id = int(parts[0])
        right_id = int(parts[1])
        pairs.add(_normalize_pair(left_id, right_id))

    return [(left_id, right_id, 1) for left_id, right_id in sorted(pairs)]


def collect_code_ids(dump_path: Path) -> list[int]:
    ids: list[int] = []

    for line in _iter_copy_table(dump_path, "public.pretty_printed_functions", "Pass 2/5: Function code ids"):
        function_id, _ = line.split("\t", 1)
        ids.append(int(function_id))

    return ids


def sample_candidate_nonclones(
    code_ids: list[int],
    excluded_pairs: set[tuple[int, int]],
    wanted: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    max_attempts = max(wanted * 100, 100_000)
    attempts = 0

    with tqdm(total=wanted, desc="Sampling candidate non-clones", unit="pair") as pbar:
        while len(candidates) < wanted and attempts < max_attempts:
            attempts += 1
            left_id = rng.choice(code_ids)
            right_id = rng.choice(code_ids)
            if left_id == right_id:
                continue

            key = _normalize_pair(left_id, right_id)
            if key in excluded_pairs or key in candidates:
                continue

            candidates.add(key)
            pbar.update(1)

    if len(candidates) < wanted:
        raise RuntimeError(f"Could only sample {len(candidates)} non-clone candidates out of {wanted}.")

    return candidates


def remove_known_clones(
    dump_path: Path,
    candidate_nonclones: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if not candidate_nonclones:
        return candidate_nonclones

    validated = set(candidate_nonclones)

    for line in _iter_copy_table(dump_path, "public.clones", "Pass 3/5: Remove known clones"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        key = _normalize_pair(int(parts[0]), int(parts[1]))
        validated.discard(key)

        if len(validated) == len(candidate_nonclones):
            # Keep scanning: a future row might still match a candidate.
            pass

    return validated


def extract_codes(
    dump_path: Path,
    needed_ids: set[int],
    output_path: Path,
) -> int:
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for line in _iter_copy_table(dump_path, "public.pretty_printed_functions", "Pass 4/5: Extract code"):
            function_id_raw, code_raw = line.split("\t", 1)
            function_id = int(function_id_raw)
            if function_id not in needed_ids:
                continue

            out.write(json.dumps({"idx": str(function_id), "func": _copy_unescape(code_raw)}, ensure_ascii=False))
            out.write("\n")
            written += 1

            if written == len(needed_ids):
                break

    return written


def write_pairs(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for left_id, right_id, label in rows:
            f.write(f"{left_id}\t{right_id}\t{label}\n")


def write_type_labels(path: Path, positives: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for left_id, right_id, _ in positives:
            f.write(f"{left_id}\t{right_id}\ttype_1\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Type-1-only BCB benchmark from the full BCB dump.")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-pairs", type=int, default=100_000)
    parser.add_argument("--max-positive-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dump.exists():
        raise FileNotFoundError(f"BCB dump not found: {args.dump}")
    if args.target_pairs < 1:
        raise ValueError("--target-pairs must be at least 1.")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positives = collect_type1_pairs(args.dump)
    code_ids = collect_code_ids(args.dump)
    code_id_set = set(code_ids)

    positives = [row for row in positives if row[0] in code_id_set and row[1] in code_id_set]
    if args.max_positive_pairs > 0 and len(positives) > args.max_positive_pairs:
        positives = rng.sample(positives, args.max_positive_pairs)
        positives.sort()

    positive_keys = {_normalize_pair(left_id, right_id) for left_id, right_id, _ in positives}
    wanted_negatives = max(0, args.target_pairs - len(positives))

    candidate_count = max(wanted_negatives + 10_000, int(wanted_negatives * 1.25))
    candidate_nonclones = sample_candidate_nonclones(code_ids, positive_keys, candidate_count, rng)
    validated_nonclones = remove_known_clones(args.dump, candidate_nonclones)

    if len(validated_nonclones) < wanted_negatives:
        raise RuntimeError(
            f"Only {len(validated_nonclones)} validated non-clones remained; "
            f"need {wanted_negatives}. Re-run with a larger candidate multiplier."
        )

    negatives = [(left_id, right_id, 0) for left_id, right_id in sorted(validated_nonclones)]
    negatives = rng.sample(negatives, wanted_negatives)

    rows = positives + negatives
    rng.shuffle(rows)

    needed_ids = {left_id for left_id, _, _ in rows} | {right_id for _, right_id, _ in rows}
    written_functions = extract_codes(args.dump, needed_ids, args.output_dir / "data.jsonl")

    write_pairs(args.output_dir / "train.txt", rows)
    write_type_labels(args.output_dir / "type_labels.tsv", positives)

    metadata = {
        "dump_path": str(args.dump.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "target_pairs": args.target_pairs,
        "seed": args.seed,
        "source": "public.pretty_printed_functions + public.clones",
        "available_code_ids": len(code_ids),
        "type_1_clones": len(positives),
        "non_clones": len(negatives),
        "total_pairs": len(rows),
        "needed_function_ids": len(needed_ids),
        "written_functions": written_functions,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[+] Full BCB Type-1 benchmark created.")
    print(f"    Output Dir: {args.output_dir}")
    print(f"    Functions written: {written_functions}")
    print(f"    Type-1 clones: {len(positives)}")
    print(f"    Non-clones: {len(negatives)}")
    print(f"    Total pairs: {len(rows)}")


if __name__ == "__main__":
    main()
