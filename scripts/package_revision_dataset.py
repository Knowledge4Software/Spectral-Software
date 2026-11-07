"""Package the prepared revision bundle for manual Kaggle upload; never publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def package(root: Path, output: Path) -> dict:
    manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") not in {"revision_problem_disjoint_v1", "revision_reference_audited_v1"}:
        raise ValueError("Expected an audited revision release")
    if not manifest.get("overlap") or any(n for group in manifest["overlap"].values() for n in group.values()):
        raise ValueError("Disjointness audit is absent or nonzero")
    required = ["codes.jsonl.gz", "pairs.csv.gz", "graph_spectra.jsonl.gz", "metadata.json"]
    required += [name for name in ("problem_splits.csv", "coverage_by_language.json", "coverage_by_language_split.csv",
                                  "coverage_by_language_split.json") if (root / name).exists()]
    alignment = manifest.get("ddg_ast_alignment")
    if alignment:
        required += [alignment["file"], "ddg_alignment_audit.json"]
        if hashlib.sha256((root / alignment["file"]).read_bytes()).hexdigest() != alignment["sha256"]:
            raise ValueError("Alignment sidecar checksum mismatch")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Graphs are already gzipped; avoid an ineffective second compression pass.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for name in required:
            archive.write(root / name, f"clean_data/{name}")
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = {"file": str(output.resolve()), "bytes": output.stat().st_size,
              "sha256": digest.hexdigest(), "members": required}
    output.with_suffix(".sha256").write_text(f"{result['sha256']}  {output.name}\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.output), indent=2))
