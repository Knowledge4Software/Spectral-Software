"""Small validation-report downloads, independent of training checkpoints."""
from __future__ import annotations

import inspect

SCREENING_INTRO = """# Validation-only screening summary

For this collection, attach only these saved CodeNet training outputs:

- CodeBERT, seed 42
- C4-adapted, seed 42
- Qwen3-LoRA, seed 42

Run on CPU; no GPU, Internet, or training dataset is needed.
Collect validation status and export `revision_small_reports.zip`
containing only small JSON reports and summary CSVs. Download that ZIP instead of
multi-gigabyte checkpoints. Incomplete runs are listed but not ranked. Checkpoints,
token arrays and final tests are never opened. Keep the full saved outputs on
Kaggle for continuation and final evaluation.
"""


def export_small_reports(input_root, output_zip, summary_directory):
    """Package known report files; never open weights, arrays or final tests."""
    import hashlib
    import json
    import re
    import zipfile
    from pathlib import Path

    input_root, output_zip = Path(input_root), Path(output_zip)
    summary_directory = Path(summary_directory)
    required = {"identity", "training", "implementation_hash", "arm"}
    max_report_bytes = 8 * 1024 * 1024
    selected = []
    runs = []
    for path in sorted(input_root.rglob("status.json")):
        if path.stat().st_size > max_report_bytes:
            raise ValueError(f"Unexpectedly large status report: {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not required <= state.keys():
            continue
        relative = path.parent.relative_to(input_root).as_posix()
        source_id = hashlib.sha256(relative.encode()).hexdigest()[:16]
        label = re.sub(r"[^A-Za-z0-9_-]", "_", (
            f"{state['training']['dataset']}_{state['arm']}_seed{state['training']['seed']}"))
        prefix = f"runs/{label}/{source_id}"
        runs.append({"source": relative, "identity": state["identity"], "archive_directory": prefix,
                     "status": state.get("status"), "complete": state.get("complete", False)})
        for filename in ("status.json", "benchmark.json", "baseline_support.json",
                         "numerical_failure.json", "transfer_metadata.json"):
            report = path.parent / filename
            if report.is_file():
                if report.stat().st_size > max_report_bytes:
                    raise ValueError(f"Unexpectedly large report: {report}")
                selected.append((report, f"{prefix}/{filename}"))
    if not runs:
        raise ValueError("No revision status.json found. Attach saved training notebook outputs; no dataset or checkpoint download is required.")
    for filename in ("validation_screening.csv", "run_ledger.csv", "README.md"):
        report = summary_directory / filename
        if report.is_file():
            selected.append((report, "summary/" + filename))
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"reports_only": True, "runs": runs, "files": [],
                "checkpoint_integrity_verified": False,
                "note": "For validation screening only. Checkpoints and token arrays are preserved in the original Kaggle outputs and are not read or copied here."}
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, member in selected:
            raw = path.read_bytes()
            archive.writestr(member, raw)
            manifest["files"].append({"member": member, "bytes": len(raw),
                                      "sha256": hashlib.sha256(raw).hexdigest()})
        archive.writestr("report_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return {"zip": str(output_zip), "bytes": output_zip.stat().st_size,
            "runs": len(runs), "report_files": len(selected), "contains_checkpoints": False}


def screening_run_source():
    """Embed the exporter so the uploaded notebook needs no repository checkout."""
    return inspect.getsource(export_small_reports) + '''
from spectral_code.revision.screening import collect_validation
summary_dir = Path("/kaggle/working/validation_screening")
summary = collect_validation(INPUT_ROOT, summary_dir, QUEUE_CSV)
print(summary.to_string(index=False))
download = export_small_reports(INPUT_ROOT, Path("/kaggle/working/revision_small_reports.zip"), summary_dir)
import json
print(json.dumps(download, indent=2))
try:
    from IPython.display import FileLink, display
    display(FileLink(download["zip"]))
except ImportError:
    pass
'''
