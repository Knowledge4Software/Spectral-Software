"""Publish four bounded numerical-recovery jobs without rewriting frozen jobs."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from scripts.prepare_revision_core_repeats import CORE, ROOT
from spectral_code.revision.numerics import VALIDATION_RECOVERY_SOURCE, can_upgrade
from spectral_code.revision.protocol import fingerprint
from spectral_code.revision.screening import VALIDATION_RECOVERY_HASH
from spectral_code.revision.versioning import TRAINING_SOURCES, training_implementation_hash

JOBS = (("classification_only", 123, True), ("classification_only", 231, False),
        ("mean_max_classification_only", 231, False), ("attention_classification_only", 123, False))


def prepare(root=None):
    base = Path(root) if root is not None else ROOT / "revision"
    source_hash = training_implementation_hash()
    if source_hash != VALIDATION_RECOVERY_HASH or not can_upgrade(VALIDATION_RECOVERY_SOURCE):
        raise ValueError("The exact approved validation-only runtime has changed; review before generating recovery jobs")
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "spectral_code/revision").glob("*.py"))}
    rows = []
    for arm, seed, resume in JOBS:
        group = dict(CORE)[arm]
        original = f"03_selected_repeats/codenet/{group}/codenet_{arm}_seed{seed}.ipynb"
        nb = json.loads((base / original).read_text(encoding="utf-8"))
        bootstrap = "".join(nb["cells"][2]["source"])
        tree = ast.parse(bootstrap)
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "_sources")
        old = ast.literal_eval(node.value)
        if fingerprint({name: old[name] for name in TRAINING_SOURCES}) != VALIDATION_RECOVERY_SOURCE:
            raise ValueError("Recovery source notebook does not contain the audited input-v2 snapshot")
        changed = {name for name in TRAINING_SOURCES if old[name] != sources[name]}
        if changed != {"training.py", "numerics.py"}:
            raise ValueError(f"Unexpected scientific-runtime changes: {changed}")
        bootstrap = bootstrap.replace(ast.get_source_segment(bootstrap, node), "_sources = " + repr(sources), 1)
        nb["cells"][2]["source"] = bootstrap.splitlines(keepends=True)
        config = "".join(nb["cells"][1]["source"])
        if resume:
            config = config.replace("ALLOW_NUMERICAL_UPGRADE = False", "ALLOW_NUMERICAL_UPGRADE = True")
            config += '''
# Attach the SAVED OUTPUT of this exact failed seed-123 job, plus CodeNet data.
# Model/data/seed/loss settings and checkpoint identity are checked by training.
_matches = sorted(Path("/kaggle/input").rglob("revision_classification_only_seed123_input_v2/progress.pt"))
if len(_matches) != 1:
    raise RuntimeError(f"Attach exactly one failed classification-only seed-123 output. Found: {_matches}")
RESUME_CHECKPOINT = str(_matches[0])
'''
        nb["cells"][1]["source"] = config.splitlines(keepends=True)
        nb["cells"][0]["source"] = (
            f"# CodeNet {arm}, seed {seed}: validation numerical recovery\n\n"
            + ("Continue the failed checkpoint. Epoch 43 training is already saved; validation is retried first.\n"
               if resume else "Fresh seed: no old model checkpoint is needed.\n")
            + "\nAttach the existing codenet_revision_common_support dataset; GPU on, Internet off. "
            "An invalid AMP prediction pass is retried once in FP32 for all validation pairs. "
            "Finite AMP predictions and optimizer updates are unchanged. Persistent FP32 failure stops the run. "
            "No model, data, loss or stopping setting is changed. FINAL_TEST=False. "
            "This separately versioned notebook preserves the original frozen notebook and records any migration/recovery.\n"
        ).splitlines(keepends=True)
        name = f"codenet_{arm}_seed{seed}_" + ("resume_validation" if resume else "validation_recovery") + ".ipynb"
        relative = "03_selected_repeats/codenet/validation_recovery/" + name
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        rows.append(dict(job_id=f"codenet_{arm}_seed{seed}", notebook=relative,
                         notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         implementation_hash=source_hash,
                         resume_checkpoint="auto: exact failed seed-123 output" if resume else "None",
                         allow_numerical_upgrade=resume, final_test=False, original_notebook=original))
    manifest = base / "queues/validation_recovery_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Prepared {len(rows)} validation-recovery notebooks; frozen notebooks untouched.")
    return rows


if __name__ == "__main__":
    prepare()
