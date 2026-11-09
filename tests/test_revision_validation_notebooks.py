import ast
import hashlib
import json
import shutil

from scripts.prepare_revision_core_repeats import CORE, ROOT
from scripts.prepare_revision_validation_recovery import JOBS, prepare
from spectral_code.revision.numerics import VALIDATION_RECOVERY_SOURCE, can_upgrade
from spectral_code.revision.protocol import fingerprint
from spectral_code.revision.screening import (INPUT_V1_HASH, INPUT_V2_HASH, VALIDATION_RECOVERY_HASH,
                                             comparison_implementation_hash)
from spectral_code.revision.versioning import TRAINING_SOURCES


def test_recovery_notebooks_are_versioned_compile_and_preserve_frozen_files(tmp_path):
    (tmp_path / "queues").mkdir()
    original_bytes = {}
    for arm, seed, _ in JOBS:
        relative = f"03_selected_repeats/codenet/{dict(CORE)[arm]}/codenet_{arm}_seed{seed}.ipynb"
        source = ROOT / "revision" / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        original_bytes[relative] = target.read_bytes()
    rows = prepare(tmp_path)
    assert len(rows) == 4 and sum(r["allow_numerical_upgrade"] for r in rows) == 1
    for r in rows:
        path = tmp_path / r["notebook"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["notebook_sha256"]
        nb = json.loads(path.read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), path.name, "exec")
        config = "".join(nb["cells"][1]["source"])
        assert "FINAL_TEST = False" in config and "MAX_EPOCHS = 100" in config
        assert ("_matches = sorted" in config) == r["allow_numerical_upgrade"]
        assert ("ALLOW_NUMERICAL_UPGRADE = True" in config) == r["allow_numerical_upgrade"]
        tree = ast.parse("".join(nb["cells"][2]["source"]))
        sources = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_sources"))
        assert fingerprint({n: sources[n] for n in TRAINING_SOURCES}) == VALIDATION_RECOVERY_HASH
    assert all((tmp_path / path).read_bytes() == content for path, content in original_bytes.items())
    first = (tmp_path / "queues/validation_recovery_manifest.csv").read_bytes()
    prepare(tmp_path)
    assert (tmp_path / "queues/validation_recovery_manifest.csv").read_bytes() == first
    resume_row = next(r for r in rows if r["allow_numerical_upgrade"])
    resume_bytes = (tmp_path / resume_row["notebook"]).read_bytes()
    restarted = prepare(tmp_path, restart_seed123=True)
    assert len(restarted) == 4 and not any(r["allow_numerical_upgrade"] for r in restarted)
    fresh = next(r for r in restarted if r["job_id"] == "codenet_classification_only_seed123")
    assert fresh["resume_checkpoint"] == "None" and fresh["notebook"] != resume_row["notebook"]
    nb = json.loads((tmp_path / fresh["notebook"]).read_text(encoding="utf-8"))
    config = "".join(nb["cells"][1]["source"])
    assert "RESUME_CHECKPOINT = None" in config and "ALLOW_NUMERICAL_UPGRADE = False" in config
    assert "_matches" not in config
    assert "_input_v2_restart_validation_fix" in "".join(nb["cells"][3]["source"])
    for c in nb["cells"]:
        if c["cell_type"] == "code":compile("".join(c["source"]), fresh["notebook"], "exec")
    assert (tmp_path / resume_row["notebook"]).read_bytes() == resume_bytes


def test_only_exact_approved_recovery_can_migrate_or_share_comparison_hash(monkeypatch):
    from spectral_code.revision import numerics
    assert can_upgrade(VALIDATION_RECOVERY_SOURCE)
    assert not can_upgrade(INPUT_V1_HASH) and not can_upgrade("arbitrary")
    assert comparison_implementation_hash(dict(arm="full", implementation_hash=VALIDATION_RECOVERY_HASH)) == INPUT_V2_HASH
    assert comparison_implementation_hash(dict(arm="full", implementation_hash=INPUT_V1_HASH)) == INPUT_V1_HASH
    assert comparison_implementation_hash(dict(arm="full", implementation_hash="arbitrary")) == "arbitrary"
    monkeypatch.setattr(numerics, "upgrade_target_hash", lambda: "changed runtime")
    assert not can_upgrade(VALIDATION_RECOVERY_SOURCE)
