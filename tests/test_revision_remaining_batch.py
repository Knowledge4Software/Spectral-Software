import ast
import csv
import hashlib
import json
import shutil

from scripts.prepare_revision_remaining_batch import CONTROLS, ROOT, prepare
from spectral_code.revision.protocol import fingerprint
from spectral_code.revision.screening import VALIDATION_RECOVERY_HASH
from spectral_code.revision.versioning import TRAINING_SOURCES


def test_independent_batch_preserves_experiments_and_never_requeues_reported_jobs(tmp_path):
    source = ROOT / "revision"
    for name in ("results/run_progress.csv", "results/control_progress.csv", "queues/notebook_manifest.csv"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, tmp_path / name)
    # Keep the fixture independent of results that arrive after this test was
    # written: all 15 declared controls begin unreported in the test scenario.
    tracker = tmp_path / "results/control_progress.csv"
    with tracker.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f); tracker_fields = reader.fieldnames; tracker_rows = list(reader)
    planned_controls = {f"codenet_{arm}_seed42" for arm, _, _ in CONTROLS}
    for row in tracker_rows:
        if row["job_id"] in planned_controls:row["status"] = "not_reported"
    with tracker.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tracker_fields); writer.writeheader(); writer.writerows(tracker_rows)
    originals = [f"02_controls_seed42/codenet/{g}/codenet_{a}_seed42.ipynb" for a, g, _ in CONTROLS]
    originals += [f"03_selected_repeats/{d}/baselines/{d}_full_seed{s}.ipynb"
                  for d in ("atcoder", "bigclonebench") for s in (123, 231)]
    before = {}
    for relative in originals:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        before[relative] = target.read_bytes()
    jobs = prepare(tmp_path)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 19
    assert sum(j["dataset"] == "codenet" for j in jobs) == 15
    assert not any(j["job_id"] == "codenet_classification_only_seed123" for j in jobs)
    for job in jobs:
        path = tmp_path / job["notebook"]
        assert b"\r\n" not in path.read_bytes()  # Raw hashes must survive an LF Git checkout on Windows.
        assert hashlib.sha256(path.read_bytes()).hexdigest() == job["notebook_sha256"]
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":compile("".join(cell["source"]), path.name, "exec")
        config = "".join(nb["cells"][1]["source"])
        assert f"ARM = '{job['arm']}'" in config and f"SEED = {job['seed']}" in config
        for setting in ("RESUME_CHECKPOINT = None", "ALLOW_NUMERICAL_UPGRADE = False", "FINAL_TEST = False",
                        "MAX_EPOCHS = 100", "MIN_EPOCHS = 10", "PATIENCE = 10", "SESSION_HOURS = 10.0"):
            assert setting in config
        old = json.loads((tmp_path / job["original_notebook"]).read_text(encoding="utf-8"))
        assert nb["cells"][1] == old["cells"][1]
        run = "".join(nb["cells"][3]["source"])
        assert "_input_v2_validation_fix" in run and "experiment_label=ARM" in run
        assert ('selection_metric="F1"' in run) == (job["dataset"] == "bigclonebench")
        if job["dataset"] != "codenet":
            assert f"dataset='{job['dataset']}'" in run and f"find_bundle(CLEAN_DATA_DIR, '{job['dataset']}')" in run
        tree = ast.parse("".join(nb["cells"][2]["source"]))
        packed = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_sources"))
        assert fingerprint({n: packed[n] for n in TRAINING_SOURCES}) == VALIDATION_RECOVERY_HASH
        assert job["dataset_zip"].startswith(job["dataset"] + "_revision_common_support")
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    # Reported completions and a live control must disappear from fresh launches.
    path = tmp_path / "results/control_progress.csv"
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    for row in rows:
        if row["job_id"] == "codenet_projected_seed42":row["status"] = "running"
        if row["job_id"] == "codenet_projected_ast_seed42":row["status"] = "complete_verified"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    prepare(tmp_path)
    with (tmp_path / "queues/next_independent_run_queue.csv").open(encoding="utf-8", newline="") as f:
        pending = list(csv.DictReader(f))
    assert len(pending) == 17
    assert not {"codenet_projected_seed42", "codenet_projected_ast_seed42"} & {j["job_id"] for j in pending}
