"""Run the paper's frozen experiment notebooks on a Linux CUDA host."""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from queue import Empty, Queue
import re
import shutil
import signal
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "revision"
SEEDS = (42, 123, 231)
BUNDLES = {
    "codenet": ("codenet_revision_common_support", "ec86d556b37b32180c44be50b3d79336fabae97432fe004a0baaebc7b9e4b46f"),
    "atcoder": ("atcoder_revision_common_support_v2", "3525d3a750d395fd912da6879f96986dc4d9fa639fd78293c4c542c7d62d7318"),
    "bigclonebench": ("bigclonebench_revision_common_support_v2", "b6bc759554b73379f61d1c2e3f0de63addfd28d68f2dae91a9d494c342ce9bd5"),
}
FOCUSED = ("source_only", "direct", "two_hop", "three_hop", "target_reference")
EXTERNAL = ("astnn", "rtvnn", "cdlh", "deepsim", "fa_ast_ggnn", "fa_ast_gmn",
            "codebert", "c4_adapted", "qwen3_lora")
PENDING = {
    "codenet_codebert_seed42", "codenet_qwen3_lora_seed42",
    "atcoder_codebert_seed42", "atcoder_qwen3_lora_seed42",
    "bigclonebench_codebert_seed42", "bigclonebench_cdlh_seed42", "bigclonebench_qwen3_lora_seed42",
}
OVERRIDES = {
    "RESUME_CHECKPOINT", "RESUME_TOKENS_DIR", "MODEL_REVISION", "INPUT_ROOT",
    "QUEUE_CSV", "EMBEDDINGS_DIR", "RESUME_EMBEDDINGS_DIR", "RESUME_PROBE_DIR",
    "CLEAN_DATA_DIR",
    "NOTEBOOK_VARIANT",
}


def code(cell):
    return "".join(cell["source"])


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_for(path):
    name = path.name
    if name.startswith(("atcoder", "01_atcoder", "04_atcoder", "collect_atcoder")):
        return "atcoder"
    if name.startswith("bigclonebench"):
        return "bigclonebench"
    if path.parts[0] == "05_analysis" and "f1_seed" not in name:
        return None
    return "codenet"


def row(path, group=None, settings=None, dataset=None, job_id=None):
    notebook = json.loads((REVISION / path).read_bytes())
    settings = dict(settings or {})
    if path.name.startswith(("codenet_", "atcoder_", "bigclonebench_")):
        if "_qwen3_lora_" in path.name:
            settings.setdefault("MODEL_REVISION", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
        elif "_codebert_" in path.name or "_c4_adapted_" in path.name:
            settings.setdefault("MODEL_REVISION", "3b0952feddeffad0063f274080e3c23d75e7eb39")
    return {
        "id": job_id or path.stem, "notebook": path.as_posix(), "sha256": sha256(REVISION / path),
        "group": group or path.parts[0], "dataset": dataset or dataset_for(path),
        "gpu": notebook["metadata"].get("kaggle", {}).get("accelerator", "gpu") != "none",
        "settings": settings,
    }


def make_catalog(external="c4_adapted", sibling="mean_max"):
    jobs = {}
    for notebook in sorted(REVISION.rglob("*.ipynb")):
        path = notebook.relative_to(REVISION)
        if path.parts[0] == "archive":
            job_id = path.stem + "__" + path.parts[1]
        else:
            job_id = path.stem
        if job_id in jobs:
            job_id = "__".join((*path.parts[:-1], job_id))
        jobs[job_id] = row(path, job_id=job_id)

    for seed in SEEDS:
        path = Path(f"07_atcoder_readout_check_v1/atcoder_mean_max_classification_only_seed{seed}_readout_check_v1.ipynb")
        for arm, tag in (("mean_max", "full_objective_v1"), ("auxiliary_half", "f1_v1"),
                         ("without_spectral", "f1_v1")):
            job_id = f"atcoder_{arm}_seed{seed}_{tag}"
            group = "loss_experiment" if arm == "without_spectral" else "atcoder_followup"
            jobs[job_id] = row(path, group, {"ARM": arm}, job_id=job_id)

    # Complete the exploratory architecture/loss matrix on the other two datasets
    # using their own full-model templates, pair supports and selection metrics.
    for dataset in ("atcoder", "bigclonebench"):
        for seed in SEEDS:
            full_id = f"{dataset}_full_seed{seed}"
            full = jobs[full_id]
            for candidate in tuple(jobs.values()):
                match = re.fullmatch(rf"codenet_(.+)_seed{seed}", candidate["id"])
                if not match or candidate["group"] not in {"02_controls_seed42", "03_selected_repeats"}:
                    continue
                arm = match[1]
                job_id = f"{dataset}_{arm}_seed{seed}"
                if job_id not in jobs and arm not in EXTERNAL:
                    jobs[job_id] = {**full, "id": job_id, "group": "expanded_controls",
                                    "settings": {"ARM": arm}}

    # Historical table8 files already implement the fixed 30-epoch protocol.
    # The transfer selector dispatches on this configuration key.
    for condition in (*FOCUSED, "two_hop_reinforced"):
        key = "focused|" + condition
        for seed in SEEDS:
            for model in ("full", "mean_max", "attention", *EXTERNAL):
                if model in ("full", "mean_max", "attention"):
                    path = Path(f"04_transfer/spectra/table8_J_P_seed{seed}.ipynb")
                    settings = {"TRANSFER_KEY": key, "ARM": model}
                else:
                    path = Path(f"04_transfer/baselines/codenet_{model}_table8_J_P_seed{seed}.ipynb")
                    settings = {"TRANSFER_KEY": key}
                job_id = f"focused_{condition}_{model}_seed{seed}"
                group = ("focused" if condition in FOCUSED and model in {"full", sibling, external}
                         else "focused_candidates")
                jobs[job_id] = row(path, group, settings, "codenet", job_id)

    frozen = next(Path("optional") / p.relative_to(REVISION / "optional")
                  for p in (REVISION / "optional").rglob("*.ipynb"))
    for seed in SEEDS:
        job_id = f"qwen3_frozen_probe_seed{seed}"
        jobs[job_id] = row(frozen, "frozen_probe", {"PHASE": "probe", "SEED": seed},
                           "codenet", job_id)
        jobs[job_id]["dependency"] = "qwen3_frozen_encoder_and_probe"
        # This part fits a supervised linear probe with CPU tensors.
        jobs[job_id]["gpu"] = False
    return jobs


class ServerPaths(ast.NodeTransformer):
    """Translate literal mount paths without altering embedded Python modules."""

    def __init__(self, work, inputs):
        self.paths = {"/kaggle/working": str(work), "/kaggle/input": str(inputs)}

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for source, target in self.paths.items():
                if node.value == source or node.value.startswith(source + "/"):
                    return ast.copy_location(ast.Constant(target + node.value[len(source):]), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "run_external"
                and not any(k.arg == "output_root" for k in node.keywords)):
            node.keywords.append(ast.keyword(arg="output_root",
                                             value=ast.Constant(self.paths["/kaggle/working"])))
        return node


def compile_cells(job, work, inputs):
    path = REVISION / job["notebook"]
    if sha256(path) != job["sha256"]:
        raise ValueError(f"Notebook hash changed: {path}")
    notebook = json.loads(path.read_bytes())
    transformer = ServerPaths(work, inputs)
    result = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        filename = f"{path}#cell{i}"
        tree = transformer.visit(ast.parse(code(cell), filename=filename))
        result.append((i, compile(ast.fix_missing_locations(tree), filename, "exec")))
    return result


def find_bundle(directory, dataset):
    if not dataset:
        return None
    root = directory / BUNDLES[dataset][0]
    options = (root, root / "clean_data")
    bundle = next((p for p in options if (p / "metadata.json").is_file()), None)
    if bundle is None:
        raise FileNotFoundError(f"Unpacked {dataset} bundle required under {root}")
    for stem in ("codes.jsonl", "pairs.csv", "graph_spectra.jsonl"):
        if not any((bundle / (stem + suffix)).is_file() for suffix in ("", ".gz", ".gz.tmp")) and not (
                bundle / stem).is_dir():
            raise FileNotFoundError(f"Missing {stem} in {bundle}")
    return bundle


def unpack(archives, directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for dataset, (name, digest) in BUNDLES.items():
        source = archives / f"{name}.zip"
        target = directory / name
        if target.exists():
            find_bundle(directory, dataset)
            continue
        if sha256(source) != digest:
            raise ValueError(f"Unexpected data archive hash: {source}")
        staging = directory / f"{name}.unpacking"
        staging.mkdir()
        with ZipFile(source) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename or ":" in member.filename:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(staging)
        staging.rename(target)
        find_bundle(directory, dataset)
        print(f"Verified and unpacked {dataset}", flush=True)


def load_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_record(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def usable_json(path):
    try:
        value = load_json(path, {})
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def resume_priority(job, root):
    work = root / job["id"]
    if usable_json(work / "server-result.json").get("complete"):
        return (2, 0)
    for report in work.glob("*/status.json"):
        state = usable_json(report)
        if (state.get("status") == "incomplete_session_budget"
                and report.with_name("progress.pt").is_file()
                and state.get("optimizer_steps", 0) > 0):
            fixed = job["group"] == "focused"
            return (0, -state["optimizer_steps"] / 1410 if fixed else 0)
    return (1, 0)


@contextmanager
def gpu_telemetry(output, enabled):
    if not enabled or shutil.which("nvidia-smi") is None:
        if enabled:
            print("GPU telemetry unavailable: nvidia-smi is missing", flush=True)
        yield
        return
    with (output / f"gpu-telemetry-{time.time_ns()}.csv").open("wb") as stream:
        command = ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,"
                   "utilization.memory,memory.used,memory.total,power.draw",
                   "--format=csv", "--loop=5"]
        with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as monitor:
            try:
                yield
            finally:
                monitor.terminate()


def paired_inputs_ready(root, methods):
    groups = {method: [] for method in methods}
    for path in root.glob("*/*/final_test.json"):
        result = usable_json(path)
        arm = result.get("arm", "")
        if arm.startswith("transfer/"):
            pieces = arm.split("/", 2)
            arm = pieces[1] if len(pieces) == 3 else "full"
        elif arm.startswith("baseline/"):
            arm = "/".join(arm.split("/")[:2])
        if arm not in groups or not (path.parent / "test_predictions.csv.gz").is_file():
            continue
        condition = usable_json(path.parent / "transfer_metadata.json").get("key", "main")
        groups[arm].append((condition, result.get("seed")))
    a, b = (groups[method] for method in methods)
    conditions = {condition for condition, _ in a}
    expected = {(condition, seed) for condition in conditions for seed in SEEDS}
    return bool(expected) and len(a) == len(b) == len(expected) and set(a) == set(b) == expected


def restore_inputs(settings, work):
    if settings.get("RESUME_CHECKPOINT"):
        checkpoint = Path(settings["RESUME_CHECKPOINT"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if list(work.rglob("progress.pt")):
            # The notebook's own progress takes precedence at later sessions.
            settings.pop("RESUME_CHECKPOINT")
            settings.pop("RESUME_TOKENS_DIR", None)
    tokens = settings.get("RESUME_TOKENS_DIR")
    if tokens:
        token_dir = Path(tokens)
        token_state = load_json(token_dir / "tokens.json")
        if not token_state or not all((token_dir / n).is_file() for n in ("tokens.npy", "masks.npy")):
            raise FileNotFoundError(f"Checkpoint token cache incomplete: {token_dir}")
        revision = token_state["revision"]
        if settings.get("MODEL_REVISION", revision) != revision:
            raise ValueError("Model revision differs from the saved token cache")
        settings["MODEL_REVISION"] = revision


def run_worker(spec):
    job, work = spec["job"], Path(spec["work"])
    inputs = Path(spec["inputs"])
    settings = dict(job["settings"])
    settings.update(spec["overrides"])
    if job.get("dependency"):
        parent = work.parent / job["dependency"] / "qwen3_embeddings"
        if "EMBEDDINGS_DIR" not in settings:
            settings["EMBEDDINGS_DIR"] = str(parent)
    if spec.get("bundle"):
        settings.setdefault("CLEAN_DATA_DIR", spec["bundle"])
    if spec.get("session_hours") is not None:
        settings["SESSION_HOURS"] = spec["session_hours"]
    settings["FINAL_TEST"] = spec["final"]
    restore_inputs(settings, work)
    if spec["final"]:
        candidates = list(work.rglob("progress.pt"))
        if not candidates and settings.get("RESUME_CHECKPOINT"):
            candidates.append(Path(settings["RESUME_CHECKPOINT"]))
        if len(candidates) != 1:
            raise ValueError("Final evaluation requires exactly one completed checkpoint")
        checkpoint_report = load_json(candidates[0].with_name("status.json"), {})
        if not checkpoint_report.get("complete"):
            raise ValueError("Final evaluation requires a completed training status report")
        settings["MEASURE_COST"] = False
    ns = {"__name__": "__main__"}
    for i, cell in compile_cells(job, work, inputs):
        exec(cell, ns)
        if i == 1:
            ns.update(settings)
    report = ns.get("summary", ns.get("status"))
    result = {"job": job["id"], "finished_at": time.time(), "final_test": spec["final"]}
    if isinstance(report, dict):
        result["status"] = report.get("status", "execution_finished")
        result["complete"] = report.get("complete", result["status"] == "complete")
    else:
        result.update(status="execution_finished", complete=True)
    if spec["final"]:
        result["complete"] = bool(list(work.rglob("final_test.json")))
        if result["complete"]:
            result["status"] = "final_test_complete"
    write_record(work / ("server-final.json" if spec["final"] else "server-result.json"), result)
    print(json.dumps(result), flush=True)
    return 0


def selected_jobs(catalog, profile, patterns):
    if patterns:
        found = {job_id for pattern in patterns for job_id in catalog if fnmatch.fnmatchcase(job_id, pattern)}
        for pattern in patterns:
            if not any(fnmatch.fnmatchcase(job_id, pattern) for job_id in found):
                raise ValueError(f"No jobs match {pattern!r}")
        return [job for job_id, job in catalog.items() if job_id in found]
    if profile == "remaining":
        return [job for job_id, job in catalog.items() if job_id in PENDING]
    if profile == "focused":
        return [job for job in catalog.values() if job["group"] == "focused"]
    if profile == "fresh":
        ids = (
            "focused_source_only_c4_adapted_seed231",
            "focused_direct_c4_adapted_seed123", "focused_direct_c4_adapted_seed231",
            "focused_direct_full_seed231", "focused_direct_mean_max_seed231",
            *(f"atcoder_without_spectral_seed{seed}_f1_v1" for seed in SEEDS),
        )
        return [catalog[job_id] for job_id in ids]
    if profile == "lease":
        focused = [job for job in catalog.values() if job["group"] == "focused"]
        first = [
            "codenet_codebert_seed42", "atcoder_codebert_seed42",
            "bigclonebench_codebert_seed42", "bigclonebench_cdlh_seed42",
            *[job["id"] for job in focused
              if job["id"].endswith("_seed42")
              and job["id"].startswith(("focused_source_only_", "focused_direct_"))
              and job["settings"].get("ARM")],
        ]
        next_wave = [job["id"] for job in focused if job["id"].endswith("_seed42")]
        qwen = sorted(job_id for job_id in PENDING if "qwen" in job_id)
        repeats = [job["id"] for job in focused if not job["id"].endswith("_seed42")]
        order = dict.fromkeys(first + next_wave + qwen + repeats)
        return [catalog[job_id] for job_id in order]
    if profile in {"training", "matrix"}:
        groups = {
            "01_baselines_seed42", "02_controls_seed42", "03_selected_repeats",
            "06_atcoder_confirmation_v1", "07_atcoder_readout_check_v1", "atcoder_followup"
        }
        if profile == "matrix":
            groups.update({"expanded_controls", "focused"})
        return [job for job in catalog.values() if job["group"] in groups and job["gpu"]]
    return list(catalog.values())


def watch_worker(command, *, work, stream, env, deadline, idle_seconds):
    """Bound a hung worker and its child processes, leaving saved files intact."""
    worker = subprocess.Popen(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                              env=env, start_new_session=(os.name == "posix"))
    last_size = -1
    last_activity = time.monotonic()
    while True:
        try:
            return worker.wait(timeout=1), None
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            size = os.fstat(stream.fileno()).st_size
            if size != last_size:
                last_activity, last_size = now, size
            reason = ("lease_cutoff" if deadline and now >= deadline else
                      "idle_timeout" if idle_seconds and now - last_activity >= idle_seconds else None)
            if not reason:
                continue
            if os.name == "posix":
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if os.name == "posix":
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif worker.poll() is None:
                worker.kill()
            return worker.wait(), reason


def launch(job, args, inputs, device):
    work = args.output / job["id"]
    work.mkdir(parents=True, exist_ok=True)
    result_path = work / ("server-final.json" if args.final else "server-result.json")
    if load_json(result_path, {}).get("complete"):
        return job["id"], "already_complete"
    if job.get("dependency") and "EMBEDDINGS_DIR" not in inputs.get(job["id"], {}):
        dependency = load_json(args.output / job["dependency"] / "server-result.json", {})
        if not dependency.get("complete"):
            return job["id"], "waiting_for_embeddings"
    session_hours = args.session_hours
    if args.hard_deadline and time.monotonic() >= args.hard_deadline:
        return job["id"], "waiting_for_next_lease"
    if args.deadline:
        available = (args.deadline - time.monotonic()) / 3600
        if available <= 0.75:
            return job["id"], "waiting_for_next_lease"
        session_hours = min(available, session_hours or 10)
    elif args.hard_deadline:
        session_hours = min(10, 0.75 + (args.hard_deadline - time.monotonic()) / 3600)
    overrides = dict(inputs.get(job["id"], {}))
    unknown = set(overrides) - OVERRIDES
    if unknown:
        raise ValueError(f"{job['id']}: unknown input setting {sorted(unknown)}")
    variant = overrides.pop("NOTEBOOK_VARIANT", None)
    if variant is not None:
        if variant != job["id"] + "__input_v1":
            raise ValueError(f"{job['id']}: unsupported checkpoint variant {variant!r}")
        archived = make_catalog()[variant]
        job = {**archived, "id": job["id"]}
    for key, value in overrides.items():
        if value is not None and key != "MODEL_REVISION":
            overrides[key] = str((args.config.resolve().parent / value).resolve())
    bundle = (None if "CLEAN_DATA_DIR" in overrides else find_bundle(args.data, job["dataset"]))
    if args.final and (job["group"] in {"05_analysis", "optional", "frozen_probe"}
                       or not job["gpu"]):
        raise ValueError(f"Final mode is for checkpoint-based training jobs: {job['id']}")
    spec = {"job": job, "work": str(work), "inputs": str(args.inputs), "overrides": overrides,
            "bundle": str(bundle) if bundle else None, "final": args.final,
            "session_hours": session_hours}
    spec_path = work / "server-job.json"
    write_record(spec_path, spec)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = device if job["gpu"] else ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(args.threads)
    print(f"{job['id']}: starting on {'GPU ' + device if job['gpu'] else 'CPU'}", flush=True)
    log = work / f"session-{time.time_ns()}.log"
    with log.open("wb") as stream:
        exit_code, reason = watch_worker(
            [sys.executable, __file__, "_worker", str(spec_path)], work=work, stream=stream,
            env=env, deadline=args.hard_deadline, idle_seconds=args.idle_minutes * 60,
        )
    if reason:
        write_record(work / "server-interrupted.json", {"reason": reason, "log": log.name,
                                                        "finished_at": time.time()})
        return job["id"], f"failed ({reason}; {log})"
    result = load_json(result_path, {}) if exit_code == 0 else {}
    return job["id"], result.get("status", f"failed ({log})")


def run_queue(jobs, args, catalog):
    if args.config and not args.config.is_file():
        raise FileNotFoundError(args.config)
    inputs = load_json(args.config, {}) if args.config else {}
    devices = args.gpus.split(",") if args.gpus else []
    gpu_jobs = [j for j in jobs if j["gpu"]]
    cpu_jobs = [j for j in jobs if not j["gpu"]]
    if gpu_jobs and (not devices or any(not d for d in devices) or len(devices) != len(set(devices))):
        raise ValueError("Supply unique CUDA IDs using --gpus 0,1 or --gpus 0")
    for job in jobs:
        overrides = inputs.get(job["id"], {})
        if "CLEAN_DATA_DIR" not in overrides:
            find_bundle(args.data, job["dataset"])
    # Refuse concurrent queues that write into the same experiment tree.
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".server.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.deadline = (time.monotonic() + (args.lease_hours - 0.25) * 3600
                         if args.lease_hours is not None else None)
        auto_analysis = args.profile in {"lease", "fresh"} and not args.final
        args.hard_deadline = args.deadline - 600 if auto_analysis else args.deadline
        failures = 0
        def lane(device, queue, settings=args):
            results = []
            while True:
                try:
                    job = queue.get_nowait()
                except Empty:
                    break
                try:
                    results.append(launch(job, settings, inputs, device))
                except Exception as exc:
                    results.append((job["id"], f"failed ({exc})"))
                print(*results[-1], flush=True)
            return results
        def pending(batch):
            queue = Queue()
            for job in batch:
                queue.put(job)
            return queue
        if gpu_jobs:
            if args.resume_first and not args.final:
                gpu_jobs.sort(key=lambda job: resume_priority(job, args.output))
            queue = pending(gpu_jobs)
            with gpu_telemetry(args.output, args.gpu_telemetry), concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(lane, device, queue) for device in devices]
                failures += sum(status.startswith("failed") for future in futures
                                for _, status in future.result())
        failures += sum(status.startswith("failed") for _, status in lane("", pending(cpu_jobs)))
        if auto_analysis:
            post = copy.copy(args)
            post.deadline, post.hard_deadline = None, args.deadline
            post.inputs = args.output
            collection = copy.copy(post)
            collection.output = args.output / "_analysis"
            collection.inputs = collection.output / f"validation-inputs-{time.time_ns()}"
            for report in args.output.glob("*/*/status.json"):
                data = usable_json(report)
                if data:
                    target = collection.inputs / report.relative_to(args.output)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_record(target, data)
            if collection.inputs.exists():
                collector = {**catalog["validation_screening"], "id": f"validation_screening_{time.time_ns()}"}
                failures += sum(status.startswith("failed")
                                for _, status in lane("", pending([collector]), collection))
            post.final = True
            finals = []
            for job_id in args.final_jobs:
                job = catalog[job_id]
                complete = [p for p in (args.output / job_id).rglob("status.json")
                            if usable_json(p).get("complete") and p.with_name("progress.pt").is_file()]
                if len(complete) == 1:
                    finals.append(job)
                else:
                    print(job_id, "waiting_for_completed_checkpoint", flush=True)
            if finals:
                queue = pending(finals)
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                    futures = [pool.submit(lane, device, queue, post) for device in devices]
                    failures += sum(status.startswith("failed") for future in futures
                                    for _, status in future.result())
            post.final = False
            post.output = args.output / "_analysis"
            if args.paired_methods:
                analysis = {**catalog["paired_problem_uncertainty"],
                            "settings": {**dict(zip(("METHOD_A", "METHOD_B"), args.paired_methods)),
                                         "METRIC": args.paired_metric}}
                if paired_inputs_ready(args.output, args.paired_methods):
                    failures += sum(status.startswith("failed")
                                    for _, status in lane("", pending([analysis]), post))
                else:
                    print("paired_problem_uncertainty waiting_for_test_predictions", flush=True)
    return int(bool(failures))


def cli():
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        return run_worker(load_json(Path(sys.argv[2])))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "verify", "unpack", "run"))
    parser.add_argument("--profile", choices=("remaining", "focused", "lease", "fresh", "training", "matrix", "all"), default="remaining")
    parser.add_argument("--jobs", nargs="+", default=(), metavar="JOB_OR_GLOB")
    parser.add_argument("--external", choices=EXTERNAL, default="c4_adapted")
    parser.add_argument("--sibling", choices=("mean_max", "attention"), default="mean_max")
    parser.add_argument("--data", type=Path, default=Path("server-data"))
    parser.add_argument("--archives", type=Path, default=Path("benchmark-zips"))
    parser.add_argument("--output", type=Path, default=Path("server-output"))
    parser.add_argument("--inputs", type=Path, default=Path("server-input"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume-first", action="store_true",
                        help="prioritize incomplete local checkpoints before fresh jobs")
    parser.add_argument("--gpu-telemetry", action="store_true",
                        help="sample NVIDIA utilization, VRAM and power every five seconds")
    parser.add_argument("--session-hours", type=float)
    parser.add_argument("--lease-hours", type=float, help="remaining rental time; leave 15 minutes for export")
    parser.add_argument("--idle-minutes", type=float, default=25,
                        help="stop a worker that produces no log output; zero disables this check")
    parser.add_argument("--final-jobs", nargs="*", default=(), metavar="FROZEN_JOB_ID",
                        help="after a lease, evaluate only these explicitly selected completed jobs")
    parser.add_argument("--paired-methods", nargs=2, metavar=("METHOD_A", "METHOD_B"),
                        help="after a lease, attempt the selected paired test analysis if predictions exist")
    parser.add_argument("--paired-metric", choices=("F1", "Accuracy"), default="F1")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--hardware", action="store_true", help="also verify CUDA and the data bundle paths")
    args = parser.parse_args()
    for field in ("data", "inputs", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if args.command == "unpack":
        unpack(args.archives, args.data)
        return 0
    if args.command == "run" and args.profile == "all" and not args.jobs:
        parser.error("Select explicit jobs or a run profile. The inventory contains alternative and archived jobs.")
    if args.command == "run" and args.profile in {"lease", "fresh"} and args.lease_hours is None:
        parser.error("Set --lease-hours to the actual remaining paid time")
    if args.session_hours is not None and args.session_hours <= 0.75:
        parser.error("--session-hours must exceed the 45-minute checkpoint reserve")
    if args.lease_hours is not None and args.lease_hours <= 1:
        parser.error("--lease-hours must exceed the combined one-hour reserve")
    catalog = make_catalog(args.external, args.sibling)
    if args.idle_minutes < 0:
        parser.error("--idle-minutes cannot be negative")
    if args.final_jobs or args.paired_methods:
        if args.command != "run" or args.profile != "lease" or args.final:
            parser.error("Post-training selections require run --profile lease")
        if any(j not in catalog or not catalog[j]["gpu"] for j in args.final_jobs):
            parser.error("--final-jobs requires known GPU experiment IDs")
        if args.paired_methods and len(set(args.paired_methods)) != 2:
            parser.error("--paired-methods requires two distinct methods")
    jobs = selected_jobs(catalog, args.profile, args.jobs)
    if args.command == "list":
        print(json.dumps(jobs, indent=2))
    elif args.command == "verify":
        for job in jobs:
            compile_cells(job, args.output / job["id"], args.inputs)
        if args.hardware:
            import torch
            if any(j["gpu"] for j in jobs) and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Verify NVIDIA drivers and the PyTorch CUDA build.")
            for dataset in sorted({job["dataset"] for job in jobs if job["dataset"]}):
                print(dataset, find_bundle(args.data, dataset))
            print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
        print(f"Verified {len(jobs)} experiment entries")
    else:
        return run_queue(jobs, args, catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
