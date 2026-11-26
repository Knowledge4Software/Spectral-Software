import ast
from pathlib import Path

from scripts.run_revision_server import ServerPaths, make_catalog, selected_jobs


def test_inventory_spans_all_seeds_and_preserves_transfer_budget():
    catalog = make_catalog()
    assert len(selected_jobs(catalog, "remaining", [])) == 7
    focused = selected_jobs(catalog, "focused", [])
    assert len(focused) == 45
    lease = selected_jobs(catalog, "lease", [])
    assert len(lease) == 52 and len({j["id"] for j in lease}) == 52
    assert sum(j["group"] == "focused" for j in lease[:8]) == 4
    assert all(j["gpu"] for j in lease)
    assert {j["id"].rsplit("seed", 1)[-1] for j in focused} == {"42", "123", "231"}
    for dataset in ("codenet", "atcoder", "bigclonebench"):
        for seed in (42, 123, 231):
            for arm in ("full", "mean_max", "classification_only"):
                assert catalog[f"{dataset}_{arm}_seed{seed}"]["dataset"] == dataset
    assert catalog["qwen3_frozen_probe_seed231"]["dependency"] == "qwen3_frozen_encoder_and_probe"


def test_legacy_paths_are_mapped_without_changing_embedded_code():
    embedded = 'from pathlib import Path\nroot = Path("/kaggle/working")\n'
    source = f"_sources = {{'jobs.py': {embedded!r}}}\n"
    source += 'path = Path(f"/kaggle/working/run_{42}")\nresult = run_external()'
    tree = ServerPaths(Path("/tmp/job"), Path("/tmp/input")).visit(ast.parse(source))
    ns = {"Path": Path, "run_external": lambda **kwargs: kwargs}
    exec(compile(ast.fix_missing_locations(tree), "server_test", "exec"), ns)
    assert ns["_sources"]["jobs.py"] == embedded
    assert ns["path"] == Path("/tmp/job/run_42")
    assert Path(ns["result"]["output_root"]) == Path("/tmp/job")
