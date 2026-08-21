"""Build the two self-contained Kaggle notebooks used by RQ1."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# The generated notebooks run on Kaggle, so they live with the other
# Kaggle notebooks rather than beside this builder.
DESTINATION = Path(__file__).resolve().parents[2] / "kaggle" / "rq1"
SOURCES = (
    (
        ROOT / "kaggle" / "rq2" / "atcoder" / "method" / "spectra_siam_lex.ipynb",
        DESTINATION / "01_atcoder_export_latent_graphs.ipynb",
        "atcoder_v3",
        "ATCoder V3",
    ),
    (
        ROOT / "kaggle" / "rq2" / "xglue" / "method" / "spectra_siam_lex.ipynb",
        DESTINATION / "02_xglue_export_latent_graphs.ipynb",
        "codexglue_v3",
        "CodeXGLUE (XGLUE) V3",
    ),
)


EXPORT_HELPERS = r'''
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_rq1_latent_graphs(
    model: CanonicalSpectraSiam,
    frames: dict[str, pd.DataFrame],
    graphs: dict[str, CanonicalGraph],
    checkpoint_path: Path,
    validation_selection_metric: str,
    validation_is_balanced: bool,
    validation_label_counts: dict[str, int],
) -> dict:
    """Export every benchmark endpoint's soft latent graph and exact spectrum."""
    export_root = WORK_DIR / f"rq1_{RQ1_DATASET_SLUG}_latent_graphs"
    if export_root.exists():
        shutil.rmtree(export_root)
    shard_root = export_root / "shards"
    shard_root.mkdir(parents=True)

    ordered_ids = sorted(graphs)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise RuntimeError("Latent export received duplicate code IDs")
    shard_records = []
    model.eval()
    for shard_index, shard_start in enumerate(range(0, len(ordered_ids), RQ1_EXPORT_SHARD_SIZE)):
        shard_ids = ordered_ids[shard_start:shard_start + RQ1_EXPORT_SHARD_SIZE]
        adjacency_parts, eigenvalue_parts, temperature_parts = [], [], []
        for batch_start in tqdm(
            range(0, len(shard_ids), RQ1_EXPORT_BATCH_SIZE),
            desc=f"RQ1 latent shard {shard_index + 1}",
            leave=False,
        ):
            batch_ids = shard_ids[batch_start:batch_start + RQ1_EXPORT_BATCH_SIZE]
            batch = pack_graphs([graphs[code_id] for code_id in batch_ids], MAX_AST_NODES).to(DEVICE)
            with torch.no_grad():
                _, _, auxiliary = model.encoder(batch, None)
            if "latent_adjacency" not in auxiliary or "latent_eigenvalues" not in auxiliary:
                raise RuntimeError("RQ1 encoder did not expose latent graph tensors in evaluation mode")
            adjacency_parts.append(auxiliary["latent_adjacency"].cpu().numpy().astype(np.float16))
            eigenvalue_parts.append(auxiliary["latent_eigenvalues"].cpu().numpy().astype(np.float32))
            temperature_parts.append(auxiliary["adaptive_temperature"].cpu().numpy().astype(np.float32))

        adjacency = np.concatenate(adjacency_parts, axis=0)
        eigenvalues = np.concatenate(eigenvalue_parts, axis=0)
        temperatures = np.concatenate(temperature_parts, axis=0)
        if adjacency.shape != (len(shard_ids), LATENT_NODES, LATENT_NODES):
            raise RuntimeError(f"Unexpected latent adjacency shape {adjacency.shape}")
        if eigenvalues.shape != (len(shard_ids), LATENT_NODES):
            raise RuntimeError(f"Unexpected latent eigenvalue shape {eigenvalues.shape}")
        shard_path = shard_root / f"latent_graphs_{shard_index:05d}.npz"
        np.savez_compressed(
            shard_path,
            code_ids=np.asarray(shard_ids),
            languages=np.asarray([graphs[code_id].language for code_id in shard_ids]),
            original_node_counts=np.asarray([graphs[code_id].node_count for code_id in shard_ids], dtype=np.int32),
            adjacency=adjacency,
            eigenvalues=eigenvalues,
            adaptive_temperature=temperatures,
        )
        shard_records.append(
            {
                "path": f"shards/{shard_path.name}",
                "code_count": len(shard_ids),
                "first_code_id": shard_ids[0],
                "last_code_id": shard_ids[-1],
                "bytes": shard_path.stat().st_size,
            }
        )

    pair_frame = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()], ignore_index=True
    )
    pair_path = export_root / "rq1_pairs.csv.gz"
    pair_frame.to_csv(pair_path, index=False, compression="gzip")
    checkpoint_path = Path(checkpoint_path)
    manifest = {
        "format": "spectra-rq1-latent-graphs-v1",
        "dataset": RQ1_DATASET_SLUG,
        "research_question": "learned latent graph versus conventional program graphs under PSS",
        "code_count": len(ordered_ids),
        "pair_count": len(pair_frame),
        "pair_counts_by_split": {
            str(key): int(value) for key, value in pair_frame.groupby("split").size().items()
        },
        "latent_nodes": LATENT_NODES,
        "adjacency_dtype": "float16",
        "eigenvalue_dtype": "float32",
        "eigenvalue_source": "normalized Laplacian of the pre-refinement learned soft adjacency",
        "adjacency_source": "symmetric sigmoid attention plus projected conventional-graph prior",
        "training_scope": "official train pairs only; best checkpoint selected on official validation",
        "validation_selection_metric": validation_selection_metric,
        "validation_is_balanced": bool(validation_is_balanced),
        "validation_label_counts": validation_label_counts,
        "evaluation_scope": "official validation threshold selection and untouched official test",
        "classifier_used_by_local_pss": False,
        "embedding_used_by_local_pss": False,
        "conventional_graph_types": ["ast", "cfg", "ddg", "cpg"],
        # This fingerprint makes a downloaded graph archive auditable against
        # the main SPECTRA-Siam notebook.  RQ1 must never silently reuse a
        # latent graph produced before an input-ablation change.
        "method_variant": METHOD_VARIANT,
        "method_input_signature": {
            "input_ablation": INPUT_ABLATION,
            "use_node_lexical": bool(model.config.use_node_lexical),
            "use_source_lexical": bool(model.config.use_source_lexical),
        },
        "model_config": asdict(model.config),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "shards": shard_records,
    }
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = WORK_DIR / f"rq1_{RQ1_DATASET_SLUG}_latent_graphs.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(manifest_path, "manifest.json")
        archive.write(pair_path, "rq1_pairs.csv.gz")
        archive.write(checkpoint_path, checkpoint_path.name)
        for shard in shard_records:
            archive.write(export_root / shard["path"], shard["path"])
    export = {
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "code_count": len(ordered_ids),
        "pair_count": len(pair_frame),
        "shard_count": len(shard_records),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    print("RQ1 latent graph export:", json.dumps(export, indent=2))
    return export


'''


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _expose_latent_graphs(model_source: str) -> str:
    encoder_start = model_source.index("class CanonicalSpectraEncoder")
    start = model_source.index("        return embedding, descriptor, {", encoder_start)
    end_marker = "\n        }\n\n\nclass CanonicalSpectraSiam"
    end = model_source.index(end_marker, start)
    original = model_source[start:end + len("\n        }")]
    replacement = original.replace(
        "        return embedding, descriptor, {",
        "        encoder_auxiliary = {",
        1,
    )
    replacement += '''
        if not self.training:
            # RQ1 export only: these detached tensors cannot affect training,
            # the pair classifier, or any non-zero loss term.
            encoder_auxiliary["latent_adjacency"] = adjacency.detach()
            encoder_auxiliary["latent_eigenvalues"] = eigenvalues.detach()
        return embedding, descriptor, encoder_auxiliary'''
    return model_source[:start] + replacement + model_source[start + len(original):]


def build(source_path: Path, destination: Path, slug: str, title: str) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = _lines(
        f"# RQ1 — {title} learned latent-graph export\n\n"
        "This self-contained Kaggle run trains the repository's latest descriptor-only SPECTRA-Siam, freezes the validation-selected checkpoint, and exports the learned soft latent graph and normalized-Laplacian spectrum for every benchmark pair endpoint. The archive is evaluated locally with the same PSS implementation used for AST, CFG, DDG, and CPG.\n"
    )

    config = "".join(notebook["cells"][1]["source"])
    config = config.replace("import json\n", "import hashlib\nimport json\nimport shutil\n", 1)
    marker = "RUN_EXPERIMENT = True\n"
    additions = (
        "RUN_EXPERIMENT = True\n"
        "RQ1_EXPORT_LATENT_GRAPHS = True\n"
        f'RQ1_DATASET_SLUG = "{slug}"\n'
        "RQ1_EXPORT_BATCH_SIZE = 64\n"
        "RQ1_EXPORT_SHARD_SIZE = 4096\n"
    )
    if config.count(marker) != 1:
        raise RuntimeError(f"Could not configure RQ1 export in {source_path}")
    config = config.replace(marker, additions, 1)
    notebook["cells"][1]["source"] = _lines(config)

    model = "".join(notebook["cells"][9]["source"])
    required = (
        "separately normalized spectral blocks only",
        "self.block_sizes = (config.density_bins, config.heat_samples)",
        "self.classifier(spectral_pair_features)",
        "embedding_contrastive_weight must remain zero",
    )
    missing = [fragment for fragment in required if fragment not in model]
    if missing:
        raise RuntimeError(f"{source_path} is not the latest SPECTRA-Siam design: {missing}")
    notebook["cells"][9]["source"] = _lines(_expose_latent_graphs(model))

    helpers = "".join(notebook["cells"][11]["source"])
    run_start = helpers.index("def run_experiment() -> dict:")
    helpers = helpers[:run_start] + EXPORT_HELPERS + helpers[run_start:]
    elapsed_marker = "    elapsed_seconds = float(time.perf_counter() - started)\n"
    export_call = (
        "    rq1_export = (\n"
        "        export_rq1_latent_graphs(\n"
        "            model, frames, graphs, final_path, validation_selection_metric,\n"
        "            validation_is_balanced, validation_label_counts,\n"
        "        )\n"
        "        if RQ1_EXPORT_LATENT_GRAPHS else None\n"
        "    )\n"
        + elapsed_marker
    )
    if helpers.count(elapsed_marker) != 1:
        raise RuntimeError(f"Could not attach RQ1 export to {source_path}")
    helpers = helpers.replace(elapsed_marker, export_call, 1)
    result_marker = '    result = {"dataset": DATASET_KEY, '
    if helpers.count(result_marker) != 1:
        raise RuntimeError(f"Could not add RQ1 result metadata to {source_path}")
    helpers = helpers.replace(
        result_marker,
        '    result = {"dataset": DATASET_KEY, "rq1_latent_export": rq1_export, ',
        1,
    )
    notebook["cells"][11]["source"] = _lines(helpers)

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    for source, destination, slug, title in SOURCES:
        build(source, destination, slug, title)
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
