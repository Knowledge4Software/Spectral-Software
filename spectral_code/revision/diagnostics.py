"""Validation-only spectral distance, size/density and slot-profile diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .batching import pack_graphs
from .model import RevisionModelConfig, RevisionPairModel
from .protocol import RunConfig, fingerprint
from .training import _load_checkpoint, load_pair_data, SessionBudgetReached
from .versioning import training_implementation_hash
from tqdm.auto import tqdm


def _fixed_features(eigenvalues):
    values = np.asarray(eigenvalues, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return None
    density = np.exp(-.5 * ((values[:, None] - np.linspace(0, 2, 32)) / .07) ** 2).mean(0)
    density /= max(density.sum(), 1e-12)
    heat = np.exp(-values[:, None] * np.logspace(-2, 2, 24)).mean(0)
    return np.r_[density, heat]


def _distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a / max(np.linalg.norm(a), 1e-12) - b / max(np.linalg.norm(b), 1e-12)))


@torch.no_grad()
def validation_diagnostics(root, checkpoint, output, *, clock=None, maximum_pairs=4000, seed=9173):
    state = _load_checkpoint(checkpoint, "cpu")
    if state.get("implementation_hash") != training_implementation_hash():
        raise ValueError("Diagnostic input/runtime differs from checkpoint; use its original notebook or an input-v2 checkpoint")
    if state["model_config"].get("family") is not None or state["best_model"] is None:
        raise ValueError("Use a validation-selected SPECTRA checkpoint")
    corpus, frames, data_hash = load_pair_data(Path(root), require_disjoint=state["training"]["dataset"] == "codenet")
    if data_hash != state["data_hash"]:
        raise ValueError("Diagnostic bundle differs from checkpoint; use a main-experiment checkpoint, not a transfer job")
    frame = frames["valid"]
    if len(frame) > maximum_pairs:
        # Uniform seeded sample independent of predictions; report all strata.
        frame = frame.sample(maximum_pairs, random_state=seed).reset_index(drop=True)
    ids = sorted(set(frame.left_id) | set(frame.right_id))
    config = RunConfig(**state["training"])
    graphs = corpus.load_graphs(ids, max_nodes=config.max_nodes)
    if set(ids) - set(graphs):
        raise ValueError("Validation endpoints are missing graphs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RevisionPairModel(RevisionModelConfig(**state["model_config"])).to(device).eval()
    model.load_state_dict(state["best_model"])
    model.encoder.capture_diagnostics = True
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts, profiles, profile_counts = {}, {}, {}
    try:
        for start in tqdm(range(0, len(ids), config.batch_size), desc="Validation descriptors",
                          unit="batch", mininterval=2):
            if clock:
                clock.check()
            keys = ids[start:start + config.batch_size]
            batch = pack_graphs([graphs[key] for key in keys], config.max_nodes).to(device)
            _, descriptor, auxiliary = model.encoder(batch, None)
            assignments = auxiliary["assignment"].float()
            onehot = torch.nn.functional.one_hot(batch.canonical_ids, model.config.canonical_vocab_size).float()
            onehot = onehot * batch.mask[:, :, None]
            masses = torch.einsum("bnk,bnc->bkc", assignments, onehot).cpu().numpy()
            for i, key in enumerate(keys):
                graph = graphs[key]
                adjacency = auxiliary["adjacency"][i].float().cpu().numpy()
                values = auxiliary["eigenvalues"][i].float().cpu().numpy()
                vector = descriptor[i].float().cpu().numpy()
                n = graph.node_count
                observed_edges = {tuple(edge) for relation in (graph.edges[0], graph.edges[2]) for edge in relation if edge[0] != edge[1]}
                artifacts[key] = {"descriptor": vector, "eigenvalues": values,
                                  "nodes": n, "observed_density": len(observed_edges) / max(1, n * (n - 1)),
                                  "latent_density": float(adjacency.sum() / (len(adjacency) * (len(adjacency) - 1)))}
                language = graph.language
                normalized = masses[i] / masses[i].sum(-1, keepdims=True).clip(1e-12)
                profiles[language] = profiles.get(language, np.zeros_like(normalized)) + normalized
                profile_counts[language] = profile_counts.get(language, 0) + 1
        rows = []
        for row in tqdm(frame.to_dict("records"), desc="Pair distances", unit="pair", mininterval=2):
            a, b = str(row["left_id"]), str(row["right_id"])
            base = {key: row.get(key) for key in ("pair_id", "label", "configuration_id", "left_problem_id", "right_problem_id")}
            base.update(mean_nodes=(artifacts[a]["nodes"] + artifacts[b]["nodes"]) / 2,
                        mean_observed_density=(artifacts[a]["observed_density"] + artifacts[b]["observed_density"]) / 2,
                        mean_latent_density=(artifacts[a]["latent_density"] + artifacts[b]["latent_density"]) / 2)
            rows.append({**base, "view": "learned_eigenvalue_density_heat", "distance": _distance(
                _fixed_features(artifacts[a]["eigenvalues"]), _fixed_features(artifacts[b]["eigenvalues"]))})
            rows.append({**base, "view": "learned_full_descriptor", "distance": _distance(artifacts[a]["descriptor"], artifacts[b]["descriptor"])})
            for layer in ("ast", "cfg", "ddg", "cpg"):
                left = _fixed_features(graphs[a].eigenvalues.get(layer, []))
                right = _fixed_features(graphs[b].eigenvalues.get(layer, []))
                if left is not None and right is not None:
                    rows.append({**base, "view": "fixed_" + layer, "distance": _distance(left, right)})
        distances = pd.DataFrame(rows)
        # Quantile boundaries use validation covariates only. The same strata
        # apply to every view. Match label counts within each stratum.
        covariates = distances.drop_duplicates("pair_id")
        size_edges = np.unique(np.quantile(covariates.mean_nodes, [0, .25, .5, .75, 1]))
        density_edges = np.unique(np.quantile(covariates.mean_observed_density, [0, .25, .5, .75, 1]))
        distances["size_bin"] = np.searchsorted(size_edges[1:-1], distances.mean_nodes)
        distances["density_bin"] = np.searchsorted(density_edges[1:-1], distances.mean_observed_density)
        summaries = []
        for view, group in distances.groupby("view"):
            matched = []
            for _, cell in group.groupby(["configuration_id", "size_bin", "density_bin"]):
                count = min(int(cell.label.eq(0).sum()), int(cell.label.eq(1).sum()))
                if count:
                    matched += [cell[cell.label.eq(label)].sample(count, random_state=seed) for label in (0, 1)]
            for selection, subset in (("all_available", group), ("size_density_label_matched", pd.concat(matched) if matched else group.iloc[:0])):
                if subset.empty or subset.label.nunique() != 2:
                    continue
                summaries.append({"view": view, "selection": selection, "pairs": len(subset),
                                  "distance_AUROC": float(roc_auc_score(subset.label, -subset.distance)),
                                  "clone_mean_distance": float(subset[subset.label.eq(1)].distance.mean()),
                                  "nonclone_mean_distance": float(subset[subset.label.eq(0)].distance.mean())})
        distances.to_csv(output / "validation_pair_distances.csv.gz", index=False, compression="gzip")
        pd.DataFrame(summaries).to_csv(output / "validation_distance_summary.csv", index=False)
        profiles = {language: values / profile_counts[language] for language, values in profiles.items()}
        profile_rows = []
        for i, a in enumerate(sorted(profiles)):
            for b in sorted(profiles)[i + 1:]:
                # Slot identities are shared within this checkpoint only.
                profile_rows.append({"language_a": a, "language_b": b,
                                     "mean_slot_profile_L1": float(np.abs(profiles[a] - profiles[b]).sum(-1).mean())})
        pd.DataFrame(profile_rows).to_csv(output / "cross_language_slot_profiles.csv", index=False)
        np.savez_compressed(output / "slot_profiles.npz", **profiles)
        np.savez_compressed(output / "validation_code_spectra.npz", code_ids=np.asarray(ids),
                            eigenvalues=np.stack([artifacts[key]["eigenvalues"] for key in ids]),
                            descriptors=np.stack([artifacts[key]["descriptor"] for key in ids]))
        result = {"status": "complete", "split": "valid", "checkpoint_identity": state["identity"],
                  "validation_pair_sample_hash": fingerprint(frame.pair_id.astype(str).tolist()),
                  "pairs": len(frame), "codes": len(ids), "selection_seed": seed,
                  "size_quantiles": size_edges.tolist(), "density_quantiles": density_edges.tolist(),
                  "distance_definition": "L2 between individually L2-normalized descriptors",
                  "limitations": ["Diagnostics, not a theorem of language invariance", "Fixed views can have different available support; counts are reported",
                                  "Slot profiles are marginal language distributions, not aligned semantics or causal explanations",
                                  "Slots must not be compared across checkpoints without permutation alignment", "Truncated node counts, not original program sizes"]}
    except SessionBudgetReached:
        result = {"status": "incomplete_session_budget", "split": "valid", "codes_processed": len(artifacts),
                  "continuation": "Rerun this bounded diagnostic job; training checkpoint remains unchanged"}
    (output / "diagnostic_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
