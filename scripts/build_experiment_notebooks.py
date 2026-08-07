"""Generate the Kaggle experiment notebooks from the canonical SPECTRA-Siam notebook.

The method notebook is ~65k characters of implementation. Copying it by hand
four more times per experiment would guarantee the variants drift apart, so each
experiment notebook is *derived*: the implementation cells are taken verbatim
from a dataset's ``method/spectra_siam.ipynb`` and only the configuration cell
and the driver cell are replaced.

Re-run this whenever the method notebook changes::

    python scripts/build_experiment_notebooks.py
    python scripts/build_experiment_notebooks.py --check   # CI-style staleness check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"
EXPERIMENTS_ROOT = KAGGLE_ROOT / "experiments"
TEMPLATE = KAGGLE_ROOT / "codexglue_v3" / "method" / "spectra_siam.ipynb"

# Index of the cells that carry configuration and the run driver. Everything in
# between is implementation and is copied unchanged.
CONFIG_CELL = 1
DRIVER_CELL = -1


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def _load_template() -> dict:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Method notebook not found: {TEMPLATE}")
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _config_overrides(dataset_key: str, extra: str, profile: str = "comparison_50k") -> str:
    """Config cell of the method notebook with the experiment's overrides appended."""
    template = _load_template()
    source = "".join(template["cells"][CONFIG_CELL]["source"])
    source = source.replace(f'DATASET_KEY = "codexglue"', f'DATASET_KEY = "{dataset_key}"', 1)
    source = source.replace('RUN_PROFILE = "final_full"', f'RUN_PROFILE = "{profile}"', 1)
    return source + "\n\n" + extra


SWEEP_DRIVER = '''
# Sweep the latent graph size. Everything else is held at the canonical setting,
# so a difference in F1 is attributable to latent capacity alone.
LATENT_NODE_GRID = [16, 24, 32, 48]

sweep_rows = []
for latent_nodes in LATENT_NODE_GRID:
    print("\\n" + "=" * 70)
    print(f"LATENT_NODES = {latent_nodes}")
    print("=" * 70, flush=True)
    LATENT_NODES = latent_nodes
    DATASET_KEY = f"{BASE_DATASET_KEY}_latent{latent_nodes}"
    result = run_experiment()
    test = result["test"]
    sweep_rows.append({
        "Experiment": "latent_capacity",
        "Dataset": BASE_DATASET_KEY,
        "LatentNodes": latent_nodes,
        "P": test["Precision"], "R": test["Recall"], "F1": test["F1"], "Acc": test["Accuracy"],
        "MacroF1": test["MacroF1"], "BalancedAccuracy": test["BalancedAccuracy"],
        "Threshold": test["Threshold"], "TestPairs": test["Pairs"],
        "BestValidF1": result["best_valid"]["F1"], "BestEpoch": result["best_epoch"],
        "RuntimeSeconds": result["seconds"], "RunProfile": RUN_PROFILE, "Seed": SEED,
        # A model that predicts "clone" for everything is not a weaker model, it
        # is a failed run; make that visible instead of averaging it in.
        "Collapsed": bool(test["Recall"] > 0.995 and test["Precision"] < 0.52),
    })
    pd.DataFrame(sweep_rows).to_csv(WORK_DIR / f"{BASE_DATASET_KEY}_latent_capacity_results.csv", index=False)

sweep = pd.DataFrame(sweep_rows)
display(sweep[["LatentNodes", "P", "R", "F1", "Acc", "RuntimeSeconds", "Collapsed"]])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(sweep.LatentNodes, sweep.F1, marker="o")
axes[0].set(title="Test F1 vs latent graph size", xlabel="latent nodes", ylabel="F1")
axes[1].plot(sweep.LatentNodes, sweep.RuntimeSeconds / 60.0, marker="o", color="tab:orange")
axes[1].set(title="Runtime vs latent graph size", xlabel="latent nodes", ylabel="minutes")
fig.tight_layout()
fig.savefig(WORK_DIR / f"{BASE_DATASET_KEY}_latent_capacity.png", dpi=160)
plt.show()
print(f"\\nSaved: {WORK_DIR / f'{BASE_DATASET_KEY}_latent_capacity_results.csv'}")
'''

ABLATION_DRIVER = '''
# Three inputs, increasing in how much lexical information they expose.
#
#   topology_only   canonical node type is replaced by a single constant, so the
#                   encoder sees pure graph structure and nothing else.
#   typed_topology  canonical categories (Control_If, Call_Expr, Literal_Num...)
#                   plus the bounded hashed label sketch. This is the canonical
#                   configuration used in the main results table.
#   source_lexical  typed topology plus the language-neutral hashed sketch of the
#                   raw source tokens.
ABLATION_VARIANTS = [
    ("topology_only", dict(strip_node_types=True, lexical_dropout=1.0, use_source_lexical=False)),
    ("typed_topology", dict(strip_node_types=False, lexical_dropout=LEXICAL_DROPOUT, use_source_lexical=False)),
    ("source_lexical", dict(strip_node_types=False, lexical_dropout=LEXICAL_DROPOUT, use_source_lexical=True)),
]

ablation_rows = []
for variant, settings in ABLATION_VARIANTS:
    print("\\n" + "=" * 70)
    print(f"VARIANT = {variant}   {settings}")
    print("=" * 70, flush=True)
    STRIP_NODE_TYPES = settings["strip_node_types"]
    LEXICAL_DROPOUT = settings["lexical_dropout"]
    USE_SOURCE_LEXICAL = settings["use_source_lexical"]
    DATASET_KEY = f"{BASE_DATASET_KEY}_{variant}"
    result = run_experiment()
    test = result["test"]
    ablation_rows.append({
        "Experiment": "feature_ablation",
        "Dataset": BASE_DATASET_KEY,
        "Variant": variant,
        "NodeTypes": not settings["strip_node_types"],
        "LexicalSketch": settings["lexical_dropout"] < 1.0,
        "SourceTokens": settings["use_source_lexical"],
        "P": test["Precision"], "R": test["Recall"], "F1": test["F1"], "Acc": test["Accuracy"],
        "MacroF1": test["MacroF1"], "BalancedAccuracy": test["BalancedAccuracy"],
        "Threshold": test["Threshold"], "TestPairs": test["Pairs"],
        "BestValidF1": result["best_valid"]["F1"], "RuntimeSeconds": result["seconds"],
        "RunProfile": RUN_PROFILE, "Seed": SEED,
        "Collapsed": bool(test["Recall"] > 0.995 and test["Precision"] < 0.52),
    })
    pd.DataFrame(ablation_rows).to_csv(WORK_DIR / f"{BASE_DATASET_KEY}_feature_ablation_results.csv", index=False)

ablation = pd.DataFrame(ablation_rows)
display(ablation[["Variant", "NodeTypes", "LexicalSketch", "SourceTokens", "P", "R", "F1", "Acc", "Collapsed"]])

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(ablation.Variant, ablation.F1, color=["tab:blue", "tab:green", "tab:orange"])
ax.set(title="How much of the score comes from each input", ylabel="Test F1")
ax.set_ylim(0, 1)
for index, value in enumerate(ablation.F1):
    ax.text(index, value + 0.02, f"{value:.3f}", ha="center")
fig.tight_layout()
fig.savefig(WORK_DIR / f"{BASE_DATASET_KEY}_feature_ablation.png", dpi=160)
plt.show()
print(f"\\nSaved: {WORK_DIR / f'{BASE_DATASET_KEY}_feature_ablation_results.csv'}")
'''

ABLATION_CONFIG = '''
# --- feature ablation controls -------------------------------------------
# STRIP_NODE_TYPES collapses every canonical category to one id, leaving the
# encoder with the adjacency structure only. The encoder reads it through
# CanonicalSpectraConfig below.
BASE_DATASET_KEY = DATASET_KEY
STRIP_NODE_TYPES = False

# The driver sets USE_SOURCE_LEXICAL per arm; the extra alignment loss must stay
# off in all of them, otherwise the "topology_only" arm is not topology only.
EMBEDDING_CONTRASTIVE_WEIGHT = 0.0
'''

SWEEP_CONFIG = '''
# --- latent capacity sweep controls --------------------------------------
BASE_DATASET_KEY = DATASET_KEY

# Latent capacity is a structural question, so the arms must differ only in the
# number of latent nodes. Stated here rather than inherited so a future change
# to the method's defaults cannot quietly make one benchmark lexical.
USE_SOURCE_LEXICAL = False
EMBEDDING_CONTRASTIVE_WEIGHT = 0.0
'''


def build_sweep(dataset: str, dataset_key: str) -> dict:
    template = _load_template()
    notebook = copy.deepcopy(template)
    cells = notebook["cells"]
    cells[0] = _markdown(
        f"# Latent capacity sweep - {dataset}\n\n"
        "Trains the canonical SPECTRA-Siam once per latent graph size (16, 24, 32, 48)\n"
        "with every other hyper-parameter fixed, so the difference in F1 is attributable\n"
        "to latent capacity alone. Node labels stay enabled; raw source tokens stay off.\n\n"
        f"Attach `{dataset}_clean_data.zip`.\n"
    )
    cells[CONFIG_CELL] = _code(_config_overrides(dataset_key, SWEEP_CONFIG))
    cells[DRIVER_CELL] = _code(SWEEP_DRIVER)
    return notebook


def build_ablation(dataset: str, dataset_key: str) -> dict:
    template = _load_template()
    notebook = copy.deepcopy(template)
    cells = notebook["cells"]
    cells[0] = _markdown(
        f"# Feature ablation - {dataset}\n\n"
        "Runs the same model three times on the same split, changing only what the\n"
        "encoder is allowed to see: bare topology, typed topology (the canonical\n"
        "setting), and typed topology plus hashed source tokens.\n\n"
        f"Attach `{dataset}_clean_data.zip`.\n"
    )
    cells[CONFIG_CELL] = _code(_config_overrides(dataset_key, ABLATION_CONFIG))
    cells[DRIVER_CELL] = _code(ABLATION_DRIVER)
    return notebook


TRANSFER_CONFIG = '''
# --- cross-dataset transfer controls -------------------------------------
# Train on one benchmark, evaluate zero-shot on the others. Attach every ZIP
# named below as a separate Kaggle input.
BASE_DATASET_KEY = DATASET_KEY
TRANSFER_SOURCE = "codexglue_v3"
TRANSFER_TARGETS = ["atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3"]

# Transfer is a claim about structure, so the source-token residual stays off on
# both sides; the graphs keep their node labels.
USE_SOURCE_LEXICAL = False
EMBEDDING_CONTRASTIVE_WEIGHT = 0.0

# Caps applied per corpus. The source needs enough pairs to train; each target
# only needs its test split.
TRANSFER_TRAIN_CAP = 50_000
TRANSFER_VALID_CAP = 10_000
TRANSFER_TEST_CAP = 10_000
'''

TRANSFER_DRIVER = '''
# Every benchmark numbers its codes from 1, so merging two corpora without
# namespacing would silently overwrite one dataset's graphs with the other's.
# Ids are tagged with their corpus before anything is merged.
def _slug(text):
    """Kaggle slugs vary in punctuation: gpt-clone-bench, gpt_clone_bench, gptclonebench."""
    return "".join(character for character in str(text).lower() if character.isalnum())


def resolve_named_clean_roots(names):
    """Map each requested dataset name to its attached clean-data directory.

    Attaching four bundles means one wrong match trains on one benchmark and
    reports the score as another, which no later check would catch. So the match
    is made on a punctuation-free slug, and an ambiguous match is an error rather
    than a silent pick of the shortest path.
    """
    found = {}
    candidates = set()
    for pattern in ("graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp"):
        for path in KAGGLE_INPUT.rglob(pattern):
            for parent in list(path.parents)[:4]:
                if is_complete_clean_data(parent):
                    candidates.add(parent)
    if not candidates:
        raise FileNotFoundError(
            "No complete clean-data bundle found under /kaggle/input. Attach each "
            "<dataset>_clean_data.zip as a separate input."
        )

    # is_complete_clean_data() searches recursively, so every ancestor of a real
    # bundle also looks complete. Keep only the innermost directory of each
    # nested chain: /a/codexglue and /a/codexglue/clean_data are one dataset,
    # not two, and treating them as two would look like an ambiguous match.
    candidates = {
        root for root in candidates
        if not any(other != root and root in other.parents for other in candidates)
    }

    listing = ", ".join(sorted(str(root) for root in candidates))
    for name in names:
        stem = _slug(name.replace("_v3", ""))
        # Match on the dataset folder, not the whole path: a shared parent
        # directory (e.g. /kaggle/input/datasets/<user>/) is not evidence.
        matches = [root for root in candidates if stem in _slug(root.parent.name) or stem in _slug(root.name)]
        if not matches:
            matches = [root for root in candidates if stem in _slug(root)]
        if not matches:
            raise FileNotFoundError(
                f"No attached clean-data bundle matches {name!r} (looking for {stem!r}). "
                f"Attached bundles: {listing}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"{name!r} matches {len(matches)} attached bundles: "
                + ", ".join(sorted(str(root) for root in matches))
                + ". Rename the Kaggle datasets so each name appears in exactly one path."
            )
        found[name] = matches[0]

    if len(set(found.values())) != len(found):
        raise RuntimeError(f"Two datasets resolved to the same folder: {found}")
    return found


def load_tagged_corpus(root, tag, splits, caps):
    """Load one corpus with every code id prefixed by its dataset tag."""
    corpus = CleanDataCorpus(root)
    frames = {}
    for offset, split in enumerate(splits):
        frame = corpus.pairs(split, maximum=caps.get(split), seed=SEED + offset).copy()
        frame["left_id"] = tag + ":" + frame["left_id"].astype(str)
        frame["right_id"] = tag + ":" + frame["right_id"].astype(str)
        frames[split] = frame

    wanted = set()
    for frame in frames.values():
        wanted |= set(frame.left_id) | set(frame.right_id)
    raw_ids = {value.split(":", 1)[1] for value in wanted}
    graphs = corpus.load_graphs(raw_ids, max_nodes=MAX_AST_NODES, strip_node_types=STRIP_NODE_TYPES)
    graphs = {f"{tag}:{code_id}": graph for code_id, graph in graphs.items()}

    for split, frame in frames.items():
        frames[split] = frame[frame.left_id.isin(graphs) & frame.right_id.isin(graphs)].reset_index(drop=True)
    languages = sorted({str(value) for frame in frames.values()
                        for value in list(frame.left_language) + list(frame.right_language)})
    print(f"  {tag:24s} " + ", ".join(f"{split}={len(frame):,}" for split, frame in frames.items())
          + f" | graphs={len(graphs):,} | languages={languages}")
    return frames, graphs


def run_transfer_experiment():
    set_seed(SEED)
    roots = resolve_named_clean_roots([TRANSFER_SOURCE, *TRANSFER_TARGETS])
    print("Attached bundles:")
    for name, root in roots.items():
        print(f"  {name:24s} {root}")

    print("\\nLoading corpora:")
    source_frames, graphs = load_tagged_corpus(
        roots[TRANSFER_SOURCE], TRANSFER_SOURCE, ("train", "valid", "test"),
        {"train": TRANSFER_TRAIN_CAP, "valid": TRANSFER_VALID_CAP, "test": TRANSFER_TEST_CAP},
    )
    target_frames = {}
    for name in TRANSFER_TARGETS:
        frames, target_graphs = load_tagged_corpus(
            roots[name], name, ("test",), {"test": TRANSFER_TEST_CAP}
        )
        target_frames[name] = frames["test"]
        graphs.update(target_graphs)

    for split in ("train", "valid"):
        if set(source_frames[split].label.astype(int).unique()) != {0, 1}:
            raise RuntimeError(f"source {split} split does not contain both classes")

    loaders = {
        "train": make_loader(source_frames["train"], graphs, max_nodes=MAX_AST_NODES, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda"),
        "valid": make_loader(source_frames["valid"], graphs, max_nodes=MAX_AST_NODES, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda"),
    }
    config = CanonicalSpectraConfig(
        hidden_dim=HIDDEN_DIM, latent_nodes=LATENT_NODES, slot_iterations=SLOT_ITERATIONS,
        attention_heads=ATTENTION_HEADS, lexical_dropout=LEXICAL_DROPOUT,
        use_source_lexical=USE_SOURCE_LEXICAL, target_density=TARGET_DENSITY,
        chebyshev_degree=CHEBYSHEV_DEGREE, relation_indices=tuple(INPUT_RELATION_INDICES),
    )
    model = CanonicalSpectraSiam(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    accumulation = max(1, math.ceil(EFFECTIVE_BATCH_SIZE / BATCH_SIZE))
    positives = int(source_frames["train"].label.sum())
    negatives = len(source_frames["train"]) - positives
    positive_weight = min(10.0, negatives / max(1, positives)) if POSITIVE_CLASS_WEIGHT == "auto" else float(POSITIVE_CLASS_WEIGHT)

    best = None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = seen = 0
        for index, (left, right, labels) in enumerate(tqdm(loaders["train"], desc=f"Epoch {epoch}/{EPOCHS}", leave=False), start=1):
            labels = labels.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                logits, auxiliary = model(left.to(DEVICE), right.to(DEVICE))
                loss, _ = canonical_spectra_loss(
                    logits, auxiliary, labels, positive_class_weight=positive_weight,
                    spectral_weight=SPECTRAL_WEIGHT, reconstruction_weight=RECONSTRUCTION_WEIGHT,
                    graph_weight=GRAPH_WEIGHT,
                )
            scaler.scale(loss / accumulation).backward()
            if index % accumulation == 0 or index == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * len(labels); seen += len(labels)

        valid_labels, valid_probabilities = predict(model, loaders["valid"], TEMPERATURE_END)
        threshold, valid = choose_threshold(valid_labels, valid_probabilities)
        print(f"epoch={epoch}/{EPOCHS} loss={total_loss / max(1, seen):.5f} "
              f"source_valid_F1={valid['F1']:.4f} threshold={threshold:.4f}")
        if best is None or valid["F1"] > best["valid"]["F1"]:
            best = {"epoch": epoch, "threshold": threshold, "valid": valid,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    source_threshold = best["threshold"]
    print(f"\\nBest source epoch {best['epoch']}, frozen threshold {source_threshold:.4f}")

    rows = []
    evaluations = [(TRANSFER_SOURCE, source_frames["test"], "in-domain")] + \\
                  [(name, frame, "zero-shot") for name, frame in target_frames.items()]
    for name, frame, kind in evaluations:
        if not len(frame):
            print(f"[skip] {name}: no usable test pairs"); continue
        loader = make_loader(frame, graphs, max_nodes=MAX_AST_NODES, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
        labels, probabilities = predict(model, loader, TEMPERATURE_END)
        # The frozen source threshold is the honest transfer number. The oracle
        # threshold is reported beside it as the ceiling a calibrated model could
        # reach, never as the transfer result itself.
        transferred = metric_summary(labels, probabilities, source_threshold)
        _, oracle = choose_threshold(labels, probabilities)
        rows.append({
            "Experiment": "cross_dataset_transfer", "TrainedOn": TRANSFER_SOURCE,
            "EvaluatedOn": name, "Setting": kind,
            "P": transferred["Precision"], "R": transferred["Recall"], "F1": transferred["F1"],
            "Acc": transferred["Accuracy"], "MacroF1": transferred["MacroF1"],
            "BalancedAccuracy": transferred["BalancedAccuracy"],
            "Threshold": source_threshold, "TestPairs": transferred["Pairs"],
            "OracleF1": oracle["F1"], "OracleThreshold": oracle["Threshold"],
            "SourceValidF1": best["valid"]["F1"], "BestEpoch": best["epoch"],
            "RuntimeSeconds": time.perf_counter() - started, "RunProfile": RUN_PROFILE, "Seed": SEED,
            "Collapsed": bool(transferred["Recall"] > 0.995 and transferred["Precision"] < 0.52),
        })
        print(f"  {name:24s} {kind:10s} F1={transferred['F1']:.4f} "
              f"(oracle {oracle['F1']:.4f}) collapsed={rows[-1]['Collapsed']}")

    table = pd.DataFrame(rows)
    out_path = WORK_DIR / f"transfer_from_{TRANSFER_SOURCE}_results.csv"
    table.to_csv(out_path, index=False)
    display(table[["EvaluatedOn", "Setting", "P", "R", "F1", "Acc", "OracleF1", "Collapsed"]])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:blue" if row.Setting == "in-domain" else "tab:orange" for row in table.itertuples()]
    ax.bar(table.EvaluatedOn, table.F1, color=colors)
    ax.axhline(0.667, ls="--", c="grey", lw=1)
    ax.text(0.01, 0.672, "all-positive collapse", fontsize=8, color="grey", transform=ax.get_yaxis_transform())
    ax.set(title=f"Trained on {TRANSFER_SOURCE}, evaluated zero-shot", ylabel="Test F1", ylim=(0, 1))
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(WORK_DIR / f"transfer_from_{TRANSFER_SOURCE}.png", dpi=160)
    plt.show()
    print(f"\\nSaved: {out_path}")
    return table


TRANSFER_RESULTS = run_transfer_experiment()
'''


CROSS_LANGUAGE_CONFIG = '''
# --- cross-language transfer controls ------------------------------------
BASE_DATASET_KEY = DATASET_KEY

# Train on one language, evaluate on every language, for each language in turn:
# the result is a full source x target transfer matrix rather than one row.
CROSS_LANGUAGE_SOURCES = ["java", "python", "c", "csharp"]

# AST only, deliberately. DDG is exported natively only for the Joern languages
# (Java, C); for Python and C# the graphs come from local tree-sitter/ast
# builders whose node ids do not align with Joern's, so the projected DDG
# relation is empty for them. Keeping DDG on would hand Java and C an extra
# relation the other two do not have, which is exactly the asymmetry a
# cross-language comparison must not contain.
INPUT_RELATION_INDICES = (0,)

# Structure is the claim being tested, so raw source tokens stay off. Node
# labels stay on, as in the main results table.
USE_SOURCE_LEXICAL = False
EMBEDDING_CONTRASTIVE_WEIGHT = 0.0

# Per-language splits are small (1.4k-4k train pairs), so the run is short but
# also data-starved; that is a property of the benchmark, not a bug.
CROSS_LANGUAGE_TRAIN_CAP = None
CROSS_LANGUAGE_VALID_CAP = None
CROSS_LANGUAGE_TEST_CAP = None
'''

CROSS_LANGUAGE_DRIVER = '''
def language_splits(corpus, language, caps):
    """Mono-language pairs of one language, per split."""
    frames = {}
    for offset, split in enumerate(("train", "valid", "test")):
        frames[split] = corpus.pairs(
            split,
            language_filter=PairLanguageFilter.mono(language),
            maximum=caps[split],
            seed=SEED + offset,
        )
    return frames


def run_cross_language_experiment():
    corpus = CleanDataCorpus(CLEAN_DATA_DIR)
    caps = {"train": get_cap(CROSS_LANGUAGE_TRAIN_CAP),
            "valid": get_cap(CROSS_LANGUAGE_VALID_CAP),
            "test": get_cap(CROSS_LANGUAGE_TEST_CAP)}

    available = sorted(set(corpus.code_languages.values()))
    languages = [language for language in CROSS_LANGUAGE_SOURCES if language in available]
    missing = [language for language in CROSS_LANGUAGE_SOURCES if language not in available]
    if missing:
        print(f"[!] not present in this bundle, skipped: {missing}")
    if len(languages) < 2:
        raise RuntimeError(f"Cross-language transfer needs at least two languages; found {languages}.")
    print(f"languages: {languages}")

    per_language = {}
    for language in languages:
        frames = language_splits(corpus, language, caps)
        code_ids = pair_code_ids(*frames.values())
        graphs = corpus.load_graphs(code_ids, max_nodes=MAX_AST_NODES, strip_node_types=STRIP_NODE_TYPES)
        for split, frame in frames.items():
            frames[split] = frame[frame.left_id.astype(str).isin(graphs)
                                  & frame.right_id.astype(str).isin(graphs)].reset_index(drop=True)
        per_language[language] = (frames, graphs)
        counts = ", ".join(f"{split}={len(frame):,}" for split, frame in frames.items())
        print(f"  {language:8s} {counts} | graphs={len(graphs):,}")

    rows = []
    for source in languages:
        source_frames, source_graphs = per_language[source]
        if set(source_frames["train"].label.astype(int).unique()) != {0, 1}:
            print(f"[skip] {source}: train split has a single class"); continue
        if not len(source_frames["valid"]):
            print(f"[skip] {source}: no validation pairs"); continue

        print("\\n" + "=" * 70)
        print(f"TRAIN ON {source.upper()}")
        print("=" * 70, flush=True)
        set_seed(SEED)

        train_loader = make_loader(source_frames["train"], source_graphs, max_nodes=MAX_AST_NODES,
                                   batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                   pin_memory=DEVICE.type == "cuda")
        valid_loader = make_loader(source_frames["valid"], source_graphs, max_nodes=MAX_AST_NODES,
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                   pin_memory=DEVICE.type == "cuda")
        config = CanonicalSpectraConfig(
            hidden_dim=HIDDEN_DIM, latent_nodes=LATENT_NODES, slot_iterations=SLOT_ITERATIONS,
            attention_heads=ATTENTION_HEADS, lexical_dropout=LEXICAL_DROPOUT,
            use_source_lexical=USE_SOURCE_LEXICAL, target_density=TARGET_DENSITY,
            chebyshev_degree=CHEBYSHEV_DEGREE, relation_indices=tuple(INPUT_RELATION_INDICES),
        )
        model = CanonicalSpectraSiam(config).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        accumulation = max(1, math.ceil(EFFECTIVE_BATCH_SIZE / BATCH_SIZE))
        positives = int(source_frames["train"].label.sum())
        negatives = len(source_frames["train"]) - positives
        positive_weight = (min(10.0, negatives / max(1, positives))
                           if POSITIVE_CLASS_WEIGHT == "auto" else float(POSITIVE_CLASS_WEIGHT))

        best = None
        started = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            total_loss = seen = 0
            for index, (left, right, labels) in enumerate(
                    tqdm(train_loader, desc=f"{source} epoch {epoch}/{EPOCHS}", leave=False), start=1):
                labels = labels.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                    logits, auxiliary = model(left.to(DEVICE), right.to(DEVICE))
                    loss, _ = canonical_spectra_loss(
                        logits, auxiliary, labels, positive_class_weight=positive_weight,
                        spectral_weight=SPECTRAL_WEIGHT, reconstruction_weight=RECONSTRUCTION_WEIGHT,
                        graph_weight=GRAPH_WEIGHT)
                scaler.scale(loss / accumulation).backward()
                if index % accumulation == 0 or index == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach().cpu()) * len(labels); seen += len(labels)

            valid_labels, valid_probabilities = predict(model, valid_loader, TEMPERATURE_END)
            threshold, valid = choose_threshold(valid_labels, valid_probabilities)
            print(f"  epoch={epoch}/{EPOCHS} loss={total_loss / max(1, seen):.5f} "
                  f"valid_F1={valid['F1']:.4f} threshold={threshold:.4f}")
            if best is None or valid["F1"] > best["valid"]["F1"]:
                best = {"epoch": epoch, "threshold": threshold, "valid": valid,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        model.load_state_dict(best["state"])
        elapsed = time.perf_counter() - started

        for target in languages:
            target_frames, target_graphs = per_language[target]
            test_frame = target_frames["test"]
            if not len(test_frame):
                print(f"  [skip] {target}: no test pairs"); continue
            loader = make_loader(test_frame, target_graphs, max_nodes=MAX_AST_NODES,
                                 batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=DEVICE.type == "cuda")
            labels, probabilities = predict(model, loader, TEMPERATURE_END)
            transferred = metric_summary(labels, probabilities, best["threshold"])
            _, oracle = choose_threshold(labels, probabilities)
            rows.append({
                "Experiment": "cross_language_transfer", "Dataset": BASE_DATASET_KEY,
                "TrainedOn": source, "EvaluatedOn": target,
                "Setting": "in-language" if source == target else "zero-shot",
                "P": transferred["Precision"], "R": transferred["Recall"], "F1": transferred["F1"],
                "Acc": transferred["Accuracy"], "MacroF1": transferred["MacroF1"],
                "BalancedAccuracy": transferred["BalancedAccuracy"],
                "Threshold": best["threshold"], "TestPairs": transferred["Pairs"],
                "OracleF1": oracle["F1"], "OracleThreshold": oracle["Threshold"],
                "SourceValidF1": best["valid"]["F1"], "BestEpoch": best["epoch"],
                "TrainPairs": int(len(source_frames["train"])),
                "RuntimeSeconds": elapsed, "RunProfile": RUN_PROFILE, "Seed": SEED,
                "Collapsed": bool(transferred["Recall"] > 0.995 and transferred["Precision"] < 0.52),
            })
            print(f"  -> {target:8s} F1={transferred['F1']:.4f} (oracle {oracle['F1']:.4f}) "
                  f"collapsed={rows[-1]['Collapsed']}")
        pd.DataFrame(rows).to_csv(WORK_DIR / f"{BASE_DATASET_KEY}_cross_language_results.csv", index=False)

    table = pd.DataFrame(rows)
    display(table[["TrainedOn", "EvaluatedOn", "Setting", "P", "R", "F1", "OracleF1", "Collapsed"]])

    matrix = table.pivot(index="TrainedOn", columns="EvaluatedOn", values="F1")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(xlabel="evaluated on", ylabel="trained on",
           title=f"Cross-language transfer F1 - {BASE_DATASET_KEY}")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.to_numpy()[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.6 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{BASE_DATASET_KEY}_cross_language.png", dpi=160)
    plt.show()
    print(f"\\nSaved: {WORK_DIR / f'{BASE_DATASET_KEY}_cross_language_results.csv'}")
    return table


CROSS_LANGUAGE_RESULTS = run_cross_language_experiment()
'''


def build_cross_language(dataset: str, dataset_key: str) -> dict:
    template = _load_template()
    notebook = copy.deepcopy(template)
    cells = notebook["cells"]
    cells[0] = _markdown(
        f"# Cross-language transfer - {dataset}\n\n"
        "Trains one model per language on that language's pairs only, then evaluates it on\n"
        "every language's test split. The result is a full source x target matrix: the\n"
        "diagonal is in-language performance, everything off it is zero-shot transfer.\n\n"
        "Runs on **AST only**. DDG is exported natively for Java and C but not for Python and\n"
        "C#, whose graphs come from local builders with a different node-id space, so leaving\n"
        "DDG on would give two of the four languages an extra relation - an asymmetry that\n"
        "would show up as a language effect it is not.\n\n"
        f"Attach `{dataset}_clean_data.zip`.\n"
    )
    cells[CONFIG_CELL] = _code(_config_overrides(dataset_key, CROSS_LANGUAGE_CONFIG))
    cells[DRIVER_CELL] = _code(CROSS_LANGUAGE_DRIVER)
    return notebook


def build_transfer(dataset: str, dataset_key: str) -> dict:
    template = _load_template()
    notebook = copy.deepcopy(template)
    cells = notebook["cells"]
    cells[0] = _markdown(
        "# Cross-dataset transfer - train on CodeXGLUE, test everywhere else\n\n"
        "Trains one SPECTRA-Siam on the CodeXGLUE (BigCloneBench) train split, freezes the\n"
        "threshold chosen on its validation split, and evaluates that single model zero-shot\n"
        "on the test split of every other benchmark.\n\n"
        "Because each benchmark numbers its codes from 1, code ids are namespaced by dataset\n"
        "before the corpora are merged - without that the graphs of one dataset would silently\n"
        "overwrite the other's.\n\n"
        "**Attach all four ZIPs** as separate Kaggle inputs:\n"
        "`codexglue_v3_clean_data.zip`, `atcoder_v3_clean_data.zip`,\n"
        "`gptclonebench_v3_clean_data.zip`, `semanticclonebench_v3_clean_data.zip`.\n\n"
        "`F1 = 0.667` with `R = 1.0` means the model predicts \"clone\" for every pair; the\n"
        "`Collapsed` column flags exactly that.\n"
    )
    cells[CONFIG_CELL] = _code(_config_overrides(dataset_key, TRANSFER_CONFIG))
    cells[DRIVER_CELL] = _code(TRANSFER_DRIVER)
    return notebook


DATASETS = {
    "codexglue_v3": "codexglue",
    "atcoder_v3": "at-coder",
    "gptclonebench_v3": "gpt-clone-bench",
    "semanticclonebench_v3": "semantic-clone-bench",
}

BUILDERS = {
    "01_latent_capacity": build_sweep,
    "04_feature_ablation": build_ablation,
}
# Only the benchmarks that actually hold several languages in mono-language
# pairs. CodeXGLUE is Java-only and AtCoder's pairs are all Java<->Python, so
# neither can supply a train-on-one-language / test-on-another split.
CROSS_LANGUAGE_DATASETS = ("semanticclonebench_v3", "gptclonebench_v3")
# One notebook only: the source corpus is fixed to CodeXGLUE and the targets are
# the other three, so a per-dataset copy would just be the same file four times.
SINGLETON_BUILDERS = {
    "02_cross_dataset": ("train_on_codexglue.ipynb", build_transfer, "codexglue_v3"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    args = parser.parse_args()

    stale = 0
    for folder, builder in BUILDERS.items():
        target_dir = EXPERIMENTS_ROOT / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            notebook = builder(dataset, DATASETS[dataset])
            path = target_dir / f"{dataset}.ipynb"
            rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current == rendered:
                status = "up to date"
            else:
                stale += 1
                status = "written" if not args.check else "STALE"
                if not args.check:
                    path.write_text(rendered, encoding="utf-8")
            print(f"  {folder:22s} {path.name:26s} {status}")

    target_dir = EXPERIMENTS_ROOT / "03_cross_language"
    target_dir.mkdir(parents=True, exist_ok=True)
    for dataset in CROSS_LANGUAGE_DATASETS:
        if dataset not in args.datasets:
            continue
        path = target_dir / f"{dataset}.ipynb"
        rendered = json.dumps(build_cross_language(dataset, DATASETS[dataset]), ensure_ascii=False, indent=1)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == rendered:
            status = "up to date"
        else:
            stale += 1
            status = "written" if not args.check else "STALE"
            if not args.check:
                path.write_text(rendered, encoding="utf-8")
        print(f"  {'03_cross_language':22s} {path.name:26s} {status}")

    for folder, (filename, builder, dataset) in SINGLETON_BUILDERS.items():
        target_dir = EXPERIMENTS_ROOT / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        rendered = json.dumps(builder(dataset, DATASETS[dataset]), ensure_ascii=False, indent=1)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == rendered:
            status = "up to date"
        else:
            stale += 1
            status = "written" if not args.check else "STALE"
            if not args.check:
                path.write_text(rendered, encoding="utf-8")
        print(f"  {folder:22s} {path.name:26s} {status}")

    if args.check and stale:
        print(f"\n[-] {stale} notebook(s) out of date; run scripts/build_experiment_notebooks.py")
        return 1
    print(f"\n[+] experiment notebooks {'checked' if args.check else 'generated'} under {EXPERIMENTS_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
