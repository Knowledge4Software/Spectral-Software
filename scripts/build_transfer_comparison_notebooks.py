"""Materialize Kaggle notebooks for the selected transfer-comparison methods."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
CROSS_DATASET = KAGGLE / "experiments" / "02_cross_dataset"
CROSS_LANGUAGE = KAGGLE / "experiments" / "03_cross_language"
METHODS = ("astnn", "rtvnn", "deepsim")


# The transfer-bundle builder embeds this manifest in every bundle.  Retaining
# it in the result avoids accidentally treating a target test score as another
# ordinary CodeXGLUE run when the CSVs are collected after Kaggle execution.
CROSS_DATASET_PROVENANCE = """
# Persist the source/target declared by the attached transfer bundle.
bundle_manifests = list(Path("/kaggle/input").rglob("metadata.json"))
transfer_manifests = []
for manifest_path in bundle_manifests:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        continue
    if manifest.get("bundle") == "cross_dataset_transfer":
        transfer_manifests.append((manifest_path, manifest))
if len(transfer_manifests) != 1:
    raise RuntimeError(
        "Attach exactly one cross-dataset transfer bundle. "
        f"Found {len(transfer_manifests)} matching metadata.json files."
    )

manifest_path, transfer_manifest = transfer_manifests[0]
source_dataset = transfer_manifest["train_valid_from"]
target_dataset = transfer_manifest["test_from"]
baseline_csv = Path("/kaggle/working") / f"{RUN_LABEL}_combined_dataset_results.csv"
annotated_results = pd.read_csv(baseline_csv)
annotated_results.insert(0, "Experiment", "cross_dataset_transfer")
annotated_results.insert(1, "TrainedOn", source_dataset)
annotated_results.insert(2, "TestedOn", target_dataset)
annotated_path = Path("/kaggle/working") / (
    f"{RUN_LABEL}_{source_dataset}_to_{target_dataset}_results.csv"
)
annotated_results.to_csv(annotated_path, index=False)
display(annotated_results)
print("Transfer manifest:", manifest_path)
print("Annotated transfer results:", annotated_path)
"""


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.splitlines()]


def make_cross_dataset_notebooks() -> list[Path]:
    """The transfer bundle makes stock baseline notebooks zero-shot runners."""
    outputs: list[Path] = []
    spectra = read_notebook(CROSS_DATASET / "train_on_codexglue.ipynb")
    spectra["cells"][0]["source"] = [
        "# Cross-dataset transfer — SPECTRA-Siam\n",
        "\n",
        "Train on CodeXGLUE and evaluate zero-shot on ATCoder, GPTCloneBench, and "
        "SemanticCloneBench. Attach all four clean-data ZIPs; source lexical features remain disabled.\n",
    ]
    path = CROSS_DATASET / "01_spectra_siam_train_on_codexglue.ipynb"
    write_notebook(path, spectra)
    outputs.append(path)

    for index, method in enumerate(METHODS, start=2):
        base = read_notebook(KAGGLE / "codexglue_v3" / "baselines" / f"{method}_baseline.ipynb")
        config = source(base["cells"][0])
        config = config.replace('RUN_PROFILE = "final_full"', 'RUN_PROFILE = "comparison_50k"')
        config = config.replace(
            f'RUN_LABEL = "{method}_baseline"',
            f'RUN_LABEL = "cross_dataset_{method}"',
        )
        set_source(base["cells"][0], config)
        base["cells"][2]["source"] = [
            f"# Cross-dataset transfer — {method.upper()}\n",
            "\n",
            "Attach exactly one transfer bundle built by scripts/build_transfer_bundles.py, "
            "then run all cells. The bundle supplies CodeXGLUE train/valid and the named "
            "target test split. Run this notebook once per target dataset.\n",
        ]
        base["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in CROSS_DATASET_PROVENANCE.splitlines()],
        })
        path = CROSS_DATASET / f"0{index}_{method}_transfer_bundle.ipynb"
        write_notebook(path, base)
        outputs.append(path)
    return outputs


LANGUAGE_PRELUDE = """
    # --- zero-shot cross-language split -------------------------------------
    language_path = find_file("codes.jsonl.gz", "codes.jsonl", "codes.jsonl.gz.tmp")
    code_languages = {}
    with open_text(language_path) as _language_handle:
        for _language_line in _language_handle:
            if _language_line.strip():
                _language_row = json.loads(_language_line)
                code_languages[str(_language_row.get("code_id"))] = str(_language_row.get("language", "")).lower()

    def _mono_language(frame: pd.DataFrame, language: str) -> pd.DataFrame:
        left = frame["left_id"].astype(str).map(code_languages)
        right = frame["right_id"].astype(str).map(code_languages)
        return frame[(left == language) & (right == language)].reset_index(drop=True)
"""

LANGUAGE_FILTER = """
    if train_language is not None:
        train_df = _mono_language(train_df, train_language)
        valid_df = _mono_language(valid_df, train_language)
    if test_language is not None:
        test_df = _mono_language(test_df, test_language)
    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(
            f"Empty language split: train={train_language!r}, test={test_language!r}; "
            f"counts={len(train_df)}/{len(valid_df)}/{len(test_df)}"
        )
    print(f"cross-language train={train_language}, test={test_language}: "
          f"{len(train_df):,}/{len(valid_df):,}/{len(test_df):,}")
"""

LANGUAGE_EPILOGUE = """
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd

BASE_DATASET_KEY = DATASET_KEYS[0]
CROSS_LANGUAGE_SOURCES = ["java", "python", "c", "csharp"]
all_language_rows = []
for source_language in CROSS_LANGUAGE_SOURCES:
    for target_language in CROSS_LANGUAGE_SOURCES:
        print("\\n" + "=" * 96)
        print(f"{METHOD_NAME}: train {source_language} -> test {target_language}")
        print("=" * 96)
        result = run_one_dataset(BASE_DATASET_KEY, source_language, target_language)
        result.insert(0, "Experiment", "cross_language_transfer")
        result.insert(1, "TrainedOnLanguage", source_language)
        result.insert(2, "TestLanguage", target_language)
        all_language_rows.append(result)

cross_language_results = pd.concat(all_language_rows, ignore_index=True)
result_path = Path("/kaggle/working") / f"{RUN_LABEL}_semanticclonebench_v3_cross_language_results.csv"
cross_language_results.to_csv(result_path, index=False)
display(cross_language_results)

matrix = cross_language_results.pivot(index="TrainedOnLanguage", columns="TestLanguage", values="F1")
matrix = matrix.reindex(index=CROSS_LANGUAGE_SOURCES, columns=CROSS_LANGUAGE_SOURCES)
fig, axis = plt.subplots(figsize=(7, 5))
image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
axis.set_xticks(range(len(CROSS_LANGUAGE_SOURCES)), CROSS_LANGUAGE_SOURCES)
axis.set_yticks(range(len(CROSS_LANGUAGE_SOURCES)), CROSS_LANGUAGE_SOURCES)
axis.set(xlabel="Test language", ylabel="Training language", title=f"Zero-shot transfer F1 — {METHOD_NAME}")
for row in range(len(CROSS_LANGUAGE_SOURCES)):
    for column in range(len(CROSS_LANGUAGE_SOURCES)):
        value = matrix.iloc[row, column]
        axis.text(column, row, "-" if pd.isna(value) else f"{value:.3f}",
                  ha="center", va="center", color="white" if pd.notna(value) and value < .55 else "black", fontsize=9)
fig.colorbar(image, ax=axis, label="Test F1")
fig.tight_layout()
figure_path = Path("/kaggle/working") / f"{RUN_LABEL}_semanticclonebench_v3_cross_language_f1.png"
fig.savefig(figure_path, dpi=180)
plt.show()
print("Saved:", result_path)
print("Saved:", figure_path)
"""


def patch_cross_language_baseline(notebook: dict, method: str) -> dict:
    notebook = copy.deepcopy(notebook)
    config = source(notebook["cells"][0])
    config = config.replace('RUN_PROFILE = "final_full"', 'RUN_PROFILE = "comparison_50k"')
    config = config.replace(
        f'RUN_LABEL = "{method}_baseline"',
        f'RUN_LABEL = "cross_language_{method}"',
    )
    set_source(notebook["cells"][0], config)
    notebook["cells"][2]["source"] = [
        f"# Zero-shot cross-language transfer — {method.upper()}\n",
        "\n",
        "For each source language (Java, Python, C, C#), train on source-language "
        "train/valid pairs only, freeze that source threshold, then test every target language. "
        "Attach semanticclonebench_v3_clean_data.zip.\n",
    ]
    body = source(notebook["cells"][3])
    body = body.replace(
        "def run_one_dataset(dataset_key: str):",
        "def run_one_dataset(dataset_key: str, train_language: str | None = None, test_language: str | None = None):",
        1,
    )
    pair_marker = "    pairs_df = load_pairs(pairs_path)\n"
    if pair_marker not in body:
        raise RuntimeError(f"Pair-loading marker missing in {method}.")
    body = body.replace(pair_marker, pair_marker + LANGUAGE_PRELUDE, 1)
    split_marker = '    print(f"using train/valid/test={len(train_df):,}/{len(valid_df):,}/{len(test_df):,}")\n'
    if split_marker not in body:
        raise RuntimeError(f"Split marker missing in {method}.")
    body = body.replace(split_marker, split_marker + LANGUAGE_FILTER, 1)
    outer_marker = "from IPython.display import display\nimport pandas as pd\n\nall_dataset_results = {}"
    start = body.find(outer_marker)
    if start < 0:
        raise RuntimeError(f"Outer runner missing in {method}.")
    display_name = {"astnn": "ASTNN", "rtvnn": "RtvNN", "deepsim": "DeepSim"}[method]
    epilogue = f'DISPLAY_METHOD = "{display_name}"\n' + LANGUAGE_EPILOGUE.replace("METHOD_NAME", "DISPLAY_METHOD")
    set_source(notebook["cells"][3], body[:start] + epilogue)
    return notebook


def make_cross_language_notebooks() -> list[Path]:
    outputs: list[Path] = []
    spectra = read_notebook(CROSS_LANGUAGE / "semanticclonebench_v3.ipynb")
    spectra["cells"][0]["source"] = [
        "# Zero-shot cross-language transfer — SPECTRA-Siam\n",
        "\n",
        "Canonical AST+DDG SPECTRA-Siam experiment. Source lexical features are disabled; "
        "each source-language model uses a frozen source validation threshold on every target language.\n",
    ]
    path = CROSS_LANGUAGE / "01_spectra_siam_semanticclonebench_v3.ipynb"
    write_notebook(path, spectra)
    outputs.append(path)
    for index, method in enumerate(METHODS, start=2):
        base = read_notebook(KAGGLE / "semanticclonebench_v3" / "baselines" / f"{method}_baseline.ipynb")
        path = CROSS_LANGUAGE / f"0{index}_{method}_semanticclonebench_v3.ipynb"
        write_notebook(path, patch_cross_language_baseline(base, method))
        outputs.append(path)
    return outputs


def main() -> None:
    for heading, paths in (
        ("Cross-dataset notebooks", make_cross_dataset_notebooks()),
        ("Cross-language notebooks", make_cross_language_notebooks()),
    ):
        print(heading + ":")
        print(*[f"  - {path.relative_to(ROOT)}" for path in paths], sep="\n")


if __name__ == "__main__":
    main()
