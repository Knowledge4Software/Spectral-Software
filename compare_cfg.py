import textwrap
import numpy as np
from spectral_code.pipeline import Pipeline
from spectral_code.config import PipelineConfig


def pad_vectors(v1, v2):
    max_len = max(len(v1), len(v2))

    v1_padded = np.pad(v1, (0, max_len - len(v1)))
    v2_padded = np.pad(v2, (0, max_len - len(v2)))

    return v1_padded, v2_padded


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )


code1 = textwrap.dedent("""
x = 5
if x > 0:
    y = x
else:
    y = -x
""")

code2 = textwrap.dedent("""
x = 5
if x <= 0:
    y = -x
else:
    y = x
""")

config = PipelineConfig(
    graph_type="cfg",
    eigen_solver="dense",
    k_eigen=None,
)

pipeline = Pipeline(config)

res1 = pipeline.run(code1)
res2 = pipeline.run(code2)

eig1 = np.sort(res1["eigenvalues"])
eig2 = np.sort(res2["eigenvalues"])

eig1, eig2 = pad_vectors(eig1, eig2)

print("Code 1 Eigenvalues:")
print(eig1)

print("\nCode 2 Eigenvalues:")
print(eig2)

print("\nEuclidean Distance:")
print(np.linalg.norm(eig1 - eig2))

print("\nCosine Similarity:")
print(cosine_similarity(eig1, eig2))