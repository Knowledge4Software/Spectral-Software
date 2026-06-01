from spectral_code.utils.errors import unknown_component_error
from spectral_code.spectral.solvers.dense import DenseEigenSolver
from spectral_code.spectral.solvers.sparse import SparseEigenSolver
from spectral_code.spectral.solvers.auto import AutoEigenSolver

SOLVER_REGISTRY = {
    "dense": DenseEigenSolver,
    "sparse": SparseEigenSolver,
    "auto": AutoEigenSolver,
}


def create_solver(name: str, k: int, **kwargs):
    if name not in SOLVER_REGISTRY:
        raise unknown_component_error("solver", name, SOLVER_REGISTRY)

    cls = SOLVER_REGISTRY[name]

    if name in {"sparse", "auto"}:
        return cls(k=k, **kwargs)

    return cls(**kwargs)