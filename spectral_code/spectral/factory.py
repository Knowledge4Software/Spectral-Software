from spectral_code.utils.errors import unknown_component_error
from spectral_code.spectral.base import SpectralAnalyzer
from spectral_code.spectral.laplacian import LaplacianSpectrum
from spectral_code.spectral.normalized_laplacian import NormalizedLaplacianSpectrum
from spectral_code.spectral.solvers.factory import create_solver

class DummySpectralAnalyzer(SpectralAnalyzer):
    def analyze(self, graph):
        return None, None

SPECTRAL_REGISTRY = {
    "laplacian": LaplacianSpectrum,
    "normalized_laplacian": NormalizedLaplacianSpectrum,
    "none": DummySpectralAnalyzer
}


def create_spectral_analyzer(mode: str, solver_name: str, k: int,
                             solver_kwargs: dict, spectral_kwargs: dict):

    if mode is None or str(mode).lower() == "none":
        return DummySpectralAnalyzer()

    if mode not in SPECTRAL_REGISTRY:
        raise unknown_component_error("spectral mode", mode, SPECTRAL_REGISTRY)

    solver = create_solver(solver_name, k, **solver_kwargs)

    cls = SPECTRAL_REGISTRY[mode]
    return cls(solver=solver, **spectral_kwargs)