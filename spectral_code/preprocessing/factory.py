from spectral_code.utils.errors import unknown_component_error
# Fixed the import name to match simple.py exactly
from spectral_code.preprocessing.simple import SimpleGraphPreprocessor

PREPROCESSOR_REGISTRY = {
    "simple": SimpleGraphPreprocessor,
    None: lambda **kwargs: None,
}


def create_preprocessor(name: str, **kwargs):
    if name not in PREPROCESSOR_REGISTRY:
        raise unknown_component_error("preprocessor", name, PREPROCESSOR_REGISTRY)

    cls = PREPROCESSOR_REGISTRY[name]
    return cls(**kwargs) if callable(cls) else cls