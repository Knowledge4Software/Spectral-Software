from spectral_code.utils.errors import unknown_component_error
from spectral_code.preprocessing.simple import SimplePreprocessor

PREPROCESSOR_REGISTRY = {
    "simple": SimplePreprocessor,
    None: lambda **kwargs: None,
}


def create_preprocessor(name: str, **kwargs):
    if name not in PREPROCESSOR_REGISTRY:
        raise unknown_component_error("preprocessor", name, PREPROCESSOR_REGISTRY)

    cls = PREPROCESSOR_REGISTRY[name]
    return cls(**kwargs) if callable(cls) else cls