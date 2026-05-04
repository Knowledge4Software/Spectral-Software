from core.interfaces.spectrum import SpectrumComputer


class SpectrumFactory:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, cls: type[SpectrumComputer]):
        self._registry[name] = cls

    def get(self, name: str) -> SpectrumComputer:
        if name not in self._registry:
            raise ValueError(f"Unknown spectrum method: {name}")
        return self._registry[name]()