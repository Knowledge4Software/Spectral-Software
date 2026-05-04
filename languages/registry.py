from languages.python.python_language import PythonLanguage
from languages.java.java_language import JavaLanguage


class LanguageRegistry:
    _registry = {
        "python": PythonLanguage,
        "java": JavaLanguage,
    }

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Unknown language: {name}")
        return cls._registry[name]()