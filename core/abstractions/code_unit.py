class CodeUnit:
    def __init__(self, source: str, language: str, metadata: dict = None):
        self.source = source
        self.language = language
        self.metadata = metadata or {}