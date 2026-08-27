from __future__ import annotations


class RagError(Exception):
    pass


class ConfigError(RagError):
    pass


class GenerationError(RagError):
    pass


class RetrievalError(RagError):
    pass
