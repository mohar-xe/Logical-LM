"""LLM client protocol and shared helpers.

The pipeline talks to LLMs only through ``LLMClient``.  Adapters for OpenAI,
Anthropic and local Ollama servers live beside it; ``MockClient`` (in
``mock.py``) lets the whole pipeline run offline with canned programs.
"""

from __future__ import annotations

import abc


class LLMClient(abc.ABC):
    """Generate text from prompts. Implementations must be deterministic at
    ``temperature=0`` by default."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Return a single completion for ``prompt``."""

    def generate_many(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt (sequential by default)."""
        return [self.generate(p) for p in prompts]

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging/output filenames."""
