"""Construct an LLM client from a backend name + options."""

from __future__ import annotations

import os

from .base import LLMClient
from .mock import MockClient


def build_llm_client(
    backend: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    mock_responses: dict[str, str] | None = None,
) -> LLMClient:
    backend = backend.lower()
    if backend in {"mock", "offline"}:
        return MockClient(mock_responses or {})
    if backend in {"openai", "gpt"}:
        from .openai import OpenAIClient

        return OpenAIClient(
            api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=model or "gpt-4o-mini",
            base_url=base_url,
        )
    if backend in {"anthropic", "claude"}:
        from .anthropic import AnthropicClient

        return AnthropicClient(
            api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            model=model or "claude-sonnet-4-5",
        )
    if backend in {"ollama", "local"}:
        from .ollama import OllamaClient

        return OllamaClient(model=model or "llama3", base_url=base_url or "http://localhost:11434/v1")
    raise ValueError(f"unknown LLM backend {backend!r}")
