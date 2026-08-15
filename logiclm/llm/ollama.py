"""Local Ollama adapter (OpenAI-compatible ``/v1/chat/completions``)."""

from __future__ import annotations

from .base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def name(self) -> str:
        return self.model
