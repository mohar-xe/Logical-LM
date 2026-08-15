"""Deterministic, offline LLM client for tests and demos.

``MockClient`` maps an exact prompt (or a substring of it) to a canned
response.  This is how the whole pipeline is exercised without a network or
an API key: the few-shot prompt templates are filled with the *same* problem
text they will be tested against, and the mock returns the hand-authored
logic program for that problem.

Use ``MockClient({prompt_fragment: response, ...})`` — keys are matched by
substring.  Responses are returned in key-declaration order when several keys
match (longest match wins).
"""

from __future__ import annotations

from .base import LLMClient


class MockClient(LLMClient):
    def __init__(self, responses: dict[str, str], default: str = ""):
        self.responses = dict(responses)
        self.default = default
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        # longest matching key wins; falls back to default
        best, best_len = self.default, -1
        for key, resp in self.responses.items():
            if key in prompt and len(key) > best_len:
                best, best_len = resp, len(key)
        return best

    def name(self) -> str:
        return "mock"
