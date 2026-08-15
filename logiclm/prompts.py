"""Prompt-template loading and filling.

Templates live in ``<project>/prompts/*.txt`` (one per dataset, plus
``self-correct-<dataset>.txt``).  They use ``[[PROBLEM]]`` / ``[[QUESTION]]`` /
``[[CHOICES]]`` placeholders for generation and ``[[PROGRAM]]`` /
``[[ERROR MESSAGE]]`` for self-correction.
"""

from __future__ import annotations

import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

_DATASETS = ("ProntoQA", "ProofWriter", "FOLIO", "LogicalDeduction", "AR-LSAT")


class PromptLibrary:
    def __init__(self, prompts_dir: str = _PROMPTS_DIR):
        self.dir = prompts_dir

    def generation_template(self, dataset: str) -> str:
        if dataset not in _DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}")
        path = os.path.join(self.dir, f"{dataset}.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def self_correct_template(self, dataset: str) -> str:
        path = os.path.join(self.dir, f"self-correct-{dataset}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no self-correction template for {dataset} at {path}. "
                "Self-refinement templates ship only for FOLIO and AR-LSAT "
                "(matching the paper); other datasets run without refinement."
            )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # -- prompt assembly --------------------------------------------------

    def build_generation_prompt(self, dataset: str, example: dict) -> str:
        """Fill a generation template with a problem/query/choices."""
        template = self.generation_template(dataset)
        prompt = template.replace("[[PROBLEM]]", example["context"])
        prompt = prompt.replace("[[QUESTION]]", example["question"])
        choices = example.get("options") or []
        choices_str = "\n".join(str(c) for c in choices)
        prompt = prompt.replace("[[CHOICES]]", choices_str)
        return prompt

    def build_self_correct_prompt(self, dataset: str, program: str, error: str) -> str:
        """Fill a self-correction template with the program + error message."""
        template = self.self_correct_template(dataset)
        return (
            template.replace("[[PROGRAM]]", program)
            .replace("[[ERROR MESSAGE]]", error)
        )
