"""Backup answer strategies for when the symbolic solver fails.

Two strategies, matching the paper:
* ``random`` — uniform random guess from the dataset's answer-letter space.
* ``LLM``   — use a precomputed chain-of-thought prediction for the example
  (loaded from a results file keyed by example id).
"""

from __future__ import annotations

import json
import random

from . import ANSWER_SPACE


class BackupAnswerGenerator:
    def __init__(self, dataset: str, strategy: str, llm_result_path: str | None = None,
                 rng: random.Random | None = None):
        if dataset not in ANSWER_SPACE:
            raise ValueError(f"unknown dataset {dataset!r}")
        self.dataset = dataset
        self.strategy = strategy
        self.rng = rng or random.Random()
        self.llm_backup: dict[str, str] = {}
        if strategy == "LLM":
            if not llm_result_path:
                raise ValueError("LLM backup strategy requires llm_result_path")
            with open(llm_result_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    self.llm_backup[item["id"]] = item.get("predicted_answer", "")

    def get_backup_answer(self, example_id: str) -> str:
        if self.strategy == "random":
            return self.rng.choice(ANSWER_SPACE[self.dataset])
        if self.strategy == "LLM":
            return self.llm_backup.get(example_id, self.rng.choice(ANSWER_SPACE[self.dataset]))
        raise ValueError(f"unknown backup strategy {self.strategy!r}")
