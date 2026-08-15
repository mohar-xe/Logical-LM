"""Evaluation metrics over a list of per-example results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    total: int = 0
    correct: int = 0
    executable: int = 0
    executable_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def executable_rate(self) -> float:
        return self.executable / self.total if self.total else 0.0

    @property
    def executable_accuracy(self) -> float:
        return self.executable_correct / self.executable if self.executable else 0.0

    def __str__(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f} ({self.correct}/{self.total})  "
            f"executable={self.executable_rate:.3f} ({self.executable}/{self.total})  "
            f"exec-acc={self.executable_accuracy:.3f} ({self.executable_correct}/{self.executable})"
        )


def compute_metrics(results: list[dict]) -> Metrics:
    m = Metrics(total=len(results))
    for r in results:
        predicted = r.get("predicted_answer")
        gold = r.get("answer", "")
        if not predicted:
            continue
        if predicted == gold:
            m.correct += 1
        if r.get("flag") == "success":
            m.executable += 1
            if predicted == gold:
                m.executable_correct += 1
    return m
