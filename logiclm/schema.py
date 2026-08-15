"""Common data schema shared by datasets, pipeline and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Example:
    id: str
    context: str
    question: str
    answer: str = ""            # ground-truth letter (A/B/C/D/E)
    options: list[str] = field(default_factory=list)
    # raw LLM-generated logic program(s); exactly one in practice
    programs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "options": self.options,
            "raw_logic_programs": self.programs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(
            id=d["id"],
            context=d.get("context", ""),
            question=d.get("question", ""),
            answer=d.get("answer", ""),
            options=d.get("options", []),
            programs=d.get("raw_logic_programs", []),
        )
