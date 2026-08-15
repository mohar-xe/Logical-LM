"""End-to-end pipeline: generation -> inference -> (refine) -> evaluation.

Stage 1 (generate): the LLM translates each problem into a logic program.
Stage 2 (infer):    the symbolic solver executes each program; failures fall
                    back to the configured backup strategy.
Stage 3 (refine):   repeated rounds where the solver's error messages drive
                    LLM revisions of failed programs (optional, <= max_rounds).
Stage 4 (evaluate): compute accuracy / executable-rate / exec-accuracy.

The module is dataset-agnostic: it takes ``Example`` objects and produces a
list of result dicts (JSON-serialisable) matching the reference format.
"""

from __future__ import annotations

import os

from .backup import BackupAnswerGenerator
from .llm.base import LLMClient
from .prompts import PromptLibrary
from .schema import Example
from .solvers.registry import SolverRegistry


class LogicLMPipeline:
    def __init__(
        self,
        dataset: str,
        llm: LLMClient,
        solver: SolverRegistry,
        backup_strategy: str = "random",
        backup_llm_path: str | None = None,
        prompts: PromptLibrary | None = None,
        seed: int = 0,
        max_refine_rounds: int = 0,
    ):
        self.dataset = dataset
        self.llm = llm
        self.solver = solver
        self.prompts = prompts or PromptLibrary()
        self.backup_strategy = backup_strategy
        self.backup_llm_path = backup_llm_path
        self.max_refine_rounds = max_refine_rounds
        import random
        self.rng = random.Random(seed)
        self.backup = BackupAnswerGenerator(
            dataset, backup_strategy, backup_llm_path, rng=self.rng
        )

    # -- stage 1: generation ---------------------------------------------

    def generate_programs(self, examples: list[Example]) -> list[Example]:
        """Fill each example's ``programs`` from the LLM."""
        for ex in examples:
            prompt = self.prompts.build_generation_prompt(self.dataset, ex.to_dict())
            program = self.llm.generate(prompt).strip()
            ex.programs = [program]
        return examples

    # -- stage 2: inference ----------------------------------------------

    def infer(self, examples: list[Example]) -> list[dict]:
        """Run the solver on each example's program; fall back on failure."""
        results = []
        for ex in examples:
            program = ex.programs[0] if ex.programs else ""
            res = self.solver.solve(self.dataset, program)
            if res.status != "success":
                predicted = self.backup.get_backup_answer(ex.id)
                results.append({
                    "id": ex.id,
                    "context": ex.context,
                    "question": ex.question,
                    "answer": ex.answer,
                    "options": ex.options,
                    "flag": res.status,
                    "error": res.error,
                    "predicted_answer": predicted,
                    "program": program,
                })
            else:
                results.append({
                    "id": ex.id,
                    "context": ex.context,
                    "question": ex.question,
                    "answer": ex.answer,
                    "options": ex.options,
                    "flag": "success",
                    "error": "",
                    "predicted_answer": res.answer,
                    "program": program,
                })
        return results

    # -- stage 3: self-refinement ----------------------------------------

    def refine(self, examples: list[Example]) -> list[Example]:
        """Up to ``max_refine_rounds`` rounds of solver-error-driven revision.

        Delegates to ``logiclm.refine.refine_programs`` (the shared contract).
        """
        from .refine import refine_programs

        refine_programs(
            examples,
            dataset=self.dataset,
            llm=self.llm,
            solver=self.solver,
            max_rounds=self.max_refine_rounds,
            prompts=self.prompts,
        )
        return examples

    # -- stage 4: run everything -----------------------------------------

    def run(self, examples: list[Example]) -> list[dict]:
        self.generate_programs(examples)
        if self.max_refine_rounds > 0:
            self.refine(examples)
        return self.infer(examples)

    @staticmethod
    def save(results: list[dict], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
