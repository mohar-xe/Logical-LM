"""Self-refinement loop: solver error messages drive LLM revisions.

The pipeline's ``refine`` method implements this inline; this module exposes
the same contract as a standalone function so tests and the CLI can drive a
single round at a time and inspect the loop bookkeeping (how many programs
were revised, how many rounds were needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm.base import LLMClient
from .prompts import PromptLibrary
from .schema import Example
from .solvers.registry import SolverRegistry


@dataclass
class RefineReport:
    rounds_used: int = 0
    revised: list[str] = field(default_factory=list)  # example ids revised
    calls_per_example: dict[str, int] = field(default_factory=dict)


def refine_programs(
    examples: list[Example],
    dataset: str,
    llm: LLMClient,
    solver: SolverRegistry,
    max_rounds: int = 3,
    prompts: PromptLibrary | None = None,
) -> RefineReport:
    """Revise failed programs for at most ``max_rounds`` rounds.

    Successful examples carry over untouched and are never re-calls to the
    LLM; the loop early-exits when no example fails anymore.  Returns a report
    of what happened.
    """
    prompts = prompts or PromptLibrary()
    report = RefineReport()

    for round_no in range(1, max_rounds + 1):
        report.rounds_used = round_no
        failed: list[Example] = []
        for ex in examples:
            program = ex.programs[0] if ex.programs else ""
            res = solver.solve(dataset, program)
            if res.status != "success":
                failed.append(ex)

        if not failed:
            report.rounds_used = round_no - 1  # last round was a no-op
            break

        for ex in failed:
            program = ex.programs[0]
            res = solver.solve(dataset, program)
            error = res.error or "Parsing Error"
            prompt = prompts.build_self_correct_prompt(dataset, program, error)
            revised = llm.generate(prompt).strip()
            report.calls_per_example[ex.id] = report.calls_per_example.get(ex.id, 0) + 1
            if revised:
                ex.programs = [revised]
                report.revised.append(ex.id)

    return report
