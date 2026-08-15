"""Per-dataset solver dispatch.

Each solver returns a ``SolveResult`` carrying the predicted answer letter
(already mapped from the symbolic verdict), a status, and an error message to
feed self-refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .csp import CSPProgram
from .datalog import DatalogProgram
from .errors import SolverError
from .fol.solver import FOLSolver
from .z3dsl.solver import solve_dsl

# statuses returned to the pipeline
STATUS_OK = "success"
STATUS_PARSE = "parsing error"
STATUS_EXEC = "execution error"


@dataclass
class SolveResult:
    answer: str | None
    status: str
    error: str = ""


class SolverRegistry:
    def __init__(self, timeout_ms: int = 10_000, strict_unknown: bool = False):
        self.timeout_ms = timeout_ms
        self.strict_unknown = strict_unknown

    def solve(self, dataset: str, program_text: str) -> SolveResult:
        try:
            if dataset in {"ProntoQA", "ProofWriter"}:
                return self._solve_datalog(dataset, program_text)
            if dataset == "FOLIO":
                return self._solve_folio(program_text)
            if dataset == "LogicalDeduction":
                return self._solve_csp(program_text)
            if dataset == "AR-LSAT":
                return self._solve_arlsat(program_text)
            raise ValueError(f"unsupported dataset {dataset!r}")
        except SolverError as e:
            # classify structured errors
            from .errors import ParseError
            if isinstance(e, ParseError):
                return SolveResult(None, STATUS_PARSE, str(e))
            return SolveResult(None, STATUS_EXEC, str(e))

    # -- individual backends ---------------------------------------------

    def _solve_datalog(self, dataset: str, program_text: str) -> SolveResult:
        prog = DatalogProgram(program_text, dataset=dataset)
        answer = prog.answer()
        return SolveResult(answer, STATUS_OK)

    def _solve_folio(self, program_text: str) -> SolveResult:
        solver = FOLSolver(timeout_ms=self.timeout_ms, strict_unknown=self.strict_unknown)
        verdict, err = solver.solve(program_text)
        if verdict in {"True", "False", "Unknown"}:
            return SolveResult(FOLSolver.answer_mapping(verdict), STATUS_OK)
        return SolveResult(None, STATUS_EXEC, err)

    def _solve_csp(self, program_text: str) -> SolveResult:
        prog = CSPProgram(program_text)
        solutions = prog.solve()
        letter = prog.answer_mapping(solutions)
        if letter is None:
            return SolveResult(None, STATUS_EXEC, "no option entailed by all solutions")
        return SolveResult(letter, STATUS_OK)

    def _solve_arlsat(self, program_text: str) -> SolveResult:
        letter, err = solve_dsl(program_text, timeout_ms=self.timeout_ms)
        if letter:
            return SolveResult(letter, STATUS_OK)
        return SolveResult(None, STATUS_EXEC, err)
