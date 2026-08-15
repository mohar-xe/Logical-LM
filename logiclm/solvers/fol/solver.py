"""FOLIO solver: prove the conclusion from the premises with Z3.

Semantics replicate Prover9's prove/refute/unknown three-way split:

1. prove conclusion C under premises P  -> ``True``  (answer A)
2. else prove ``~C`` under P            -> ``False`` (answer B)
3. else                                 -> ``Unknown`` (answer C)

A Z3 ``check()`` returning ``unknown`` (hard formula / timeout) must not be
silently reported as provable or refuted; see ``strict_unknown``.
"""

from __future__ import annotations

import z3

from .ast import Negation
from .compile import Compiler
from .program import FOLProgram
from ..errors import ExecutionError, ParseError, TimeoutError


class FOLSolver:
    def __init__(self, timeout_ms: int = 10_000, strict_unknown: bool = False):
        self.timeout_ms = timeout_ms
        self.strict_unknown = strict_unknown

    def solve(self, program: str) -> tuple[str, str]:
        """Return ``(verdict, error_message)``; verdict is one of
        ``True`` / ``False`` / ``Unknown``."""
        parsed = FOLProgram.parse(program)
        if parsed.conclusion is None:
            raise ParseError("logic program has no conclusion")
        return self._solve_parsed(parsed)

    def _solve_parsed(self, program: FOLProgram) -> tuple[str, str]:
        compiler = Compiler()
        premises = [compiler.compile_formula(p) for p in program.premises]
        conclusion = compiler.compile_formula(program.conclusion)
        pre = z3.And(*premises) if premises else z3.BoolVal(True)

        # 1. prove C: P && ~C unsat?
        with z3.Solver() as s:
            s.set("timeout", self.timeout_ms)
            s.add(pre)
            s.add(z3.Not(conclusion))
            res = s.check()
        if res == z3.unsat:
            return "True", ""
        if res == z3.unknown:
            return self._unknown_result()

        # 2. refute C: P && C unsat?
        with z3.Solver() as s:
            s.set("timeout", self.timeout_ms)
            s.add(pre)
            s.add(conclusion)
            res = s.check()
        if res == z3.unsat:
            return "False", ""
        if res == z3.unknown:
            return self._unknown_result()

        # 3. neither provable nor refutable
        return "Unknown", ""

    def _unknown_result(self) -> tuple[str, str]:
        if self.strict_unknown:
            raise ExecutionError(
                "solver returned 'unknown' (hard formula / timeout); "
                "treating as an execution error under --strict-unknown"
            )
        return "Unknown", ""

    @staticmethod
    def answer_mapping(verdict: str) -> str:
        """Map a solver verdict to the FOLIO answer letter (True->A ...)."""
        mapping = {"True": "A", "False": "B", "Unknown": "C"}
        try:
            return mapping[verdict]
        except KeyError:
            raise ValueError(f"unrecognized FOL verdict {verdict!r}")
