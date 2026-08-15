"""Solver for the AR-LSAT Z3-DSL.

Given a parsed + compiled program:
1. Build the precondition ``P`` = conjunction of all constraints.
2. For each option, decide by its wrapper semantics:
   * ``is_valid(E)``: option holds iff ``P ∧ ¬E`` is unsat (entailment).
   * ``is_sat(E)``:   option holds iff ``P ∧ E``   is sat (consistency).
   Options in the raw text are wrapped with ``is_valid``/``is_sat``; the
   wrapper is stripped during parsing, so we re-derive it from the raw line.
3. Return the single letter whose option held.  If zero or more than one hold
   (ambiguous program), the caller falls back.

The reference solves each option in a fresh subprocess against a Python script
that prints the letter when ``solver.check() == unsat``; here we run all
checks in-process against the same ``pre_conditions``.
"""

from __future__ import annotations

import re
import z3

from ..errors import ExecutionError, TimeoutError
from .compile import DSLCompiler
from .parser import Z3DSLProgram


def _option_wrapper(raw_line: str) -> str:
    """Return the wrapper ('is_valid'|'is_sat') of an option's raw text."""
    m = re.search(r"\b(is_valid|is_sat)\s*\(", raw_line)
    return m.group(1) if m else "is_valid"


def solve_dsl(raw: str, timeout_ms: int = 10_000) -> tuple[str, str]:
    """Return (answer_letter, error_message); answer letter is '' on failure."""
    program = Z3DSLProgram.parse(raw)
    compiler = DSLCompiler(program)

    pre = [compiler.lower(c) for c in program.constraints]
    pre_and = z3.And(*pre) if len(pre) > 1 else (pre[0] if pre else z3.BoolVal(True))

    # Build a small table of option -> raw line for wrapper detection.
    # The parser stripped wrappers; re-derive from the raw options section.
    raw_option_lines = _raw_option_lines(raw)

    winners: list[str] = []
    for choice, formula in program.options:
        if not choice:
            continue
        expr = compiler.lower(formula)
        wrapper = _option_wrapper(raw_option_lines.get(choice, ""))
        with z3.Solver() as s:
            s.set("timeout", timeout_ms)
            if wrapper == "is_valid":
                s.add(pre_and)
                s.add(z3.Not(expr))
                res = s.check()
                holds = res == z3.unsat
            else:
                s.add(pre_and)
                s.add(expr)
                res = s.check()
                holds = res == z3.sat
            if res == z3.unknown:
                raise ExecutionError(
                    "z3 returned 'unknown' for an option check (hard formula / timeout)"
                )
        if holds:
            winners.append(choice)

    if len(winners) == 1:
        return winners[0], ""
    if not winners:
        return "", "no option was entailed/satisfiable"
    return "", f"ambiguous: multiple options held: {sorted(winners)}"


def _raw_option_lines(raw: str) -> dict[str, str]:
    """Map choice label -> the raw option line (before ::: comment strip)."""
    mapping: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("Question"):
            continue
        if ":::" not in stripped:
            continue
        comment = stripped.split(":::", 1)[1].strip()
        m = re.match(r"\(([A-E])\)", comment)
        if m:
            mapping[m.group(1)] = stripped
    return mapping
