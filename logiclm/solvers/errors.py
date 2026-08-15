"""Structured error types for the symbolic solver layer.

Solvers raise ``SolverError`` (with a short, human-readable ``message`` for
the self-refinement prompt) when a program cannot be parsed or executed.  The
pipeline catches these and falls back to the configured backup strategy.
"""

from __future__ import annotations


class SolverError(Exception):
    """Base for all solver-layer failures. ``message`` is safe to show an LLM."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ParseError(SolverError):
    """The logic program text could not be parsed (bad syntax / structure)."""


class ExecutionError(SolverError):
    """The program parsed but could not be executed (unsat, timeout, ...)."""


class CompileError(SolverError):
    """A parsed program could not be lowered to the target backend (Z3)."""


class TimeoutError(SolverError):
    """The solver exceeded its time budget."""

    def __init__(self, message: str = "solver timed out"):
        super().__init__(message)
