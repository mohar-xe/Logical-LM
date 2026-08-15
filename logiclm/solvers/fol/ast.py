"""AST for first-order-logic formulas (FOLIO IR).

A tiny, immutable expression tree.  Variables, constants and predicates are
all plain strings; ``compile.py`` resolves them against the example's
predicate list and produces Z3 expressions.
"""

from __future__ import annotations

from dataclasses import dataclass


class Formula:
    """Base class for all FOL AST nodes."""

    __slots__ = ()


@dataclass(frozen=True)
class Variable(Formula):
    name: str


@dataclass(frozen=True)
class Constant(Formula):
    name: str


@dataclass(frozen=True)
class Predicate(Formula):
    name: str
    args: tuple[Formula, ...]


@dataclass(frozen=True)
class Quantifier(Formula):
    qname: str  # "forall" | "exists"
    var: str
    body: Formula


@dataclass(frozen=True)
class Binary(Formula):
    op: str  # "and" | "or" | "->" | "<->" | "xor"
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Negation(Formula):
    body: Formula


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_variables(formula: Formula) -> set[str]:
    """Collect the free (unbound) variables of a formula."""
    free: set[str] = set()

    def walk(f: Formula):
        nonlocal free
        if isinstance(f, Variable):
            free.add(f.name)
        elif isinstance(f, Quantifier):
            walk(f.body)
            free.discard(f.var)
        elif isinstance(f, Binary):
            walk(f.left)
            walk(f.right)
        elif isinstance(f, Negation):
            walk(f.body)
        elif isinstance(f, Predicate):
            for arg in f.args:
                if isinstance(arg, Variable):
                    free.add(arg.name)

    walk(formula)
    return free


def bound_variables(formula: Formula) -> set[str]:
    """Collect every variable name bound by a quantifier (for compile-time
    resolution of Variable nodes)."""
    bound: set[str] = set()

    def walk(f: Formula):
        nonlocal bound
        if isinstance(f, Quantifier):
            bound.add(f.var)
            walk(f.body)
        elif isinstance(f, Binary):
            walk(f.left)
            walk(f.right)
        elif isinstance(f, Negation):
            walk(f.body)
        elif isinstance(f, Predicate):
            pass

    walk(formula)
    return bound


def to_infix(formula: Formula) -> str:
    """Render the AST back to FOLIO-style infix text (used by tests + debug)."""
    if isinstance(formula, Variable):
        return formula.name
    if isinstance(formula, Constant):
        return formula.name
    if isinstance(formula, Predicate):
        return f"{formula.name}({', '.join(to_infix(a) for a in formula.args)})"
    if isinstance(formula, Negation):
        return f"~{_parenthesize(formula.body)}"
    if isinstance(formula, Quantifier):
        sym = "∀" if formula.qname == "forall" else "∃"
        return f"{sym}{formula.var} {to_infix(formula.body)}"
    if isinstance(formula, Binary):
        symbols = {"and": "∧", "or": "∨",
                   "->": "->", "<->": "<->", "xor": "⊕"}
        return (f"{_parenthesize(formula.left)} {symbols[formula.op]} "
                f"{_parenthesize(formula.right)}")
    raise TypeError(f"unknown formula node {formula!r}")


def _parenthesize(f: Formula) -> str:
    if isinstance(f, (Predicate, Variable, Constant, Negation, Quantifier)):
        return to_infix(f)
    return f"({to_infix(f)})"
