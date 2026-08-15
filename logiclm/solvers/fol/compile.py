"""Compile a parsed FOL formula AST into Z3 expressions.

Soundness notes (see plan):
* Every example gets a **fresh uninterpreted sort** ``U``; nothing is shared
  between examples.  This is sound for FOLIO because conclusions never
  quantify over objects not named in the premises — satisfiability of
  ``P ∧ ¬C`` is exactly a model-based counterexample to entailment.
* Predicates compile to uninterpreted functions ``name: U... -> Bool``.
* Names in *term position* compile to a Z3 constant of sort ``U``; a name
  bound by an enclosing quantifier becomes the quantified variable.
* A single ``Compiler`` caches its predicate functions and ground constants
  so the same name maps to the same Z3 symbol throughout one program.
"""

from __future__ import annotations

import z3

from .ast import Binary, Formula, Negation, Predicate, Quantifier, Variable
from ..errors import CompileError

Bool = z3.BoolSort()
U = z3.DeclareSort("U")


class Compiler:
    def __init__(self) -> None:
        self._predicates: dict[tuple[str, int], z3.FuncDeclRef] = {}
        self._constants: dict[str, z3.ExprRef] = {}

    # -- public -----------------------------------------------------------

    def compile_formula(self, formula: Formula) -> z3.BoolRef:
        """Compile a top-level formula (premise or conclusion) to Z3.

        Free variables are *not* allowed here: FOLIO programs are closed, so
        any unbound name in term position is treated as a ground constant.
        """
        return self._compile(formula, bound={})

    # -- internals --------------------------------------------------------

    def _compile(self, formula: Formula, bound: dict[str, z3.ExprRef]) -> z3.ExprRef:
        if isinstance(formula, Quantifier):
            var = z3.Const(formula.var, U)
            inner_bound = dict(bound)
            inner_bound[formula.var] = var
            body = self._compile(formula.body, inner_bound)
            if formula.qname == "forall":
                return z3.ForAll([var], body)
            return z3.Exists([var], body)

        if isinstance(formula, Negation):
            return z3.Not(self._compile(formula.body, bound))

        if isinstance(formula, Binary):
            left = self._compile(formula.left, bound)
            right = self._compile(formula.right, bound)
            op = formula.op
            if op == "and":
                return z3.And(left, right)
            if op == "or":
                return z3.Or(left, right)
            if op == "->":
                return z3.Implies(left, right)
            if op == "<->":
                return z3.Iff(left, right)
            if op == "xor":
                return z3.Xor(left, right)
            raise CompileError(f"unknown binary operator {op!r}")

        if isinstance(formula, Predicate):
            return self._predicate_app(formula, bound)

        if isinstance(formula, Variable):
            if formula.name in bound:
                return bound[formula.name]
            return self._constant(formula.name)

        raise CompileError(f"cannot compile node {formula!r}")

    def _predicate_app(self, formula: Predicate, bound: dict[str, z3.ExprRef]) -> z3.BoolRef:
        func = self._predicate(formula.name, len(formula.args))
        args = [self._compile(arg, bound) for arg in formula.args]
        return func(*args)

    def _predicate(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = (name, arity)
        if key not in self._predicates:
            self._predicates[key] = z3.Function(name, *([U] * arity), Bool)
        return self._predicates[key]

    def _constant(self, name: str) -> z3.ExprRef:
        if name not in self._constants:
            self._constants[name] = z3.Const(name, U)
        return self._constants[name]
