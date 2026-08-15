"""Compile the AR-LSAT DSL AST to Z3 expressions.

Semantics
---------
* ``EnumSort([Vladimir, Wendy])`` -> a fresh Z3 datatype (or UninterpretedSort
  with Const members).  We use an UninterpretedSort per sort plus one Const
  per member, so members are distinct and equality/arithmetic work as
  expected.  Numeric-member enum sorts (``lockers = EnumSort([1,2,3,4,5])``)
  become Z3 ``Int`` constants so arithmetic like ``l - 1`` works.
* ``Function([people, meals] -> [foods])`` -> ``z3.Function`` over the sorts.
* ``ForAll``/``Exists`` bind scope variables as Z3 ``Const`` of the scope's
  sort; ``Count`` becomes ``Sum(If(cond, 1, 0) ...)``; ``Distinct`` becomes
  a chain of pairwise inequalities.
* ``is_valid(E)`` (entailment) is ``unsat(P ∧ ¬E)``; ``is_sat(E)``
  (consistency) is ``sat(P ∧ E)``.  Both are computed in the solver, not the
  compiler; the compiler only lowers the formula.

The compiler is deliberately free of ``exec``/string manipulation: every node
maps to a Z3 object directly.
"""

from __future__ import annotations

import z3

from ..errors import CompileError
from .ast import (
    FApp, FArith, FBoolOp, FCmp, FConst, FCount, FDistinct, FQuant, FVar,
    FunctionDecl,
)
from .parser import Z3DSLProgram

_BOOLOP_MAP = {
    "And": lambda args: z3.And(*args),
    "Or": lambda args: z3.Or(*args),
    "Not": lambda args: z3.Not(args[0]),
    "Implies": lambda args: z3.Implies(args[0], args[1]),
    "Iff": lambda args: z3.Iff(args[0], args[1]),
    "Xor": lambda args: z3.Xor(*args),
    "True": lambda args: z3.BoolVal(True),
    "False": lambda args: z3.BoolVal(False),
}

_COMPARE_MAP = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}

_ARITH_MAP = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


class DSLCompiler:
    _instance_counter = 0

    def __init__(self, program: Z3DSLProgram):
        self.program = program
        self.sort_refs: dict[str, z3.SortRef] = {}
        self.member_consts: dict[str, dict[str, z3.ExprRef]] = {}
        self.func_refs: dict[str, z3.FuncDeclRef] = {}
        # z3's EnumSort registers sort/member names in a global symbol table;
        # two compilers (two examples, or two runs in one process) must never
        # collide, so every sort gets a per-instance-unique suffix.
        DSLCompiler._instance_counter += 1
        self._uid = f"_{DSLCompiler._instance_counter}"
        self._setup()

    # -- sorts ------------------------------------------------------------

    def _setup(self) -> None:
        # Build sorts first so member constants can reference them.
        for name, members in self.program.enum_sorts.items():
            # EnumSort gives the members real distinctness semantics (they are
            # constructors of a datatype), which DeclareSort + bare Consts
            # would not — z3 could model two different Consts as equal.
            # The per-instance suffix keeps global z3 symbol names unique.
            sort_ref, consts = z3.EnumSort(f"{name}_sort{self._uid}", list(members))
            self.sort_refs[name] = sort_ref
            self.member_consts[name] = dict(zip(members, consts))
        for name, members in self.program.int_sorts.items():
            self.sort_refs[name] = z3.IntSort()
            self.member_consts[name] = {m: z3.IntVal(int(m)) for m in members}

        for fname, decl in self.program.functions.items():
            arg_sorts = [self._sort_of(s) for s in decl.arg_sorts]
            result_sort = self._sort_of(decl.result_sort)
            self.func_refs[fname] = z3.Function(f"{fname}{self._uid}", *arg_sorts, result_sort)

    def _sort_of(self, name: str) -> z3.SortRef:
        if name == "bool":
            return z3.BoolSort()
        if name == "int":
            return z3.IntSort()
        if name in self.sort_refs:
            return self.sort_refs[name]
        raise CompileError(f"unknown sort {name!r}")

    # -- formula lowering ---------------------------------------------------

    def lower(self, formula) -> z3.ExprRef:
        return self._lower(formula, bound={})

    def _lower(self, formula, bound: dict[str, z3.ExprRef]) -> z3.ExprRef:
        if isinstance(formula, FConst):
            return self._resolve_const(formula.name)
        if isinstance(formula, FVar):
            if formula.name in bound:
                return bound[formula.name]
            return self._resolve_const(formula.name)
        if isinstance(formula, FApp):
            return self._lower_app(formula, bound)
        if isinstance(formula, FQuant):
            return self._lower_quant(formula, bound)
        if isinstance(formula, FBoolOp):
            op = formula.op
            if op not in _BOOLOP_MAP:
                raise CompileError(f"unknown boolean operator {op!r}")
            args = [self._lower(a, bound) for a in formula.args]
            return _BOOLOP_MAP[op](args)
        if isinstance(formula, FCmp):
            if formula.op not in _COMPARE_MAP:
                raise CompileError(f"unknown comparison {formula.op!r}")
            return _COMPARE_MAP[formula.op](
                self._lower(formula.left, bound),
                self._lower(formula.right, bound),
            )
        if isinstance(formula, FArith):
            if formula.op not in _ARITH_MAP:
                raise CompileError(f"unknown arithmetic op {formula.op!r}")
            return _ARITH_MAP[formula.op](
                self._lower(formula.left, bound),
                self._lower(formula.right, bound),
            )
        if isinstance(formula, FCount):
            return self._lower_count(formula, bound)
        if isinstance(formula, FDistinct):
            return self._lower_distinct(formula, bound)
        raise CompileError(f"cannot lower formula node {formula!r}")

    def _resolve_const(self, name: str) -> z3.ExprRef:
        # A bare name in term position: a domain member of some sort, or a
        # ground constant (search all sorts for a matching member).
        for sort_name, members in self.member_consts.items():
            if name in members:
                return members[name]
        # numeric literal?
        try:
            return z3.IntVal(int(name))
        except ValueError:
            pass
        # unknown ground constant — default to the first domain sort
        if self.sort_refs:
            first_sort = next(iter(self.sort_refs.values()))
            return z3.Const(name, first_sort)
        raise CompileError(f"unresolved constant {name!r}")

    def _lower_app(self, app: FApp, bound: dict[str, z3.ExprRef]) -> z3.ExprRef:
        decl = self.program.functions.get(app.func)
        if decl is None:
            # a predicate-style relation like `eats(x) == y`? treat as bool fn
            raise CompileError(f"unknown function {app.func!r}")
        args = [self._lower(a, bound) for a in app.args]
        func = self.func_refs.get(app.func)
        if func is None:
            raise CompileError(f"function not compiled: {app.func!r}")
        return func(*args)

    def _lower_quant(self, quant: FQuant, bound: dict[str, z3.ExprRef]) -> z3.ExprRef:
        inner_bound = dict(bound)
        vars_: list[z3.ExprRef] = []
        for var, scope in quant.vars:
            zvar = z3.Const(var, self._sort_of(scope))
            vars_.append(zvar)
            inner_bound[var] = zvar
        body = self._lower(quant.body, inner_bound)
        if quant.qname == "ForAll":
            return z3.ForAll(vars_, body)
        return z3.Exists(vars_, body)

    def _lower_count(self, count: FCount, bound: dict[str, z3.ExprRef]) -> z3.ExprRef:
        # Sum over the finite domain members: Sum(If(cond, 1, 0) for ...).
        # For each member, bind the scope variable to that member's constant
        # and lower the condition under that binding.
        member_lists = [
            (var, scope, self._member_terms(scope))
            for var, scope in count.vars
        ]
        if not member_lists:
            raise CompileError("Count requires at least one scope variable")

        # Cartesian product of member constants across all scope vars.
        terms: list[z3.ExprRef] = []
        import itertools

        combinations = itertools.product(*(ml[2] for ml in member_lists))
        for combo in combinations:
            inner_bound = dict(bound)
            for (var, _, _), member_expr in zip(member_lists, combo):
                inner_bound[var] = member_expr
            cond = self._lower(count.cond, inner_bound)
            terms.append(z3.If(cond, z3.IntVal(1), z3.IntVal(0)))
        return z3.Sum(terms)

    def _member_terms(self, scope: str) -> list[z3.ExprRef]:
        if scope in self.member_consts:
            return list(self.member_consts[scope].values())
        # numeric literal ranges not declared as sorts
        raise CompileError(f"Count over undeclared sort {scope!r}")

    def _lower_distinct(self, distinct: FDistinct, bound: dict[str, z3.ExprRef]) -> z3.ExprRef:
        if distinct.vars:
            # scoped form: Distinct([v1:s1, v2:s2], expr) — express "the values
            # of expr as v1..vn range over their domains are pairwise distinct".
            # Lowered as: for every pair of distinct tuples of members,
            # expr(binding1) != expr(binding2).
            import itertools

            member_lists = [self._member_terms(scope) for _, scope in distinct.vars]
            terms: list[z3.ExprRef] = []
            combos = list(itertools.product(*member_lists))
            for (c1, c2) in itertools.combinations(combos, 2):
                b1 = dict(bound)
                b2 = dict(bound)
                for (var, _), m1, m2 in zip(distinct.vars, c1, c2):
                    b1[var] = m1
                    b2[var] = m2
                e1 = self._lower(distinct.expr, b1)
                e2 = self._lower(distinct.expr, b2)
                terms.append(e1 != e2)
            return z3.And(*terms) if terms else z3.BoolVal(True)
        # plain form: Distinct([a, b, c]) — pairwise != over the element names
        elems = [self._resolve_const(e) for e in distinct.expr]
        pairs = itertools.combinations(elems, 2)
        return z3.And(*(a != b for a, b in pairs)) if len(elems) > 1 else z3.BoolVal(True)


def compile_program(program: Z3DSLProgram) -> DSLCompiler:
    return DSLCompiler(program)
