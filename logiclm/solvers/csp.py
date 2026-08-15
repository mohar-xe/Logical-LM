"""Constraint-satisfaction solver for the LogicalDeduction IR.

IR sections::

    Domain:    1: leftmost\n5: rightmost
    Variables: blue_book [IN] [1, 2, 3, 4, 5]
    Constraints:
        blue_book > yellow_book ::: comment
        AllDifferentConstraint([blue_book, yellow_book, ...]) ::: comment
    Query:
        A) green_book == 2 ::: comment
        B) ...

Constraints are Python-style comparisons over the variable names
(``>``, ``<``, ``==``, ``!=``, ``>=``, ``<=``, and ``&&`` conjunctions) plus
the special ``AllDifferentConstraint([...])``.  We hand-roll the evaluator and
enumerate *all* solutions with a forward-checking backtracking search, then
map each query option to a letter: an option is the answer iff its expression
holds in every solution (i.e. the queried variable's value is fixed across all
solutions and equals the option's value).
"""

from __future__ import annotations

import ast
import re
import time

from .errors import ParseError, TimeoutError

_ALLDIFF = "AllDifferentConstraint"


def _strip_comments(line: str) -> str:
    if ":::" in line:
        return line.split(":::", 1)[0].strip()
    return line.strip()


def parse_domain(text: str) -> dict[str, int]:
    """Parse ``1: leftmost`` lines into {semantic_label: value}."""
    mapping: dict[str, int] = {}
    for line in text.splitlines():
        line = _strip_comments(line)
        if not line:
            continue
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        mapping[right.strip()] = int(left.strip())
    return mapping


def parse_variables(text: str) -> list[tuple[str, list[int]]]:
    """Parse ``name [IN] [1, 2, 3]`` lines into (name, domain)."""
    variables: list[tuple[str, list[int]]] = []
    for line in text.splitlines():
        line = _strip_comments(line)
        if not line:
            continue
        if "[IN]" not in line:
            continue
        name, dom = line.split("[IN]", 1)
        dom = dom.strip()
        if not (dom.startswith("[") and dom.endswith("]")):
            raise ParseError(f"malformed domain in variable line: {line!r}")
        values = [int(v.strip()) for v in dom[1:-1].split(",") if v.strip() != ""]
        variables.append((name.strip(), values))
    return variables


class _ExprEval(ast.NodeVisitor):
    """Evaluate a constraint expression against a variable assignment."""

    def __init__(self, env: dict[str, int]):
        self.env = env

    def visit_Expression(self, node: ast.Expression) -> bool:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> int:
        if node.id in self.env:
            return self.env[node.id]
        raise ParseError(f"undefined variable in constraint: {node.id!r}")

    def visit_Constant(self, node: ast.Constant) -> int:
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp) -> bool:
        if isinstance(node.op, ast.Not):
            return not self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -self.visit(node.operand)
        raise ParseError(f"unsupported unary operator: {node.op!r}")

    def _bool_op(self, op: str, values: list[bool]) -> bool:
        return all(values) if op == "and" else any(values)

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        opname = "and" if isinstance(node.op, ast.And) else "or"
        return self._bool_op(opname, [self.visit(v) for v in node.values])

    def visit_BinOp(self, node: ast.BinOp) -> int | bool:
        if isinstance(node.op, ast.Add):
            return self.visit(node.left) + self.visit(node.right)
        if isinstance(node.op, ast.Sub):
            return self.visit(node.left) - self.visit(node.right)
        if isinstance(node.op, ast.Mult):
            return self.visit(node.left) * self.visit(node.right)
        raise ParseError(f"unsupported binary operator: {node.op!r}")

    def _cmp(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            else:
                raise ParseError(f"unsupported comparison: {op!r}")
            if not ok:
                return False
            left = right
        return True

    def visit_Compare(self, node: ast.Compare) -> bool:
        return self._cmp(node)


def _all_diff_ok(assignment: dict[str, int], vars_: list[str]) -> bool:
    seen: set[int] = set()
    for v in vars_:
        val = assignment.get(v)
        if val is None:
            continue  # not yet assigned
        if val in seen:
            return False
        seen.add(val)
    return True


class Constraint:
    def __init__(self, raw: str):
        self.raw = raw
        self.kind = "alldiff" if raw.startswith(_ALLDIFF) else "expr"
        if self.kind == "alldiff":
            m = re.search(r"AllDifferentConstraint\(\[(.*?)\]\)", raw, re.DOTALL)
            if not m:
                raise ParseError(f"malformed AllDifferentConstraint: {raw!r}")
            self.vars = [v.strip() for v in m.group(1).split(",") if v.strip()]
        else:
            # translate && and || to Python 'and'/'or' (the LLM IR uses &&)
            expr = raw.replace("&&", " and ").replace("||", " or ")
            try:
                self.tree = ast.parse(expr, mode="eval")
            except SyntaxError as e:
                raise ParseError(f"cannot parse constraint {raw!r}: {e}")

    def is_satisfied(self, assignment: dict[str, int], assigned: set[str]) -> bool:
        """True if the constraint holds over the assigned portion."""
        if self.kind == "alldiff":
            return _all_diff_ok(assignment, self.vars)
        # only evaluate when all referenced variables are assigned
        referenced = {n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name)}
        if not referenced <= assigned:
            return True
        try:
            return bool(_ExprEval(assignment).visit(self.tree))
        except ParseError:
            return False


class CSPProgram:
    def __init__(self, raw: str):
        self.raw = raw
        self.variables: list[tuple[str, list[int]]] = []
        self.constraints: list[Constraint] = []
        self.query: list[str] = []
        self.timeout_ms = 20_000
        self.max_solutions = 10_000
        self._parse()

    def _parse(self) -> None:
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for raw_line in self.raw.splitlines():
            line = raw_line.split(":::", 1)[0].strip()
            if not line:
                continue
            if line.rstrip(":") in {"Domain", "Variables", "Constraints", "Query"}:
                current = line.rstrip(":")
                sections.setdefault(current, [])
                continue
            if current:
                sections.setdefault(current, []).append(line)

        if "Variables" not in sections or "Constraints" not in sections or "Query" not in sections:
            raise ParseError("logic program is missing Variables/Constraints/Query sections")

        self.variables = parse_variables("\n".join(sections["Variables"]))
        if not self.variables:
            raise ParseError("no variables declared")

        self.constraints = [Constraint(_strip_comments(l)) for l in sections["Constraints"]]
        self.query = [_strip_comments(l) for l in sections["Query"] if _strip_comments(l)]
        if not self.query:
            raise ParseError("no query options")

        self._names = [name for name, _ in self.variables]

    # -- solving ----------------------------------------------------------

    def solve(self) -> list[dict[str, int]]:
        """Enumerate every satisfying assignment (cap at ``max_solutions``)."""
        solutions: list[dict[str, int]] = []
        order = self._order_variables()
        domains = {name: list(dom) for name, dom in self.variables}
        deadline = time.monotonic() + self.timeout_ms / 1000

        def backtrack(assignment: dict[str, int], assigned: set[str], idx: int) -> bool:
            if time.monotonic() > deadline:
                raise TimeoutError("CSP solver timed out")
            if idx == len(order):
                solutions.append(dict(assignment))
                return len(solutions) >= self.max_solutions
            name = order[idx]
            for value in domains[name]:
                assignment[name] = value
                assigned.add(name)
                if all(c.is_satisfied(assignment, assigned) for c in self.constraints):
                    if backtrack(assignment, assigned, idx + 1):
                        return True
                del assignment[name]
                assigned.discard(name)
            return False

        try:
            backtrack({}, set(), 0)
        except TimeoutError:
            raise
        return solutions

    def _order_variables(self) -> list[str]:
        # most-constrained-first: variables appearing in most constraints first
        counts = {name: 0 for name, _ in self.variables}
        for c in self.constraints:
            if c.kind == "alldiff":
                for v in c.vars:
                    if v in counts:
                        counts[v] += 1
            else:
                for n in ast.walk(c.tree):
                    if isinstance(n, ast.Name) and n.id in counts:
                        counts[n.id] += 1
        return sorted(self._names, key=lambda n: (-counts[n], n))

    # -- answer mapping ---------------------------------------------------

    def answer_mapping(self, solutions: list[dict[str, int]]) -> str | None:
        """Return the answer letter, or None if no option is entailed."""
        option_pattern = re.compile(r"^([A-G])\)\s*(.+)")
        for option_str in self.query:
            m = option_pattern.match(option_str)
            if not m:
                continue
            letter, expr = m.group(1), m.group(2).strip()
            try:
                tree = ast.parse(expr.replace("&&", " and ").replace("||", " or "), mode="eval")
            except SyntaxError:
                continue
            values = []
            ok = True
            for sol in solutions:
                try:
                    values.append(bool(_ExprEval(sol).visit(tree)))
                except ParseError:
                    ok = False
                    break
            if not ok or not values:
                continue
            if all(values):
                return letter
        return None


def solve_csp(program_text: str) -> tuple[str | None, list[dict[str, int]]]:
    """Parse, solve and map to an answer letter."""
    program = CSPProgram(program_text)
    solutions = program.solve()
    return program.answer_mapping(solutions), solutions
