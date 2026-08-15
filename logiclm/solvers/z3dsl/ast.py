"""DSL formula AST and declaration dataclasses.

Kept separate from ``parser.py`` to avoid a circular import between the
parser and the formula parser (which both need the node types).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnumSortDecl:
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class IntSortDecl:
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class FunctionDecl:
    name: str
    arg_sorts: tuple[str, ...]
    result_sort: str


@dataclass(frozen=True)
class FConst:
    name: str  # a constant / sort member, e.g. breakfast, 1


@dataclass(frozen=True)
class FVar:
    name: str


@dataclass(frozen=True)
class FApp:
    func: str  # function name (eats) or relation handled separately
    args: tuple["Formula", ...]


@dataclass(frozen=True)
class FQuant:
    qname: str  # "ForAll" | "Exists"
    vars: tuple[tuple[str, str], ...]  # (var, scope_sort)
    body: "Formula"


@dataclass(frozen=True)
class FBoolOp:
    op: str  # And | Or | Not | Implies | Iff | Xor (as names)
    args: tuple["Formula", ...]


@dataclass(frozen=True)
class FCmp:
    op: str  # == != < > <= >=
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class FArith:
    op: str  # + - * /
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class FCount:
    """Count([v:sort], cond) — number of bindings satisfying cond."""

    vars: tuple[tuple[str, str], ...]
    cond: "Formula"


@dataclass(frozen=True)
class FDistinct:
    """Distinct([v:sort]) or Distinct([list]) — pairwise inequality."""

    vars: tuple[tuple[str, str], ...]
    expr: "Formula | None" = None  # expr form: Distinct([...] for ...) with bound var


Formula = (FConst | FVar | FApp | FQuant | FBoolOp | FCmp | FArith | FCount | FDistinct)
"""Union alias for the DSL formula AST (kept as a global for doc purposes)."""
