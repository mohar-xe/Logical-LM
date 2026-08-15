"""Parser for the AR-LSAT (SatLM) domain-specific language.

IR sections::

    # Declarations
    people = EnumSort([Vladimir, Wendy])
    meals = EnumSort([breakfast, lunch, dinner, snack])
    foods = EnumSort([fish, hot_cakes, macaroni, omelet, poached_eggs])
    eats = Function([people, meals] -> [foods])

    # Constraints
    ForAll([m:meals], eats(Vladimir, m) != eats(Wendy, m)) ::: comment
    ...

    # Options
    Question ::: ...
    is_valid(Exists([m:meals], eats(Vladimir, m) == fish)) ::: (A)
    ...

Declarations and formulas are lowered to in-memory dataclasses/ASTs
(``decls.py`` / ``formula.py``); ``compile.py`` turns those into Z3 objects.
Nothing here touches z3 directly and nothing ever ``exec``s LLM text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast import (
    EnumSortDecl, FunctionDecl, FApp, FArith, FBoolOp, FCmp, FConst, FCount,
    FDistinct, FQuant, FVar, Formula,
)
from .formula_parser import parse_formula as _parse_dsl_formula

from ..errors import ParseError
from ...utils.unicode import normalize_text


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class Z3DSLProgram:
    raw: str
    enum_sorts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    int_sorts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    functions: dict[str, FunctionDecl] = field(default_factory=dict)
    constraints: list[Formula] = field(default_factory=list)
    options: list[tuple[str, Formula]] = field(default_factory=list)  # (choice_label, formula)
    question: str = ""

    @classmethod
    def parse(cls, raw: str) -> "Z3DSLProgram":
        prog = cls(raw=raw)
        sections = _split_sections(raw)
        for header, body in sections:
            h = header.strip().lstrip("#").strip()
            if h == "Declarations":
                for line in body.splitlines():
                    _parse_declaration(prog, line)
            elif h == "Constraints":
                for line in body.splitlines():
                    line = _strip_comment(line)
                    if line:
                        prog.constraints.append(_parse_dsl_formula(line, prog))
            elif h == "Options":
                for raw_line in body.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("Question"):
                        prog.question = line.split(":::", 1)[-1].strip()
                        continue
                    stripped = _strip_comment(line)
                    if stripped:
                        choice, formula = _parse_option(line, prog)
                        prog.options.append((choice, formula))
        if not prog.constraints:
            raise ParseError("DSL program has no constraints")
        return prog


def _split_sections(raw: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current: str | None = None
    body: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and stripped[1:].strip() in {
            "Declarations", "Constraints", "Options",
        }:
            if current is not None:
                sections.append((current, "\n".join(body)))
            current = stripped
            body = []
        else:
            body.append(line)
    if current is not None:
        sections.append((current, "\n".join(body)))
    return sections


def _strip_comment(line: str) -> str:
    return line.split(":::", 1)[0].strip()


def _parse_declaration(prog: Z3DSLProgram, line: str) -> None:
    line = _strip_comment(line)
    if not line or "=" not in line:
        return
    name, rhs = line.split("=", 1)
    name = name.strip()
    rhs = rhs.strip()
    if rhs.startswith("EnumSort("):
        members = _parse_list_arg(rhs[len("EnumSort(") : -1])
        if members and all(m.isdigit() for m in members):
            prog.int_sorts[name] = members
        else:
            prog.enum_sorts[name] = members
    elif rhs.startswith("IntSort("):
        members = _parse_list_arg(rhs[len("IntSort(") : -1])
        prog.int_sorts[name] = members
    elif rhs.startswith("Function("):
        # Function([a, b] -> [c])
        inner = rhs[len("Function(") : -1].strip()
        if "->" in inner:
            args_part, result_part = inner.split("->", 1)
            arg_sorts = _parse_list_arg(args_part.strip())
            result_sort = _parse_list_arg(result_part.strip())
            if len(result_sort) != 1:
                raise ParseError(f"Function result must be a single sort: {line!r}")
            prog.functions[name] = FunctionDecl(name, tuple(arg_sorts), result_sort[0])
        else:
            # legacy: Function([a, b]) with no result
            arg_sorts = _parse_list_arg(inner)
            prog.functions[name] = FunctionDecl(name, tuple(arg_sorts), "bool")
    else:
        raise ParseError(f"unknown declaration: {line!r}")


def _parse_list_arg(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def _parse_option(line: str, prog) -> tuple[str, Formula]:
    """Parse ``is_valid(expr) ::: (A)`` -> (label, formula)."""
    choice = ""
    if ":::" in line:
        line, comment = line.split(":::", 1)
        comment = comment.strip()
        import re as _re
        m = _re.match(r"\(([A-E])\)", comment)
        if m:
            choice = m.group(1)
    line = line.strip()
    formula = _parse_dsl_formula(line, prog)
    return choice, formula
