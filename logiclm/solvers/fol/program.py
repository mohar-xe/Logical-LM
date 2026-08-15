"""FOLIO logic-program IR: parse ``Predicates:`` / ``Premises:`` / ``Conclusion:``
sections into a ``FOLProgram`` and, per premise, a parsed ``Formula`` AST.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast import Formula
from .parser import parse_formula
from ..errors import ParseError
from ...utils.unicode import normalize_text


def split_sections(program: str) -> list[tuple[str, str]]:
    """Split a program on section headers, returning (header, body) pairs.

    Headers are ``Predicates:``, ``Premises:``, ``Conclusion:`` ...  A body
    runs until the next header, with ``:::`` comments stripped and trailing
    blanks removed.
    """
    sections: list[tuple[str, str]] = []
    current_header: str | None = None
    body: list[str] = []
    for raw_line in program.splitlines():
        line = raw_line.strip()
        if line and line.endswith(":") and not line.startswith(":::") and len(line.split()) == 1:
            # candidate header: a single word ending with a colon
            if current_header is not None:
                sections.append((current_header, "\n".join(body).strip()))
            current_header = line[:-1]
            body = []
            continue
        if ":::" in line:
            line = line.split(":::", 1)[0].strip()
        if line:
            body.append(line)
    if current_header is not None:
        sections.append((current_header, "\n".join(body).strip()))
    return sections


def parse_predicates(text: str) -> list[str]:
    """Parse the ``Predicates:`` block into predicate names (with arity used
    only for validation).  A line looks like ``Dependent(x)`` or
    ``Love(x, y)``; we return the bare names.
    """
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip any trailing ": ::x does y" annotation before the parens
        name = line.split("(", 1)[0].strip()
        names.append(name)
    return names


@dataclass
class FOLProgram:
    """A parsed FOLIO logic program."""

    raw: str
    predicates: list[str] = field(default_factory=list)
    premises: list[Formula] = field(default_factory=list)
    conclusion: Formula | None = None

    @property
    def predicate_set(self) -> set[str]:
        return set(self.predicates)

    @classmethod
    def parse(cls, raw: str) -> "FOLProgram":
        sections = split_sections(raw)
        section_map = {header: body for header, body in sections}

        if "Premises" not in section_map or "Conclusion" not in section_map:
            raise ParseError("logic program is missing 'Premises:' or 'Conclusion:' sections")

        pred_set: set[str] = set()
        preds: list[str] = []
        if "Predicates" in section_map:
            preds = parse_predicates(section_map["Predicates"])
            pred_set = set(preds)

        # A predicate may be referenced in premises without being declared;
        # collect names of the form Name(...) directly.
        premises: list[Formula] = []
        for line in section_map["Premises"].splitlines():
            line = normalize_text(line)
            if not line:
                continue
            premises.append(parse_formula(line, pred_set))

        conclusion_text = section_map["Conclusion"].splitlines()
        conclusion = parse_formula(normalize_text(conclusion_text[0]), pred_set) \
            if conclusion_text else None

        return cls(raw=raw, predicates=preds, premises=premises, conclusion=conclusion)
