"""Forward-chaining (Datalog-style) rule engine for ProntoQA / ProofWriter.

The IR produced by the LLM has four sections::

    Predicates:  Pred($x, bool) ::: comment
    Facts:       Pred(term, term, True) ::: comment
    Rules:       P($x, True) && Q($x, True) >>> R($x, True) ::: comment
    Query:       Pred(constant, ..., False) ::: comment

Semantics
---------
* Facts and derived atoms are *ground* ``(predicate, terms, bool)``.
* Rules are Horn-ish: a conjunction of atom patterns implies a conjunction of
  atoms.  A ``False``-valued premise matches only atoms literally present in
  the closed set (no negation-as-failure), which reproduces the reference's
  handling of ProofWriter's non-monotonic rules.
* The query is a ground atom; the answer depends on which bool values are
  derivable for that predicate/constants prefix.

Answer mapping
--------------
* ProntoQA (2 options): A iff the derivable label set == {expected}, else B.
* ProofWriter (3 options): C if nothing derivable, A iff == {expected}, else B.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ParseError

_TRUE = "True"
_FALSE = "False"
_VAR_PREFIX = "$"


@dataclass(frozen=True)
class Atom:
    """A ground atom: predicate, constant terms (excluding the bool), value."""

    predicate: str
    terms: tuple[str, ...]
    value: bool


@dataclass(frozen=True)
class AtomPattern:
    """A (possibly variable) atom in a rule premise/conclusion.

    ``terms`` may contain ``$x``-style variables (also bare words are treated
    as variables in rule bodies, matching the LLM IR).
    """

    predicate: str
    terms: tuple[str, ...]
    value: bool


@dataclass(frozen=True)
class Rule:
    premises: tuple[AtomPattern, ...]
    conclusions: tuple[AtomPattern, ...]


def _parse_bool(token: str) -> bool:
    if token == _TRUE:
        return True
    if token == _FALSE:
        return False
    raise ParseError(f"expected True/False in atom, got {token!r}")


def _parse_call(line: str) -> tuple[str, tuple[str, ...]]:
    """Parse ``Pred(a, b, True)`` -> (pred, (a, b, True-as-string))."""
    open_paren = line.find("(")
    close_paren = line.rfind(")")
    if open_paren < 0 or close_paren < open_paren:
        raise ParseError(f"malformed atom: {line!r}")
    name = line[:open_paren].strip()
    args = line[open_paren + 1 : close_paren].strip()
    if not name or not args:
        raise ParseError(f"malformed atom: {line!r}")
    tokens = [a.strip() for a in args.split(",") if a.strip() != ""]
    return name, tuple(tokens)


def _is_variable(token: str) -> bool:
    return token.startswith(_VAR_PREFIX) or token in {"x", "y", "z", "a", "b", "c"}


class DatalogProgram:
    def __init__(self, raw: str, dataset: str = "ProofWriter"):
        self.raw = raw
        self.dataset = dataset
        self.facts: set[Atom] = set()
        self.rules: list[Rule] = []
        self.query: Atom | None = None
        self._parse()

    # -- parsing ----------------------------------------------------------

    def _parse(self) -> None:
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for raw_line in self.raw.splitlines():
            line = raw_line.split(":::", 1)[0].strip()
            if not line:
                continue
            header = line.split(" ", 1)[0].rstrip(":")
            if header.title() in {"Predicates", "Facts", "Rules", "Query"}:
                current = header.title()
                sections.setdefault(current, [])
                continue
            if current in {"Facts", "Rules", "Query"}:
                sections.setdefault(current, []).append(line)

        for fact_line in sections.get("Facts", []):
            self.facts.add(self._parse_atom(fact_line))
        for rule_line in sections.get("Rules", []):
            self.rules.append(self._parse_rule(rule_line))

        queries = sections.get("Query", [])
        if len(queries) != 1:
            raise ParseError(f"expected exactly one Query line, got {len(queries)}")
        self.query = self._parse_atom(queries[0])

    def _parse_atom(self, line: str) -> Atom:
        pred, tokens = _parse_call(line)
        if not tokens:
            raise ParseError(f"empty atom: {line!r}")
        value = _parse_bool(tokens[-1])
        if any(_is_variable(t) for t in tokens[:-1]):
            raise ParseError(f"query/fact atom is not ground: {line!r}")
        return Atom(pred, tokens[:-1], value)

    def _parse_pattern(self, token: str) -> AtomPattern:
        pred, args = _parse_call(token)
        if not args:
            raise ParseError(f"empty atom pattern: {token!r}")
        value = _parse_bool(args[-1])
        return AtomPattern(pred, args[:-1], value)

    def _parse_rule(self, line: str) -> Rule:
        if ">>>" not in line:
            raise ParseError(f"rule missing '>>>': {line!r}")
        premise_str, conclusion_str = line.split(">>>", 1)
        premises = tuple(self._parse_pattern(t) for t in premise_str.split("&&"))
        conclusions = tuple(self._parse_pattern(t) for t in conclusion_str.split("&&"))
        if not premises or not conclusions:
            raise ParseError(f"empty rule: {line!r}")
        return Rule(premises, conclusions)

    # -- inference --------------------------------------------------------

    def derive(self) -> set[Atom]:
        """Closed set of all derivable atoms (fixpoint of forward chaining)."""
        derived: set[Atom] = set(self.facts)
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                for binding in self._match_all(rule.premises, derived):
                    for concl in rule.conclusions:
                        ground = self._instantiate(concl, binding)
                        if ground not in derived:
                            derived.add(ground)
                            changed = True
        return derived

    def _match_all(self, premises: tuple[AtomPattern, ...],
                   derived: set[Atom]) -> list[dict[str, str]]:
        """Enumerate every variable binding satisfying all premises."""
        bindings: list[dict[str, str]] = [{}]
        for premise in premises:
            next_bindings: list[dict[str, str]] = []
            for binding in bindings:
                for atom in derived:
                    if atom.predicate != premise.predicate or atom.value != premise.value:
                        continue
                    merged = self._unify(premise.terms, atom.terms, binding)
                    if merged is not None:
                        next_bindings.append(merged)
            bindings = next_bindings
            if not bindings:
                break
        return bindings

    def _unify(self, pattern_terms: tuple[str, ...], atom_terms: tuple[str, ...],
               binding: dict[str, str]) -> dict[str, str] | None:
        if len(pattern_terms) != len(atom_terms):
            return None
        merged = dict(binding)
        for p, a in zip(pattern_terms, atom_terms):
            if _is_variable(p):
                if p in merged and merged[p] != a:
                    return None
                merged[p] = a
            elif p != a:
                return None
        return merged

    def _instantiate(self, pattern: AtomPattern, binding: dict[str, str]) -> Atom:
        terms = tuple(binding.get(t, t) for t in pattern.terms)
        return Atom(pattern.predicate, terms, pattern.value)

    # -- answering --------------------------------------------------------

    def answer(self, derived: set[Atom] | None = None) -> str:
        """Return the answer letter for this program's query."""
        if self.query is None:
            raise ParseError("no query")
        closed = derived if derived is not None else self.derive()

        # all bool values derivable for the query predicate + constants prefix
        labels = {
            atom.value
            for atom in closed
            if atom.predicate == self.query.predicate
            and atom.terms == self.query.terms
        }
        expected = self.query.value

        if self.dataset == "ProntoQA":
            return "A" if labels == {expected} else "B"
        # ProofWriter open-world
        if not labels:
            return "C"
        return "A" if labels == {expected} else "B"


def solve_datalog(program_text: str, dataset: str = "ProofWriter") -> str:
    """Parse + derive + answer in one call."""
    program = DatalogProgram(program_text, dataset=dataset)
    return program.answer()
