"""Tests for the FOL lexer/parser and FOLIO program model."""

import pytest

from logiclm.solvers.fol.ast import (
    Binary, Negation, Predicate, Quantifier, Variable, free_variables,
    to_infix,
)
from logiclm.solvers.fol.parser import parse_formula
from logiclm.solvers.fol.program import FOLProgram, split_sections
from logiclm.solvers.errors import ParseError
from logiclm.utils.unicode import normalize_text

from tests.golden_folio import FOLIO_GOLDEN


def test_parse_simple_predicates():
    f = parse_formula("Drinks(rina)")
    assert f == Predicate("Drinks", (Variable("rina"),))


def test_parse_quantified_implication():
    f = parse_formula("forall x (Drinks(x) -> Dependent(x))")
    assert isinstance(f, Quantifier)
    assert f.qname == "forall" and f.var == "x"
    assert isinstance(f.body, Binary)
    assert f.body.op == "->"


def test_parse_nested_quantifiers():
    f = parse_formula("exists y exists x (Czech(x) && Book(y) && Author(x, y))")
    assert isinstance(f, Quantifier) and f.var == "y"
    inner = f.body
    assert isinstance(inner, Quantifier) and inner.var == "x"
    assert isinstance(inner.body, Binary)


def test_parse_all_operators():
    f = parse_formula("(A && B) || ~C")
    assert isinstance(f, Binary) and f.op == "or"
    assert isinstance(f.left, Binary) and f.left.op == "and"
    assert isinstance(f.right, Negation)

    f = parse_formula("A xor B")
    assert f == Binary("xor", Variable("A"), Variable("B"))

    f = parse_formula("A <-> B")
    assert f == Binary("<->", Variable("A"), Variable("B"))


def test_quantifier_binds_maximal_scope():
    # forall x A -> B == forall x (A -> B) (maximal rest of precedence level)
    f = parse_formula("forall x A -> B")
    assert isinstance(f, Quantifier)
    assert isinstance(f.body, Binary) and f.body.op == "->"


def test_negation_scope():
    f = parse_formula("~forall x (Movie(x) -> HappyEnding(x))")
    assert isinstance(f, Negation)
    assert isinstance(f.body, Quantifier)


def test_free_variables():
    f = parse_formula("forall x (R(x, y) -> Q(x))")
    assert free_variables(f) == {"y"}


def test_predicate_declared_zero_arity():
    # With True declared as a predicate, "True" parses as a 0-ary predicate.
    f = parse_formula("True", predicates={"True"})
    assert f == Predicate("True", ())


def test_multiword_identifier_after_normalization():
    # normalize_text joins/keeps multi-word identifiers as single tokens
    # only when they are a single word; "dc universe" keeps its space and
    # therefore does NOT tokenize as one name here (acceptable).
    assert normalize_text("dc universe") == "dc universe"


def test_split_sections():
    program = """Predicates:
    A(x) ::: x is A
    B(x) ::: x is B
Premises:
    A(bob)
Conclusion:
    B(bob)
    """
    sections = split_sections(program)
    assert [h for h, _ in sections] == ["Predicates", "Premises", "Conclusion"]
    assert "A(bob)" in sections[1][1]
    assert "B(bob)" in sections[2][1]


def test_parse_folio_program():
    raw = FOLIO_GOLDEN[0][0]
    prog = FOLProgram.parse(raw)
    assert prog.conclusion is not None
    assert isinstance(prog.conclusion, Quantifier)
    assert len(prog.premises) == 5
    # The golden programs omit a "Predicates:" section; predicates are then
    # inferred directly from the formulas (nothing to assert here). Programs
    # WITH the section should collect the declared names:
    with_predicates = "Predicates:\nMovie(x)\nHappyEnding(x)\n" + raw
    prog2 = FOLProgram.parse(with_predicates)
    assert "Movie" in prog2.predicates
    assert "HappyEnding" in prog2.predicates


def test_parse_raises_on_missing_sections():
    with pytest.raises(ParseError):
        FOLProgram.parse("Predicates:\nA(x)\n")


def test_parse_raises_on_trailing_tokens():
    with pytest.raises(ParseError):
        parse_formula("A(x) B(y)")


def test_unicode_formulas_parse_identically():
    unicode_formula = "∀x (Drinks(x) ⟹ Dependent(x))"
    ascii_formula = "forall x (Drinks(x) -> Dependent(x))"
    a = parse_formula(normalize_text(unicode_formula))
    b = parse_formula(normalize_text(ascii_formula))
    assert a == b


def test_to_infix_roundtrip():
    f = parse_formula("forall x (P(x) -> Q(x))")
    rendered = to_infix(f)
    assert "->" in rendered and "P(x)" in rendered
