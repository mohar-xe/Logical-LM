"""Tests for unicode normalization shared by the FOL and Z3-DSL lexers."""

from logiclm.utils.unicode import normalize_text, canonical_operator_map


def test_operator_synonyms_normalize_to_canonical_forms():
    # Every spelling of implication collapses to '->'
    for src in ["→", "⟹", "⇒", "->", "=>"]:
        assert normalize_text(f"x {src} y") == "x -> y"
    # conjunction
    assert normalize_text("a ∧ b") == "a && b"
    assert normalize_text("a ⋀ b") == "a && b"
    # disjunction / negation / xor / biconditional / quantifiers
    assert normalize_text("a ∨ b") == "a || b"
    assert normalize_text("¬ a") == "~ a"
    assert normalize_text("a ⊕ b") == "a xor b"
    assert normalize_text("a ⇔ b") == "a <-> b"
    assert normalize_text("a <=> b") == "a <-> b"
    assert normalize_text("∀ x") == "forall x"
    assert normalize_text("∃ x") == "exists x"


def test_identifiers_are_not_corrupted():
    # word-form operators must be untouched; identifiers must survive
    assert normalize_text("Andrew and Beatrice") == "Andrew and Beatrice"
    assert normalize_text("Ford") == "Ford"
    assert normalize_text("And(assigned(x) == 1, x != 2)") == "And(assigned(x) == 1, x != 2)"
    assert normalize_text("ForAll([m:meals], ...)") == "ForAll([m:meals], ...)"


def test_nbsp_and_quotes_removed():
    assert normalize_text("a  b") == "a b"
    assert normalize_text("dc’s universe") == "dc's universe"


def test_whitespace_collapsed():
    assert normalize_text("  a    b\tc  ") == "a b c"


def test_operator_map_keys_are_symbols_not_words():
    # Regression guard: every SOURCE key must be a symbol, never a bare ASCII
    # word — otherwise substitution would corrupt identifiers like "Andrew".
    for src in canonical_operator_map():
        assert not src.isalpha(), f"word-form source key: {src!r}"
