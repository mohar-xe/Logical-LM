"""Tests for the FOLIO Z3 solver (prove/refute/unknown semantics)."""

import pytest

from logiclm.solvers.fol.solver import FOLSolver
from logiclm.solvers.errors import ExecutionError, ParseError

from tests.golden_folio import FOLIO_GOLDEN


@pytest.fixture(scope="module")
def solver():
    return FOLSolver(timeout_ms=10_000)


def test_golden_programs(solver):
    # Expected verdicts from the reference repo's __main__ programs.
    expected = {
        0: "True",    # ~∀x(Movie→HappyEnding), titanic is a movie with no HE -> some movie has no HE
        1: "True",    # coffee/jokes/student caffeine chain
        2: "Unknown", # Miroslav loved music? ∃x(Musician∧Love(x,music)) doesn't force Love(miroslav)
        3: "True",    # Czech person wrote a book in 1946
        4: "False",   # no choral conductor specialized in renaissance (contradicts premise)
        5: "False",   # ∃x(Movie∧¬HappyEnding) refuted
        6: "Unknown", # John is tall vs short — neither provable nor refutable
        7: "False",   # ~Animal(tom) refuted by Cat(tom)→Animal(tom)
        8: "True",    # modus ponens felix
    }
    for idx, (program, _) in enumerate(FOLIO_GOLDEN):
        verdict, err = solver.solve(program)
        assert verdict == expected[idx], f"program {idx}: got {verdict}, want {expected[idx]}"
        assert err == ""


def test_answer_mapping():
    assert FOLSolver.answer_mapping("True") == "A"
    assert FOLSolver.answer_mapping("False") == "B"
    assert FOLSolver.answer_mapping("Unknown") == "C"
    with pytest.raises(ValueError):
        FOLSolver.answer_mapping("Maybe")


def test_strict_unknown_escalates():
    # A program whose conclusion is genuinely underdetermined is NOT an error
    # in the default mode, but --strict-unknown must not escalate a normal
    # "Unknown" verdict (it only escalates z3 'unknown', a solver failure).
    solver = FOLSolver(timeout_ms=10_000, strict_unknown=True)
    verdict, _ = solver.solve(FOLIO_GOLDEN[6][0])  # "John is tall" -> Unknown
    assert verdict == "Unknown"


def test_parse_error_is_raised():
    solver = FOLSolver()
    with pytest.raises(ParseError):
        solver.solve("Premises:\nforall x (P(x))\nConclusion:\n@@@")


def test_fresh_sorts_not_shared_between_examples(solver):
    # Constants of one example must not leak into another: 'john' in one
    # program and 'tom' in another should behave independently.
    v1, _ = solver.solve(FOLIO_GOLDEN[6][0])  # John tall/short -> Unknown
    v2, _ = solver.solve(FOLIO_GOLDEN[7][0])  # Tom cat/animal -> False
    assert v1 == "Unknown"
    assert v2 == "False"
