"""Tests for the self-refinement loop."""

from logiclm.refine import refine_programs
from logiclm.schema import Example
from logiclm.solvers.registry import SolverRegistry
from logiclm.llm.mock import MockClient

BAD = "this is not a logic program"
GOOD = """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?

Facts:
Furry(Anne, True) ::: Anne is furry.

Rules:
Furry($x, True) >>> Nice($x, True) ::: All furry things are nice.

Query:
Nice(Anne, True) ::: Anne is nice.
"""


def _ex(ex_id, program):
    return Example(id=ex_id, context="ctx", question="q", answer="A",
                   options=["A) True", "B) False", "C) Unknown"], programs=[program])


FOLIO_BAD = "Premises:\nP(a)\nConclusion:\nthis is garbage (("
FOLIO_GOOD = """Premises:
Animal(felix) ::: Felix is an animal.
forall x (Animal(x) -> Living(x)) ::: All animals are living things.
Conclusion:
Living(felix) ::: Felix is living.
"""


def _folio_ex(ex_id, program):
    return Example(id=ex_id, context="ctx", question="q", answer="A",
                   options=["A) True", "B) False", "C) Unknown"], programs=[program])


def test_converges_when_correction_fixes_program():
    # The mock returns FOLIO_GOOD for any self-correction prompt (fragment
    # match on the bad-program text inside the prompt).
    llm = MockClient({FOLIO_BAD: FOLIO_GOOD})
    solver = SolverRegistry()
    examples = [_folio_ex("a", FOLIO_BAD)]
    report = refine_programs(examples, "FOLIO", llm, solver, max_rounds=3)
    assert report.rounds_used == 1
    assert examples[0].programs == [FOLIO_GOOD.strip()]
    assert "a" in report.revised
    assert report.calls_per_example["a"] == 1


def test_success_carry_over_never_recalls_llm():
    llm = MockClient({})
    solver = SolverRegistry()
    examples = [_folio_ex("good", FOLIO_GOOD)]
    report = refine_programs(examples, "FOLIO", llm, solver, max_rounds=3)
    assert report.rounds_used == 0  # nothing failed -> no rounds
    assert report.calls_per_example == {}
    assert not report.revised


def test_stops_after_max_rounds_if_never_fixed():
    # Mock always returns another bad program; loop must stop at max_rounds.
    llm = MockClient({FOLIO_BAD: FOLIO_BAD})
    solver = SolverRegistry()
    examples = [_folio_ex("stubborn", FOLIO_BAD)]
    report = refine_programs(examples, "FOLIO", llm, solver, max_rounds=3)
    assert report.rounds_used == 3
    assert report.calls_per_example["stubborn"] == 3


def test_only_failed_examples_revised():
    llm = MockClient({FOLIO_BAD: FOLIO_GOOD})
    solver = SolverRegistry()
    examples = [_folio_ex("good", FOLIO_GOOD), _folio_ex("bad", FOLIO_BAD)]
    report = refine_programs(examples, "FOLIO", llm, solver, max_rounds=3)
    assert report.revised == ["bad"]
    assert report.calls_per_example == {"bad": 1}
    # the good example was never revised -> original (unmodified) program
    assert examples[0].programs == [FOLIO_GOOD]
    assert examples[1].programs == [FOLIO_GOOD.strip()]
