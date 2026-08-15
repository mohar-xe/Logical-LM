"""Tests for the forward-chaining datalog engine (ProntoQA / ProofWriter)."""

import pytest

from logiclm.solvers.datalog import Atom, DatalogProgram, solve_datalog
from logiclm.solvers.errors import ParseError

from tests.golden_datalog import DATALOG_GOLDEN


@pytest.mark.parametrize("idx", range(len(DATALOG_GOLDEN)))
def test_golden_programs(idx):
    program_text, expected = DATALOG_GOLDEN[idx]
    got = solve_datalog(program_text, dataset="ProofWriter")
    assert got == expected, f"golden {idx}: got {got}, want {expected}"


def test_prontoqa_two_option_mapping():
    program = """Predicates:
Furry($x, bool)
Facts:
Furry(Anne, True)
Rules:
Furry($x, True) >>> Nice($x, True)
Query:
Nice(Anne, True)
"""
    # Nice(Anne, True) derivable -> matches expected True -> A
    assert solve_datalog(program, dataset="ProntoQA") == "A"
    # query asks for a False that isn't derivable -> B
    program2 = program.replace("Nice(Anne, True)\n", "Nice(Anne, False)\n")
    assert solve_datalog(program2, dataset="ProntoQA") == "B"


def test_open_world_unknown():
    program = """Predicates:
Furry($x, bool)
Facts:
Furry(Anne, True)
Rules:
Furry($x, False) >>> Nice($x, True)
Query:
Nice(Anne, True)
"""
    # Furry(Anne, False) is absent -> the rule never fires -> nothing derived
    prog = DatalogProgram(program, dataset="ProofWriter")
    assert prog.derive() == {Atom("Furry", ("Anne",), True)}
    assert solve_datalog(program, dataset="ProofWriter") == "C"


def test_multi_arity_query():
    program = """Predicates:
Likes($x, $y, bool)
Facts:
Likes(Cat, Dog, True)
Query:
Likes(Cat, Mouse, True)
"""
    assert solve_datalog(program, dataset="ProofWriter") == "C"


def test_derive_returns_closed_set():
    program = """Facts:
A(1, True)
Rules:
A($x, True) >>> B($x, True)
B($x, True) >>> C($x, True)
Query:
C(1, True)
"""
    derived = DatalogProgram(program).derive()
    assert Atom("B", ("1",), True) in derived
    assert Atom("C", ("1",), True) in derived


def test_parse_error_missing_query():
    with pytest.raises(ParseError):
        DatalogProgram("Facts:\nA(1, True)\n")


def test_parse_error_bad_bool():
    with pytest.raises(ParseError):
        DatalogProgram("Facts:\nA(1, Maybe)\nQuery:\nA(1, True)\n")


def test_parse_error_non_ground_query():
    with pytest.raises(ParseError):
        DatalogProgram("Facts:\nA(1, True)\nQuery:\nA($x, True)\n")


def test_multi_conclusion_rule():
    program = """Facts:
P(1, True)
Rules:
P($x, True) >>> Q($x, True) && R($x, True)
Query:
R(1, True)
"""
    assert solve_datalog(program, dataset="ProofWriter") == "A"
