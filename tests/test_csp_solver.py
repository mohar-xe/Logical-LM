"""Tests for the LogicalDeduction CSP solver."""

import pytest

from logiclm.solvers.csp import CSPProgram, solve_csp, parse_domain, parse_variables
from logiclm.solvers.errors import ParseError, TimeoutError

BOOKS = """Domain:
1: leftmost
5: rightmost
Variables:
green_book [IN] [1, 2, 3, 4, 5]
blue_book [IN] [1, 2, 3, 4, 5]
white_book [IN] [1, 2, 3, 4, 5]
purple_book [IN] [1, 2, 3, 4, 5]
yellow_book [IN] [1, 2, 3, 4, 5]
Constraints:
blue_book > yellow_book ::: The blue book is to the right of the yellow book.
white_book < yellow_book ::: The white book is to the left of the yellow book.
blue_book == 4 ::: The blue book is the second from the right.
purple_book == 2 ::: The purple book is the second from the left.
AllDifferentConstraint([green_book, blue_book, white_book, purple_book, yellow_book]) ::: All books have different values.
Query:
A) green_book == 2 ::: The green book is the second from the left.
B) blue_book == 2 ::: The blue book is the second from the left.
C) white_book == 2 ::: The white book is the second from the left.
D) purple_book == 2 ::: The purple book is the second from the left.
E) yellow_book == 2 ::: The yellow book is the second from the left.
"""

BIRDS = """Domain:
1: leftmost
5: rightmost
Variables:
quail [IN] [1, 2, 3, 4, 5]
owl [IN] [1, 2, 3, 4, 5]
raven [IN] [1, 2, 3, 4, 5]
falcon [IN] [1, 2, 3, 4, 5]
robin [IN] [1, 2, 3, 4, 5]
Constraints:
owl == 1 ::: The owl is the leftmost.
robin < raven ::: The robin is to the left of the raven.
quail == 5 ::: The quail is the rightmost.
raven == 3 ::: The raven is the third from the left.
AllDifferentConstraint([quail, owl, raven, falcon, robin]) ::: All birds have different values.
Query:
A) quail == 5 ::: The quail is the rightmost.
B) owl == 5 ::: The owl is the rightmost.
C) raven == 5 ::: The raven is the rightmost.
D) falcon == 5 ::: The falcon is the rightmost.
E) robin == 5 ::: The robin is the rightmost.
"""


def test_books_golden():
    answer, solutions = solve_csp(BOOKS)
    assert answer == "D"
    assert len(solutions) == 1


def test_birds_golden():
    answer, _ = solve_csp(BIRDS)
    assert answer == "A"


def test_multiple_solutions_option_entailed_by_all():
    # Two solutions; a variable whose value differs across them must NOT be
    # an answer, but one fixed across all solutions should be.
    program = """Variables:
a [IN] [1, 2]
b [IN] [1, 2]
Constraints:
a != b
Query:
A) a == 1
B) b == 2
C) a == 5
"""
    csp = CSPProgram(program)
    solutions = csp.solve()
    assert len(solutions) == 2
    # a takes both 1 and 2, so neither "a==1" (holds in one solution only)
    # nor "a==5" holds everywhere; b similarly takes both.
    assert csp.answer_mapping(solutions) is None


def test_compound_constraint():
    program = """Variables:
a [IN] [1, 2, 3]
b [IN] [1, 2, 3]
c [IN] [1, 2, 3]
Constraints:
a > b && b > c
Query:
A) a == 3 && c == 1
B) a == 1
"""
    csp = CSPProgram(program)
    solutions = csp.solve()
    assert len(solutions) == 1
    assert csp.answer_mapping(solutions) == "A"


def test_unsat_constraints_yield_no_answer():
    program = """Variables:
a [IN] [1, 2]
b [IN] [1, 2]
Constraints:
a == b
a != b
Query:
A) a == 1
"""
    csp = CSPProgram(program)
    assert csp.solve() == []
    assert csp.answer_mapping([]) is None


def test_parse_domain_and_variables():
    domain = parse_domain("1: leftmost\n5: rightmost")
    assert domain == {"leftmost": 1, "rightmost": 5}
    vars_ = parse_variables("blue_book [IN] [1, 2, 3]\nred [IN] [2]")
    assert vars_ == [("blue_book", [1, 2, 3]), ("red", [2])]


def test_missing_sections_raises():
    with pytest.raises(ParseError):
        CSPProgram("Variables:\na [IN] [1]\nConstraints:\n")
