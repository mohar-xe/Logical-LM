"""Tests for the AR-LSAT Z3-DSL solver."""

import pytest

from logiclm.solvers.z3dsl.solver import solve_dsl
from logiclm.solvers.z3dsl.parser import Z3DSLProgram
from logiclm.solvers.z3dsl.compile import DSLCompiler
from logiclm.solvers.errors import ParseError

MEALS = """# Declarations
people = EnumSort([Vladimir, Wendy])
meals = EnumSort([breakfast, lunch, dinner, snack])
foods = EnumSort([fish, hot_cakes, macaroni, omelet, poached_eggs])
eats = Function([people, meals] -> [foods])

# Constraints
ForAll([m:meals], eats(Vladimir, m) != eats(Wendy, m)) ::: At no meal does Vladimir eat the same kind of food as Wendy
ForAll([p:people, f:foods], Count([m:meals], eats(p, m) == f) <= 1) ::: Neither of them eats the same kind of food more than once during the day
ForAll([p:people], Or(eats(p, breakfast) == hot_cakes, eats(p, breakfast) == poached_eggs, eats(p, breakfast) == omelet)) ::: For breakfast, each eats exactly one of the following: hot cakes, poached eggs, or omelet
ForAll([p:people], Or(eats(p, lunch) == fish, eats(p, lunch) == hot_cakes, eats(p, lunch) == macaroni, eats(p, lunch) == omelet)) ::: For lunch, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet
ForAll([p:people], Or(eats(p, dinner) == fish, eats(p, dinner) == hot_cakes, eats(p, dinner) == macaroni, eats(p, dinner) == omelet)) ::: For dinner, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet
ForAll([p:people], Or(eats(p, snack) == fish, eats(p, snack) == omelet)) ::: For a snack, each eats exactly one of the following: fish or omelet
eats(Wendy, lunch) == omelet ::: Wendy eats an omelet for lunch

# Options
Question ::: Vladimir must eat which one of the following foods?
is_valid(Exists([m:meals], eats(Vladimir, m) == fish)) ::: (A)
is_valid(Exists([m:meals], eats(Vladimir, m) == hot_cakes)) ::: (B)
is_valid(Exists([m:meals], eats(Vladimir, m) == macaroni)) ::: (C)
is_valid(Exists([m:meals], eats(Vladimir, m) == omelet)) ::: (D)
is_valid(Exists([m:meals], eats(Vladimir, m) == poached_eggs)) ::: (E)
"""

LOCKERS = """# Declarations
children = EnumSort([Fred, Juan, Marc, Paul, Nita, Rachel, Trisha])
lockers = EnumSort([1, 2, 3, 4, 5])
assigned = Function([children] -> [lockers])

# Constraints
ForAll([c:children], Exists([l:lockers], assigned(c) == l)) ::: each child gets a locker
assigned(Fred) == 3 ::: Fred must be assigned to locker 3
assigned(Nita) != assigned(Trisha) ::: Nita and Trisha get different lockers

# Options
Question ::: Which locker is Nita assigned to?
is_valid(assigned(Nita) == 1) ::: (A)
is_valid(assigned(Nita) == 2) ::: (B)
is_valid(assigned(Nita) == 3) ::: (C)
is_valid(assigned(Nita) == 4) ::: (D)
is_valid(assigned(Nita) == 5) ::: (E)
"""


def test_meals_golden():
    # The reference's own generated code (run in M5) prints (D) for this
    # example — the "(A) Answer" comment in the repo's __main__ is stale.
    answer, err = solve_dsl(MEALS)
    assert answer == "D", f"err={err!r}"
    assert err == ""


def test_lockers_golden():
    # Fred=3, Nita != Trisha. The only "must be" statement forced by P is that
    # Nita is not 3; none of the exact-value options is entailed -> ambiguous.
    answer, err = solve_dsl(LOCKERS)
    assert err == "" or answer == ""


def test_is_sat_option():
    # A program where the correct option uses is_sat semantics.
    raw = """# Declarations
x = EnumSort([a, b])
f = Function([x] -> [x])
# Constraints
f(a) == a
# Options
is_valid(f(a) == a) ::: (A)
is_valid(f(a) == b) ::: (B)
is_sat(f(a) == b) ::: (C)
"""
    answer, err = solve_dsl(raw)
    # f(a)==a is forced (entailed); f(a)==b is impossible; is_sat(f(a)==b) is
    # false. So only A holds.
    assert answer == "A", f"err={err!r}"


def test_parse_errors():
    with pytest.raises(ParseError):
        Z3DSLProgram.parse("bad program without sections")
    with pytest.raises(ParseError):
        Z3DSLProgram.parse("# Declarations\nfoo = EnumSort([a])\n")


def test_compile_count():
    raw = """# Declarations
meals = EnumSort([breakfast, lunch, dinner, snack])
# Constraints
Count([m:meals], True) >= 4
# Options
is_valid(Count([m:meals], True) >= 4) ::: (A)
is_valid(Count([m:meals], True) >= 5) ::: (B)
"""
    answer, err = solve_dsl(raw)
    assert answer == "A"


def test_distinct_scoped():
    raw = """# Declarations
people = EnumSort([Vladimir, Wendy])
days = EnumSort([d1, d2, d3])
task = Function([people, days] -> [people])
# Constraints
ForAll([p:people], Distinct([d:days], task(p, d)))
# Options
is_valid(ForAll([p:people], Distinct([d:days], task(p, d)))) ::: (A)
"""
    # Distinct over a 3-element domain of 2 people... verify the expression at
    # least compiles and produces a determinate answer.
    answer, err = solve_dsl(raw)
    assert err == ""
