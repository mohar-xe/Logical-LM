"""Golden ProntoQA/ProofWriter logic programs adapted from the reference
repo's ``pyke_solver.py.__main__``.  Each tuple is (program_text, expected
answer letter), in the exact order of the reference's tested programs
(A,B,C,A,B,C,B per the plan).

They exercise facts, forward rules, multi-arity predicates, negated-valued
facts, and open-world "unknown" answers.  All run under ProofWriter semantics
(what the reference actually exercised).
"""

from __future__ import annotations

# Answer letters: ProofWriter uses A=proved, B=refuted, C=unknown.
DATALOG_GOLDEN: list[tuple[str, str]] = []

# reference program 4 -> A (Kind(Charlie, True) derivable via rules)
DATALOG_GOLDEN.append((
    """Predicates:
Cold($x, bool) ::: Is x cold?
Quiet($x, bool) ::: Is x quiet?
Red($x, bool) ::: Is x red?
Smart($x, bool) ::: Is x smart?
Kind($x, bool) ::: Is x kind?
Rough($x, bool) ::: Is x rough?
Round($x, bool) ::: Is x round?

Facts:
Cold(Bob, True) ::: Bob is cold.
Quiet(Bob, True) ::: Bob is quiet.
Red(Bob, True) ::: Bob is red.
Smart(Bob, True) ::: Bob is smart.
Kind(Charlie, True) ::: Charlie is kind.
Quiet(Charlie, True) ::: Charlie is quiet.
Red(Charlie, True) ::: Charlie is red.
Rough(Charlie, True) ::: Charlie is rough.
Cold(Dave, True) ::: Dave is cold.
Kind(Dave, True) ::: Dave is kind.
Smart(Dave, True) ::: Dave is smart.
Quiet(Fiona, True) ::: Fiona is quiet.

Rules:
Quiet($x, True) && Cold($x, True) >>> Smart($x, True) ::: If something is quiet and cold then it is smart.
Red($x, True) && Cold($x, True) >>> Round($x, True) ::: Red, cold things are round.
Kind($x, True) && Rough($x, True) >>> Red($x, True) ::: If something is kind and rough then it is red.
Quiet($x, True) >>> Rough($x, True) ::: All quiet things are rough.
Cold($x, True) && Smart($x, True) >>> Red($x, True) ::: Cold, smart things are red.
Rough($x, True) >>> Cold($x, True) ::: If something is rough then it is cold.
Red($x, True) >>> Rough($x, True) ::: All red things are rough.
Smart(Dave, True) && Kind(Dave, True) >>> Quiet(Dave, True) ::: If Dave is smart and Dave is kind then Dave is quiet.

Query:
Kind(Charlie, True) ::: Charlie is kind.
""",
    "A",
))

# reference program 2 -> B (Green(Harry, False) refuted: Harry is green)
DATALOG_GOLDEN.append((
    """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?
Smart($x, bool) ::: Is x smart?
Young($x, bool) ::: Is x young?
Green($x, bool) ::: Is x green?
Big($x, bool) ::: Is x big?
Round($x, bool) ::: Is x round?

Facts:
Furry(Anne, True) ::: Anne is furry.
Nice(Anne, True) ::: Anne is nice.
Smart(Anne, True) ::: Anne is smart.
Young(Bob, True) ::: Bob is young.
Nice(Erin, True) ::: Erin is nice.
Smart(Harry, True) ::: Harry is smart.
Young(Harry, True) ::: Harry is young.

Rules:
Young($x, True) >>> Furry($x, True) ::: Young things are furry.
Nice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.
Green($x, True) >>> Nice($x, True) ::: All green things are nice.
Nice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.
Green($x, True) >>> Smart($x, True) ::: All green things are smart.
Big($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.
Green($x, True) >>> Big($x, True) ::: All green things are big.
Young(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.
Furry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.

Query:
Green(Harry, False) ::: Harry is not green.
""",
    "B",
))

# reference program 3 -> C (Likes(Lion, Cat, _) never derivable -> unknown)
DATALOG_GOLDEN.append((
    """Predicates:
Chases($x, $y, bool) ::: Does x chase y?
Rough($x, bool) ::: Is x rough?
Young($x, bool) ::: Is x young?
Needs($x, $y, bool) ::: Does x need y?
Green($x, bool) ::: Is x green?
Likes($x, $y, bool) ::: Does x like y?
Blue($x, bool) ::: Is x blue?
Round($x, bool) ::: Is x round?

Facts:
Chases(Cat, Lion, True) ::: The cat chases the lion.
Rough(Cat, True) ::: The cat is rough.
Young(Cat, True) ::: The cat is young.
Needs(Cat, Lion, True) ::: The cat needs the lion.
Needs(Cat, Rabbit, True) ::: The cat needs the rabbit.
Green(Dog, True) ::: The dog is green.
Young(Dog, True) ::: The dog is young.
Likes(Dog, Cat, True) ::: The dog likes the cat.
Blue(Lion, True) ::: The lion is blue.
Green(Lion, True) ::: The lion is green.
Chases(Rabbit, Lion, True) ::: The rabbit chases the lion.
Blue(Rabbit, True) ::: The rabbit is blue.
Rough(Rabbit, True) ::: The rabbit is rough.
Likes(Rabbit, Dog, True) ::: The rabbit likes the dog.
Needs(Rabbit, Dog, True) ::: The rabbit needs the dog.
Needs(Rabbit, Lion, True) ::: The rabbit needs the lion.

Rules:
Chases($x, Lion, True) >>> Round($x, True) ::: If someone chases the lion then they are round.
Needs(Lion, Rabbit, True) && Chases(Rabbit, Dog, True) >>> Likes(Lion, Dog, True) ::: If the lion needs the rabbit and the rabbit chases the dog then the lion likes the dog.
Round($x, True) && Chases($x, Lion, True) >>> Needs($x, Cat, True) ::: If someone is round and they chase the lion then they need the cat.
Needs($x, Cat, True) && Chases($x, Dog, True) >>> Likes($x, Rabbit, True) ::: If someone needs the cat and they chase the dog then they like the rabbit.
Chases($x, Lion, True) && Blue(Lion, True) >>> Round(Lion, True) ::: If someone chases the lion and the lion is blue then the lion is round.
Chases($x, Rabbit, True) >>> Rough($x, True) ::: If someone chases the rabbit then they are rough.
Rough($x, True) && Likes($x, Rabbit, True) >>> Young(Rabbit, True) ::: If someone is rough and they like the rabbit then the rabbit is young.
Chases(Rabbit, Cat, True) && Needs(Cat, Lion, True) >>> Young(Rabbit, True) ::: If the rabbit chases the cat and the cat needs the lion then the rabbit is young.
Round($x, True) && Needs($x, Cat, True) >>> Chases($x, Dog, True) ::: If someone is round and they need the cat then they chase the dog.

Query:
Likes(Lion, Cat, False) ::: The lion does not like the cat.
""",
    "C",
))

# reference program 1 -> A (Nice(Anne, True) derivable via Furry rule)
DATALOG_GOLDEN.append((
    """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?
Smart($x, bool) ::: Is x smart?
Young($x, bool) ::: Is x young?
Green($x, bool) ::: Is x green?
Big($x, bool) ::: Is x big?
Round($x, bool) ::: Is x round?

Facts:
Furry(Anne, True) ::: Anne is furry.
Nice(Anne, True) ::: Anne is nice.
Smart(Anne, True) ::: Anne is smart.
Young(Bob, True) ::: Bob is young.
Nice(Erin, True) ::: Erin is nice.
Smart(Harry, True) ::: Harry is smart.
Young(Harry, True) ::: Harry is young.

Rules:
Young($x, True) >>> Furry($x, True) ::: Young things are furry.
Nice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.
Green($x, True) >>> Nice($x, True) ::: All green things are nice.
Nice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.
Green($x, True) >>> Smart($x, True) ::: All green things are smart.
Big($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.
Green($x, True) >>> Big($x, True) ::: All green things are big.
Young(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.
Furry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.

Query:
Nice(Anne, True) ::: Anne is nice.
""",
    "A",
))

# reference program 5 -> B (Nice(Anne, False) refuted)
DATALOG_GOLDEN.append((
    """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?
Smart($x, bool) ::: Is x smart?
Young($x, bool) ::: Is x young?
Green($x, bool) ::: Is x green?
Big($x, bool) ::: Is x big?
Round($x, bool) ::: Is x round?

Facts:
Furry(Anne, True) ::: Anne is furry.
Nice(Anne, True) ::: Anne is nice.
Smart(Anne, True) ::: Anne is smart.
Young(Bob, True) ::: Bob is young.
Nice(Erin, True) ::: Erin is nice.
Smart(Harry, True) ::: Harry is smart.
Young(Harry, True) ::: Harry is young.

Rules:
Young($x, True) >>> Furry($x, True) ::: Young things are furry.
Nice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.
Green($x, True) >>> Nice($x, True) ::: All green things are nice.
Nice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.
Green($x, True) >>> Smart($x, True) ::: All green things are smart.
Big($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.
Green($x, True) >>> Big($x, True) ::: All green things are big.
Young(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.
Furry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.

Query:
Nice(Anne, False) ::: Anne is not nice.
""",
    "B",
))

# reference program 6 -> C (Furry(Anne, False) absent -> rule never fires)
DATALOG_GOLDEN.append((
    """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?

Facts:
Furry(Anne, True) ::: Anne is furry.

Rules:
Furry($x, False) >>> Nice($x, True) ::: All non-furry things are nice.

Query:
Nice(Anne, True) ::: Anne is nice.
""",
    "C",
))

# reference program 7 -> B (Green(Harry, False) refuted)
DATALOG_GOLDEN.append((
    """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?
Smart($x, bool) ::: Is x smart?
Young($x, bool) ::: Is x young?
Green($x, bool) ::: Is x green?
Big($x, bool) ::: Is x big?
Round($x, bool) ::: Is x round?

Facts:
Furry(Anne, True) ::: Anne is furry.
Nice(Anne, True) ::: Anne is nice.
Smart(Anne, True) ::: Anne is smart.
Young(Bob, True) ::: Bob is young.
Nice(Erin, True) ::: Erin is nice.
Smart(Harry, True) ::: Harry is smart.
Young(Harry, True) ::: Harry is young.

Rules:
Young($x, True) >>> Furry($x, True) ::: Young things are furry.
Nice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.
Green($x, True) >>> Nice($x, True) ::: All green things are nice.
Nice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.
Green($x, True) >>> Smart($x, True) ::: All green things are smart.
Big($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.
Green($x, True) >>> Big($x, True) ::: All green things are big.
Young(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.
Furry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.

Query:
Green(Harry, False) ::: Harry is not green.
""",
    "B",
))
