"""Golden FOLIO logic programs adapted from the reference repo's
``prover9_solver.py.__main__`` — these are valid FOLIO IR with known answers.
Used as offline gold tests for the FOL parser and Z3 solver.
"""

from __future__ import annotations

# (program text, expected verdict) — expected is the SOLVER verdict vocabulary
# ("True"/"False"/"Unknown"), which maps to A/B/C.
FOLIO_GOLDEN: list[tuple[str, str]] = []

FOLIO_GOLDEN.append((
    """Premises:
    ~forall x (Movie(x) -> HappyEnding(x)) ::: Not all movies have a happy ending.
    Movie(titanic) ::: Titanic is a movie.
    ~HappyEnding(titanic) ::: Titanic does not have a happy ending.
    Movie(lionKing) ::: Lion King is a movie.
    HappyEnding(lionKing) ::: Lion King has a happy ending.
    Conclusion:
    exists x (Movie(x) && ~HappyEnding(x)) ::: Some movie does not have a happy ending.
    """,
    "True",
))

FOLIO_GOLDEN.append((
    """Premises:
    forall x (Drinks(x) -> Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    forall x (Drinks(x) xor Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    forall x (Jokes(x) -> ~Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug.
    (Student(rina) && Unaware(rina)) xor ~(Student(rina) || Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug.
    ~(Dependent(rina) && Student(rina)) -> (Dependent(rina) && Student(rina)) xor ~(Dependent(rina) || Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Conclusion:
    Jokes(rina) xor Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
    """,
    "True",
))

FOLIO_GOLDEN.append((
    """Premises:
    Czech(miroslav) && ChoralConductor(miroslav) && Specialize(miroslav, renaissance) && Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    forall x (ChoralConductor(x) -> Musician(x)) ::: Any choral conductor is a musician.
    exists x (Musician(x) && Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) && Author(miroslav, methodOfStudyingGregorianChant) && Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    Love(miroslav, music) ::: Miroslav Venhoda loved music.
    """,
    "Unknown",
))

FOLIO_GOLDEN.append((
    """Premises:
    Czech(miroslav) && ChoralConductor(miroslav) && Specialize(miroslav, renaissance) && Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    forall x (ChoralConductor(x) -> Musician(x)) ::: Any choral conductor is a musician.
    exists x (Musician(x) && Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) && Author(miroslav, methodOfStudyingGregorianChant) && Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    exists y exists x (Czech(x) && Author(x, y) && Book(y) && Publish(y, year1946)) ::: A Czech person wrote a book in 1946.
    """,
    "True",
))

FOLIO_GOLDEN.append((
    """Premises:
    Czech(miroslav) && ChoralConductor(miroslav) && Specialize(miroslav, renaissance) && Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    forall x (ChoralConductor(x) -> Musician(x)) ::: Any choral conductor is a musician.
    exists x (Musician(x) && Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) && Author(miroslav, methodOfStudyingGregorianChant) && Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    ~exists x (ChoralConductor(x) && Specialize(x, renaissance)) ::: No choral conductor specialized in the performance of Renaissance.
    """,
    "False",
))

FOLIO_GOLDEN.append((
    """Premises:
    forall x (Movie(x) -> HappyEnding(x)) ::: Not all movies have a happy ending.
    Movie(titanic) ::: Titanic is a movie.
    HappyEnding(titanic) ::: Titanic has a happy ending.
    Movie(lionKing) ::: Lion King is a movie.
    HappyEnding(lionKing) ::: Lion King has a happy ending.
    Conclusion:
    exists x (Movie(x) && ~HappyEnding(x)) ::: Some movie does not have a happy ending.
    """,
    "False",
))

FOLIO_GOLDEN.append((
    """Premises:
    forall x (Person(x) -> (Tall(x) || Short(x))) ::: Every person is either tall or short.
    Person(john) ::: John is a person.
    Conclusion:
    Tall(john) ::: John is tall.
    """,
    "Unknown",
))

# A deliberately unprovable (but satisfiable-against) formula: the conclusion
# contradicts the premises — refuted.
FOLIO_GOLDEN.append((
    """Premises:
    forall x (Cat(x) -> Animal(x)) ::: All cats are animals.
    Cat(tom) ::: Tom is a cat.
    Conclusion:
    ~Animal(tom) ::: Tom is not an animal.
    """,
    "False",
))

# Not from the reference: premises entail the conclusion by modus ponens.
FOLIO_GOLDEN.append((
    """Premises:
    Animal(felix) ::: Felix is an animal.
    forall x (Animal(x) -> Living(x)) ::: All animals are living things.
    Conclusion:
    Living(felix) ::: Felix is living.
    """,
    "True",
))
