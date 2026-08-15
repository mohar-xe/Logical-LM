"""End-to-end pipeline tests using the offline MockClient and synthetic data."""

import pytest

from logiclm.pipeline import LogicLMPipeline
from logiclm.schema import Example
from logiclm.solvers.registry import SolverRegistry
from logiclm.evaluate import compute_metrics
from logiclm.llm.mock import MockClient

GOOD_PROGRAM = """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?

Facts:
Furry(Anne, True) ::: Anne is furry.

Rules:
Furry($x, True) >>> Nice($x, True) ::: All furry things are nice.

Query:
Nice(Anne, True) ::: Anne is nice.
"""

BAD_PROGRAM = "this is not a valid logic program at all"


def _example(example_id="e1", answer="A", program=GOOD_PROGRAM):
    return Example(
        id=example_id,
        context="Anne is furry. All furry things are nice.",
        question="Is Anne nice?",
        answer=answer,
        options=["A) True", "B) False", "C) Unknown"],
        programs=[program],
    )


def test_pipeline_good_program_succeeds():
    llm = MockClient({})
    solver = SolverRegistry()
    pipe = LogicLMPipeline("ProofWriter", llm=llm, solver=solver)
    results = pipe.infer([_example()])
    assert results[0]["flag"] == "success"
    assert results[0]["predicted_answer"] == "A"


def test_pipeline_bad_program_uses_random_backup():
    llm = MockClient({})
    solver = SolverRegistry()
    pipe = LogicLMPipeline("ProofWriter", llm=llm, solver=solver, seed=7)
    results = pipe.infer([_example(program=BAD_PROGRAM)])
    assert results[0]["flag"] != "success"
    assert results[0]["predicted_answer"] in ("A", "B", "C")


def test_pipeline_llm_backup_uses_cot_results():
    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"id": "e1", "predicted_answer": "B"}], f)
        path = f.name
    llm = MockClient({})
    solver = SolverRegistry()
    pipe = LogicLMPipeline(
        "ProofWriter", llm=llm, solver=solver,
        backup_strategy="LLM", backup_llm_path=path,
    )
    results = pipe.infer([_example(program=BAD_PROGRAM)])
    assert results[0]["predicted_answer"] == "B"


def test_generate_programs_populates_from_llm():
    llm = MockClient({"Anne is furry": GOOD_PROGRAM})
    solver = SolverRegistry()
    pipe = LogicLMPipeline("ProofWriter", llm=llm, solver=solver)
    ex = Example(id="e1", context="Anne is furry.", question="Is Anne nice?",
                 answer="A", options=["A) True", "B) False", "C) Unknown"])
    pipe.generate_programs([ex])
    assert ex.programs == [GOOD_PROGRAM.strip()]


def test_full_run_generation_inference():
    llm = MockClient({"Anne is furry": GOOD_PROGRAM})
    solver = SolverRegistry()
    pipe = LogicLMPipeline("ProofWriter", llm=llm, solver=solver)
    ex = Example(id="e1", context="Anne is furry.", question="Is Anne nice?",
                 answer="A", options=["A) True", "B) False", "C) Unknown"])
    results = pipe.run([ex])
    assert results[0]["flag"] == "success"
    assert results[0]["predicted_answer"] == "A"


def test_metrics():
    results = [
        {"answer": "A", "predicted_answer": "A", "flag": "success"},
        {"answer": "B", "predicted_answer": "B", "flag": "success"},
        {"answer": "C", "predicted_answer": "A", "flag": "success"},
        {"answer": "A", "predicted_answer": "C", "flag": "parsing error"},
    ]
    m = compute_metrics(results)
    assert m.accuracy == 0.5
    assert m.executable_rate == 0.75
    assert m.executable_accuracy == 2 / 3


def test_folio_pipeline():
    program = """Premises:
    Animal(felix) ::: Felix is an animal.
    forall x (Animal(x) -> Living(x)) ::: All animals are living things.
    Conclusion:
    Living(felix) ::: Felix is living.
    """
    llm = MockClient({"Felix is an animal": program})
    solver = SolverRegistry()
    pipe = LogicLMPipeline("FOLIO", llm=llm, solver=solver)
    ex = Example(id="f1", context="Felix is an animal.", question="Is Felix living?",
                 answer="A", options=["A) True", "B) False", "C) Unknown"])
    results = pipe.run([ex])
    assert results[0]["flag"] == "success"
    assert results[0]["predicted_answer"] == "A"
