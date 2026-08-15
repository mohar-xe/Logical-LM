"""Offline end-to-end: every bundled sample runs through the full pipeline
(generation via MockClient -> inference -> evaluation) and must match its
gold answer."""

import pytest

from logiclm.datasets import load_dataset
from logiclm.evaluate import compute_metrics
from logiclm.llm.mock import MockClient
from logiclm.pipeline import LogicLMPipeline
from logiclm.solvers.registry import SolverRegistry

ALL_DATASETS = ("ProntoQA", "ProofWriter", "FOLIO", "LogicalDeduction", "AR-LSAT")


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_sample_pipeline_end_to_end(name):
    examples = load_dataset(name, split="samples")
    assert examples, f"no samples for {name}"

    # Key the mock on the FULL question (unique per sample) -> canned program.
    responses = {}
    for ex in examples:
        assert ex.programs, f"{name} sample {ex.id} has no canned program"
        responses[ex.question] = ex.programs[0]

    pipe = LogicLMPipeline(
        name,
        llm=MockClient(responses),
        solver=SolverRegistry(timeout_ms=5000),
    )
    results = pipe.run(examples)
    m = compute_metrics(results)

    # Every sample's program must execute successfully and answer correctly.
    assert m.executable_rate == 1.0, (
        f"{name}: not all samples executed: "
        f"{[r['flag'] for r in results]}"
    )
    assert m.accuracy == 1.0, (
        f"{name}: answers mismatch gold: "
        f"{[(r['id'], r['predicted_answer'], r['answer']) for r in results]}"
    )
