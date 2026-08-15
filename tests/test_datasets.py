"""Tests for the data-loading layer (offline samples + network-gated fetch)."""

import os

import pytest

from logiclm.datasets import load_dataset, download_all
from logiclm.schema import Example

SAMPLES = {"ProntoQA", "ProofWriter", "FOLIO", "LogicalDeduction", "AR-LSAT"}


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_samples_load_offline(name):
    examples = load_dataset(name, split="samples")
    assert len(examples) >= 1
    for ex in examples:
        assert isinstance(ex, Example)
        assert ex.id
        assert ex.context
        assert ex.question
        assert ex.answer in {"A", "B", "C", "D", "E"}


def test_samples_have_canned_programs():
    ex = load_dataset("ProntoQA", split="samples")[0]
    assert ex.programs, "samples should bundle a ready-to-run program"


@pytest.mark.network
def test_fetch_prontoqa():
    examples = load_dataset("ProntoQA", split="dev", data_dir="/tmp/logiclm-data-test")
    assert len(examples) >= 1
    assert examples[0].answer in {"A", "B"}


@pytest.mark.network
def test_download_all(tmp_path):
    counts = download_all(data_dir=str(tmp_path))
    assert counts["ProntoQA"] >= 1
    assert counts["FOLIO"] >= 1


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("Nope", split="dev")


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        load_dataset("ProntoQA", split="bogus")
