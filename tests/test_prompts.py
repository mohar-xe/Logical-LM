"""Tests for prompt-template loading and filling."""

import pytest

from logiclm.prompts import PromptLibrary


@pytest.fixture(scope="module")
def lib():
    return PromptLibrary()


def test_generation_templates_exist(lib):
    for ds in ("ProntoQA", "ProofWriter", "FOLIO", "LogicalDeduction", "AR-LSAT"):
        t = lib.generation_template(ds)
        assert "[[PROBLEM]]" in t
        assert "[[QUESTION]]" in t


def test_choices_placeholder_present_for_csp_and_arlsat(lib):
    for ds in ("LogicalDeduction", "AR-LSAT"):
        assert "[[CHOICES]]" in lib.generation_template(ds)


def test_self_correct_templates_exist(lib):
    for ds in ("FOLIO", "AR-LSAT"):
        t = lib.self_correct_template(ds)
        assert "[[PROGRAM]]" in t
        assert "[[ERROR MESSAGE]]" in t


def test_build_generation_prompt(lib):
    example = {
        "context": "Alex is a tumpus.",
        "question": "True or false: Alex is not shy.",
        "options": [],
    }
    prompt = lib.build_generation_prompt("ProntoQA", example)
    assert "Alex is a tumpus." in prompt
    assert "True or false: Alex is not shy." in prompt
    assert "[[PROBLEM]]" not in prompt  # placeholders all filled


def test_build_generation_prompt_with_choices(lib):
    example = {
        "context": "Three birds.",
        "question": "Which of the following is true?",
        "options": ["A) quail == 5", "B) owl == 5"],
    }
    prompt = lib.build_generation_prompt("LogicalDeduction", example)
    assert "A) quail == 5" in prompt
    assert "B) owl == 5" in prompt


def test_build_self_correct_prompt(lib):
    program = "Premises:\nP(a)\nConclusion:\nQ(a)"
    err = "Parsing Error"
    prompt = lib.build_self_correct_prompt("FOLIO", program, err)
    assert program in prompt
    assert err in prompt
    assert "[[PROGRAM]]" not in prompt
    assert "[[ERROR MESSAGE]]" not in prompt


def test_unknown_dataset_raises(lib):
    with pytest.raises(ValueError):
        lib.generation_template("Nope")
