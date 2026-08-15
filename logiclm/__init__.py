"""Logic-LM: LLMs + symbolic solvers for faithful logical reasoning.

A from-scratch reimplementation of the Logic-LM framework
(arXiv:2305.12295). An LLM translates natural-language problems into
symbolic logic programs; deterministic symbolic solvers execute them;
a self-refinement loop revises faulty programs from solver error messages.
"""

__version__ = "0.1.0"

# Datasets supported end-to-end and the solver backend each uses.
DATASETS = ("ProntoQA", "ProofWriter", "FOLIO", "LogicalDeduction", "AR-LSAT")

# Letter space of answers per dataset (also used for the random backup).
# LogicalDeduction has 3/5/7-object variants -> up to 7 options (A..G).
ANSWER_SPACE = {
    "ProntoQA": ("A", "B"),
    "ProofWriter": ("A", "B", "C"),
    "FOLIO": ("A", "B", "C"),
    "LogicalDeduction": ("A", "B", "C", "D", "E", "F", "G"),
    "AR-LSAT": ("A", "B", "C", "D", "E"),
}
