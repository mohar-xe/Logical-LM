"""Unicode normalization for logical symbols produced by LLMs.

LLMs emit a zoo of look-alike symbols for the same logical operator
(``→``/``⟹``/``⇒``/``->``, ``∧``/``⋀``, ``¬``/``~``, ``∀``/``forall``...),
plus stray non-breaking spaces and smart quotes.  ``normalize_text``
canonicalizes the *symbol* spellings (padding each with spaces) so both the
FOL and Z3-DSL lexers only ever need to recognize one symbol per operator.

Word-form operators (``and``, ``or``, ``not``, ``forall``, ``exists``,
``xor``) and function-style calls (``And(...)``, ``ForAll(...)`` ... used by
the AR-LSAT DSL) are intentionally left alone: the lexers accept those
natively, and rewriting them here would mangle identifiers like ``Andrew``.
"""

from __future__ import annotations

import re
import unicodedata

# (source symbol -> canonical token). Canonical tokens are all symbols (never
# bare ASCII words) so substitution can never corrupt an identifier.
_OPERATOR_MAP: dict[str, str] = {
    # negation
    "¬": "~",
    # conjunction
    "∧": "&&",
    "⋀": "&&",
    # disjunction
    "∨": "||",
    "⋁": "||",
    # implication
    "→": "->",
    "⟶": "->",
    "⟹": "->",
    "⇒": "->",
    "⭢": "->",
    # biconditional
    "↔": "<->",
    "⇔": "<->",
    # exclusive-or
    "⊕": "xor",
    # quantifiers
    "∀": "forall",
    "∃": "exists",
    # ASCII symbol spellings
    "&&": "&&",
    "||": "||",
    "->": "->",
    "<->": "<->",
    "=>": "->",
    "<=>": "<->",
    "~": "~",
    # NOTE: "!" is deliberately NOT mapped to "~": the Z3-DSL uses "!="
    # heavily and "!x" is vanishingly rare in these corpora.
}

# Look-alike characters that must become ASCII punctuation (not operators).
_CHAR_REPLACEMENTS = {
    " ": " ",  # NO-BREAK SPACE
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "​": "",   # ZERO WIDTH SPACE
    "﻿": "",   # ZERO WIDTH NO-BREAK SPACE
}

# Sort by length desc so longer symbols (``<->``, ``⟹``) win over their
# prefixes. Every alternation member is a symbol, never a bare ASCII word.
_OPERATOR_PATTERN = re.compile(
    "(" + "|".join(sorted(map(re.escape, _OPERATOR_MAP), key=len, reverse=True)) + ")"
)


def normalize_text(text: str) -> str:
    """Canonicalize a formula string for the lexers.

    * NFKC fold (decomposes ligatures, full-width forms, minus-sign).
    * Replace smart quotes / dashes / NBSPs with ASCII equivalents.
    * Substitute every operator symbol with ``' token '`` (space-padded), so
      ``∀x`` and ``∀ x`` tokenize identically.
    * Collapse runs of whitespace to single spaces.
    """
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = _OPERATOR_PATTERN.sub(lambda m: f" {_OPERATOR_MAP[m.group(0)]} ", text)
    return " ".join(text.split())


def canonical_operator_map() -> dict[str, str]:
    """Expose the operator map (used by tests)."""
    return dict(_OPERATOR_MAP)
