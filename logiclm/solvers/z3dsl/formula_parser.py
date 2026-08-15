"""Recursive-descent parser for AR-LSAT DSL formulas.

The DSL formula grammar (after ``normalize_text``) mixes:

* boolean combinators: ``And(...)``, ``Or(...)``, ``Not(...)``, ``Implies(...)``,
  ``Iff(...)``, ``Xor(...)``
* quantifiers: ``ForAll([m:meals], ...)``, ``Exists([c:children], ...)``
* count: ``Count([m:meals], cond)``
* distinct: ``Distinct([a:sort, b:sort])`` (scoped) or ``Distinct([a, b, c])``
* function applications: ``eats(Vladimir, m)``
* comparisons: ``== != < > <= >=``, arithmetic ``+ - * /``
* ``is_valid(...)`` / ``is_sat(...)`` wrappers used in options (semantics live
  in the compiler; the wrapper returns its inner formula)

Names in *argument* position resolve to a bound ``FVar`` when they appear in
the quantifier scope in force, else to a ``FConst`` (a domain member or a
ground constant).  A name whose declared sort/function is unknown is treated
as a constant.
"""

from __future__ import annotations

from ..errors import ParseError
from .ast import (
    FApp, FArith, FBoolOp, FCmp, FConst, FCount, FDistinct, FQuant, FVar,
)

_BOOLOPS = {"And", "Or", "Not", "Implies", "Iff", "Xor"}
_UNARY_BOOLOPS = {"Not"}
_COMPARATORS = {"==", "!=", "<", ">", "<=", ">="}
_ARITH = {"+", "-", "*", "/"}
_QUANTIFIERS = {"ForAll", "Exists"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Return (kind, value) pairs; kind is 'name' | 'int' | 'symbol'."""
    tokens: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isalnum() or ch == "_":
            buf.append(ch)
            i += 1
            continue
        if buf:
            tokens.append(("name", "".join(buf)))
            buf = []
        two = text[i : i + 2]
        if two in {"==", "!=", "<=", ">=", "->", "<->"}:
            tokens.append(("symbol", two))
            i += 2
            continue
        if ch in "()[],:+-*/<>=":
            tokens.append(("symbol", ch))
            i += 1
            continue
        if not ch.isspace():
            raise ParseError(f"unexpected character {ch!r} in DSL formula")
        i += 1
    if buf:
        tokens.append(("name", "".join(buf)))
    return tokens


class FormulaParser:
    def __init__(self, text: str, prog):
        self.tokens = _tokenize(text)
        self.pos = 0
        self.prog = prog
        self._bound: set[str] = set()

    # -- token stream -----------------------------------------------------

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek_kind_value(self, kind: str, value: str) -> bool:
        tok = self.peek()
        return bool(tok and tok[0] == kind and tok[1] == value)

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of DSL formula")
        self.pos += 1
        return tok

    def accept(self, value: str) -> bool:
        if self.peek_kind_value("symbol", value):
            self.pos += 1
            return True
        return False

    def _expect_open(self) -> None:
        if not self.accept("("):
            raise ParseError(f"expected '(' in DSL formula, got {self.peek()!r}")

    def _expect_close(self) -> None:
        if not self.accept(")"):
            raise ParseError(f"expected ')' in DSL formula, got {self.peek()!r}")

    def _expect_list_open(self) -> None:
        if not self.accept("["):
            raise ParseError(f"expected '[' in DSL formula, got {self.peek()!r}")

    def _expect_list_close(self) -> None:
        if not self.accept("]"):
            raise ParseError(f"expected ']' in DSL formula, got {self.peek()!r}")

    def _expect_name(self, what: str) -> str:
        tok = self.peek()
        if tok is None or tok[0] != "name":
            raise ParseError(f"expected {what}, got {tok!r}")
        self.pos += 1
        return tok[1]

    # -- entry point ------------------------------------------------------

    def parse(self) -> Formula:
        formula = self._parse_formula()
        if self.peek() is not None:
            raise ParseError(f"trailing tokens in DSL formula: {self.peek()!r}")
        return formula

    # -- grammar ----------------------------------------------------------

    def _parse_formula(self) -> Formula:
        tok = self.peek()
        if tok is None:
            raise ParseError("empty DSL formula")
        if tok[0] == "name" and tok[1] in {"is_valid", "is_sat"}:
            self.pos += 1
            self._expect_open()
            inner = self._parse_formula()
            self._expect_close()
            return inner
        return self._parse_expr()

    def _parse_expr(self) -> Formula:
        tok = self.peek()
        if tok is None:
            raise ParseError("empty expression")
        if tok[0] == "name":
            name = tok[1]
            if name in _BOOLOPS:
                return self._parse_boolop(name)
            if name in _QUANTIFIERS:
                return self._parse_quant(name)
            if name == "Count":
                # Count(...) participates in comparison/arith ops (>= 4).
                return self._parse_primary_with_ops_from(self._parse_count)
            if name == "Distinct":
                return self._parse_distinct()
            if name in {"True", "False"}:
                self.pos += 1
                return FBoolOp(name, ())
            if self.peek_kind_value("symbol", "("):
                return self._parse_app()
        return self._parse_primary_with_ops()

    def _parse_boolop(self, name: str) -> Formula:
        self.pos += 1
        self._expect_open()
        args = [self._parse_expr()]
        while self.accept(","):
            args.append(self._parse_expr())
        self._expect_close()
        if name == "Not" and len(args) != 1:
            raise ParseError("Not() takes exactly one argument")
        return FBoolOp(name, tuple(args))

    def _parse_quant(self, qname: str) -> Formula:
        self.pos += 1
        self._expect_open()
        self._expect_list_open()
        scopes = self._parse_scopes()
        self._expect_list_close()
        self.accept(",")
        self._bound |= {v for v, _ in scopes}
        body = self._parse_expr()
        self._bound -= {v for v, _ in scopes}
        self._expect_close()
        return FQuant(qname, scopes, body)

    def _parse_count(self) -> Formula:
        self.pos += 1
        self._expect_open()
        self._expect_list_open()
        scopes = self._parse_scopes()
        self._expect_list_close()
        self.accept(",")
        self._bound |= {v for v, _ in scopes}
        cond = self._parse_expr()
        self._bound -= {v for v, _ in scopes}
        self._expect_close()
        return FCount(scopes, cond)

    def _parse_scopes(self) -> tuple[tuple[str, str], ...]:
        scopes: list[tuple[str, str]] = []
        while True:
            var = self._expect_name("a bound variable")
            if not self.accept(":"):
                raise ParseError("expected ':' after bound variable")
            scope = self._expect_name("a sort name")
            scopes.append((var, scope))
            if not self.accept(","):
                break
        return tuple(scopes)

    def _parse_distinct(self) -> Formula:
        self.pos += 1
        self._expect_open()
        self._expect_list_open()
        # scoped form: Distinct([a:sort, b:sort], expr) ; plain form: Distinct([a, b, c])
        scopes: list[tuple[str, str]] = []
        elems: list[str] = []
        tok = self.peek()
        scoped = False
        if tok and tok[0] == "name" and self._lookahead_is(":", 1):
            scoped = True
            while True:
                var = self._expect_name("a bound variable")
                self.accept(":")
                scope = self._expect_name("a sort name")
                scopes.append((var, scope))
                if not self.accept(","):
                    break
        else:
            while True:
                elems.append(self._expect_name("a name"))
                if not self.accept(","):
                    break
        self._expect_list_close()
        if scoped:
            self.accept(",")
            self._bound |= {v for v, _ in scopes}
            expr = self._parse_expr()
            self._bound -= {v for v, _ in scopes}
            self._expect_close()
            return FDistinct(tuple(scopes), expr)
        self._expect_close()
        return FDistinct((), tuple(elems))

    def _lookahead_is(self, value: str, offset: int) -> bool:
        idx = self.pos + offset
        if idx < len(self.tokens):
            k, v = self.tokens[idx]
            return k == "symbol" and v == value
        return False

    def _parse_app(self) -> Formula:
        name = self._expect_name("a function name")
        self._expect_open()
        args = [self._parse_term()]
        while self.accept(","):
            args.append(self._parse_term())
        self._expect_close()
        return FApp(name, tuple(args))

    def _parse_term(self) -> Formula:
        """A term inside a function application: constant / var / number."""
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of term")
        if tok[0] == "int":
            self.pos += 1
            return FConst(tok[1])
        if tok[0] == "name":
            name = tok[1]
            self.pos += 1
            if name in self._bound:
                return FVar(name)
            return FConst(name)
        raise ParseError(f"unexpected token in term: {tok!r}")

    # -- primary expression with comparison/arithmetic ----------------------

    def _parse_primary_with_ops_from(self, parse_primary) -> Formula:
        left = parse_primary()
        return self._continue_ops(left)

    def _continue_ops(self, left: Formula) -> Formula:
        while True:
            tok = self.peek()
            if tok is None or tok[0] != "symbol":
                break
            op = tok[1]
            if op in _COMPARATORS:
                self.pos += 1
                right = self._parse_primary()
                left = FCmp(op, left, right)
            elif op in _ARITH:
                self.pos += 1
                right = self._parse_primary()
                left = FArith(op, left, right)
            else:
                break
        return left

    def _parse_primary_with_ops(self) -> Formula:
        left = self._parse_primary()
        return self._continue_ops(left)

    def _parse_primary(self) -> Formula:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end in primary expression")
        if tok[0] == "int":
            self.pos += 1
            return FConst(tok[1])
        if tok[0] == "symbol" and tok[1] == "(":
            self.pos += 1
            inner = self._parse_expr()
            self._expect_close()
            return inner
        if tok[0] == "name":
            name = tok[1]
            if self._lookahead_is("(", 1):
                return self._parse_app()
            self.pos += 1
            if name in self._bound:
                return FVar(name)
            return FConst(name)
        raise ParseError(f"unexpected token in primary expression: {tok!r}")


def parse_formula(text: str, prog) -> Formula:
    """Parse a DSL formula string into the DSL AST."""
    return FormulaParser(text, prog).parse()
