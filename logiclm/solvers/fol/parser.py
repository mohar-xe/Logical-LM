"""Recursive-descent parser for the FOLIO first-order-logic grammar.

Grammar (canonical tokens after ``unicode.normalize_text``):

    formula  := equiv
    equiv    := impl  ( "<->"  impl )*
    impl     := xor   ( "->"   xor  )*
    xor      := or    ( "xor"  or   )*
    or       := and   ( "||"   and  )*
    and      := unary ( "&&"   unary)*
    unary    := "~" unary | quant | atom
    quant    := ("forall"|"exists") NAME formula
    atom     := "(" formula ")" | NAME "(" terms ")" | NAME
    terms    := term ("," term)*
    term     := NAME

A name in *term position* is parsed as a ``Variable`` when it is bound by an
enclosing quantifier, else as a ``Constant``.  Predicate names are never
variables.  Quantifiers bind the maximal rest of the current precedence level
(``forall x A -> B`` == ``forall x (A -> B)``).
"""

from __future__ import annotations

from .ast import Binary, Constant, Formula, Negation, Predicate, Quantifier, Variable
from ..errors import ParseError


class Lexer:
    __slots__ = ("tokens", "pos")

    # Multi-character operator tokens the lexer recognizes atomically.
    _MULTI = ("<->", "->", "&&", "||")

    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        buf: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            matched = False
            for op in self._MULTI:
                if text.startswith(op, i):
                    if buf:
                        tokens.append("".join(buf))
                        buf.clear()
                    tokens.append(op)
                    i += len(op)
                    matched = True
                    break
            if matched:
                continue
            if ch.isalnum() or ch == "_" or ch == "$":
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf))
                    buf.clear()
                if not ch.isspace():
                    tokens.append(ch)
            i += 1
        if buf:
            tokens.append("".join(buf))
        return tokens

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of input")
        self.pos += 1
        return tok

    def accept(self, tok: str) -> bool:
        if self.peek() == tok:
            self.pos += 1
            return True
        return False

    def expect(self, tok: str) -> None:
        if not self.accept(tok):
            raise ParseError(f"expected {tok!r}, got {self.peek()!r}")


class Parser:
    __slots__ = ("lexer", "predicates", "_bound")

    def __init__(self, text: str, predicates: set[str] | None = None):
        self.lexer = Lexer(text)
        self.predicates = predicates or set()
        self._bound: set[str] = set()

    # -- entry point ------------------------------------------------------

    def parse(self) -> Formula:
        formula = self._parse_equiv()
        if self.lexer.peek() is not None:
            raise ParseError(f"trailing tokens: {self.lexer.peek()!r}")
        return formula

    # -- grammar productions ----------------------------------------------

    def _parse_equiv(self) -> Formula:
        left = self._parse_impl()
        while self.lexer.accept("<->"):
            left = Binary("<->", left, self._parse_impl())
        return left

    def _parse_impl(self) -> Formula:
        left = self._parse_xor()
        while self.lexer.accept("->"):
            left = Binary("->", left, self._parse_xor())
        return left

    def _parse_xor(self) -> Formula:
        left = self._parse_or()
        while self.lexer.accept("xor"):
            left = Binary("xor", left, self._parse_or())
        return left

    def _parse_or(self) -> Formula:
        left = self._parse_and()
        while self.lexer.accept("||"):
            left = Binary("or", left, self._parse_and())
        return left

    def _parse_and(self) -> Formula:
        left = self._parse_unary()
        while self.lexer.accept("&&"):
            left = Binary("and", left, self._parse_unary())
        return left

    def _parse_unary(self) -> Formula:
        if self.lexer.accept("~"):
            return Negation(self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> Formula:
        tok = self.lexer.peek()
        if tok == "(":
            self.lexer.next()
            inner = self._parse_equiv()
            self.lexer.expect(")")
            return inner
        if tok == "forall":
            self.lexer.next()
            return self._parse_quantifier("forall")
        if tok == "exists":
            self.lexer.next()
            return self._parse_quantifier("exists")
        if tok is None:
            raise ParseError("unexpected end of input in atom")
        # predicate/constant: NAME | NAME(terms)
        name = self.lexer.next()
        if self.lexer.accept("("):
            args = [self._parse_term()]
            while self.lexer.accept(","):
                args.append(self._parse_term())
            self.lexer.expect(")")
            return Predicate(name, tuple(args))
        if name in self.predicates:
            return Predicate(name, ())
        return self._parse_term_name(name)

    def _parse_quantifier(self, qname: str) -> Formula:
        var = self.lexer.next()
        # a quantifier variable must be a plain name token, not an operator
        if var in self.lexer._MULTI or var in {"forall", "exists", "and", "or", "xor", "~"}:
            raise ParseError(f"expected a variable name, got {var!r}")
        self._bound.add(var)
        try:
            body = self._parse_equiv()
        finally:
            self._bound.discard(var)
        return Quantifier(qname, var, body)

    def _parse_term(self) -> Formula:
        tok = self.lexer.next()
        if tok in self.predicates:
            return Predicate(tok, ())
        return self._parse_term_name(tok)

    def _parse_term_name(self, name: str) -> Formula:
        # Every name in term position is a Variable node; bound-ness is
        # decided at compile time (bound -> quantified var, unbound -> ground
        # constant symbol of the domain sort).
        return Variable(name)


def parse_formula(text: str, predicates: set[str] | None = None) -> Formula:
    """Parse a normalized FOL formula string into an AST."""
    return Parser(text, predicates).parse()
