from __future__ import annotations
import re
from typing import Any

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

_NORMALISE = str.maketrans({
    "∀": "forall ", "∃": "exists ", "→": "->", "↔": "<->",
    "∧": " and ", "∨": " or ", "¬": "not ", "⊑": "subclass",
    "≡": "<->", "⊂": "subclass",
    "⟹": "->", "⟺": "<->",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "α": "a", "β": "b", "γ": "c",
    "≠": " neq ", "∈": " in ", "∉": " notin ",
    "≤": " leq ", "≥": " geq ", "⊂": " subset ", "⊆": " subseteq ",
})


_SCAFFOLD_LINE = re.compile(r"^\s*[*_#\\\[( ]*(step\s*\d|branch\s*[abc])\b", re.IGNORECASE)
_UNFILLED_SLOT = re.compile(r"\[conditions\]|\{label\}|\.\.\.|…")


def _is_fol_candidate(line: str) -> bool:
    low = line.lower()
    return ("forall" in low or "for all" in low or "exists " in low
            or "<->" in line or "->" in line)


def extract_fol_block(text: str) -> str | None:
    text_n = text.translate(_NORMALISE)
    candidates = [ln.strip() for ln in text_n.splitlines() if _is_fol_candidate(ln.strip())]
    usable = []
    for line in candidates:
        stripped = _SCAFFOLD_LINE.sub("", line).lstrip(" :.)-*_#").strip()
        if not stripped or not _is_fol_candidate(stripped):
            continue
        if _UNFILLED_SLOT.search(stripped):
            continue
        usable.append(stripped)
    if not usable:
        return None
    biconditionals = [ln for ln in usable if "<->" in ln]
    return biconditionals[-1] if biconditionals else usable[-1]


_CONTR_AND = re.compile(
    r"(\w+)\(x\)\s+and\s+not\s+\1\(x\)"
    r"|not\s+(\w+)\(x\)\s+and\s+\2\(x\)",
    re.IGNORECASE,
)

def _is_self_negation(block: str, concept_name: str) -> bool:
    name = re.escape(_sanitise(concept_name))
    pat = re.compile(rf"\b{name}\s*\(x\)\s*<->\s*not\s+{name}\s*\(x\)", re.IGNORECASE)
    return bool(pat.search(block))


class _ParseError(Exception):
    pass


def _sanitise(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).strip("_") or "Pred"


class _Z3Builder:

    def __init__(self) -> None:
        self._sort: Any = None
        self._fns: dict[str, Any] = {}

    def _sort_(self) -> Any:
        if self._sort is None:
            self._sort = z3.DeclareSort("Entity")
        return self._sort

    def _var(self, name: str) -> Any:
        return z3.Const(name, self._sort_())

    def _fn(self, name: str, arity: int) -> Any:
        if name not in self._fns:
            if arity == 1:
                self._fns[name] = z3.Function(name, self._sort_(), z3.BoolSort())
            else:
                self._fns[name] = z3.Function(
                    name, *([self._sort_()] * arity), z3.BoolSort()
                )
        return self._fns[name]


    _TOKEN_RE = re.compile(
        r"(?P<KW>forall|exists|and|or|not|true|false)"
        r"|(?P<IFF><->)"
        r"|(?P<IMP>->)"
        r"|(?P<LP>\()"
        r"|(?P<RP>\))"
        r"|(?P<COMMA>,)"
        r"|(?P<COLON>:)"
        r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_ ]*?(?=\s*[\(,\)\s:and\sor\snot<\-]|$))"
        r"|(?P<WS>\s+)",
        re.IGNORECASE,
    )

    def _tokenise(self, text: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        pos = 0
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            if i + 2 <= n and text[i:i+3] == "<->":
                tokens.append(("IFF", "<->"))
                i += 3
                continue
            if i + 1 < n and text[i:i+2] == "->":
                tokens.append(("IMP", "->"))
                i += 2
                continue
            if text[i] == "(":
                tokens.append(("LP", "("))
                i += 1; continue
            if text[i] == ")":
                tokens.append(("RP", ")"))
                i += 1; continue
            if text[i] == ",":
                tokens.append(("COMMA", ","))
                i += 1; continue
            if text[i] == ":":
                tokens.append(("COLON", ":"))
                i += 1; continue
            if text[i] in ("=", "≠", "<", ">"):
                i += 1; continue
            if text[i].isalpha() or text[i] == "_":
                j = i
                _KEYWORDS = {"forall", "exists", "and", "or", "not", "true", "false"}
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                word = text[i:j]
                if word.lower() in _KEYWORDS:
                    tokens.append(("KW", word.lower()))
                    i = j
                    continue
                while j < n and text[j] == " ":
                    k = j + 1
                    while k < n and (text[k].isalnum() or text[k] == "_"):
                        k += 1
                    next_word = text[j+1:k]
                    if not next_word or next_word.lower() in _KEYWORDS:
                        break
                    kk = k
                    while kk < n and text[kk] == " ":
                        kk += 1
                    if kk < n and text[kk] == "(":
                        word += " " + next_word
                        j = k
                    else:
                        break
                tokens.append(("IDENT", _sanitise(word)))
                i = j
                continue
            i += 1
        return tokens

    _pos: int
    _tokens: list[tuple[str, str]]
    _vars: dict[str, Any]

    def _peek(self) -> tuple[str, str] | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, kind: str | None = None, val: str | None = None) -> tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end of tokens")
        if kind and tok[0] != kind:
            raise _ParseError(f"Expected {kind}, got {tok}")
        if val and tok[1].lower() != val.lower():
            raise _ParseError(f"Expected '{val}', got '{tok[1]}'")
        self._pos += 1
        return tok


    def _parse_formula(self) -> Any:
        tok = self._peek()
        if tok and tok[0] == "KW" and tok[1].lower() in ("forall", "exists"):
            return self._parse_quantifier()
        return self._parse_biconditional()

    def _parse_quantifier(self) -> Any:
        q = self._consume("KW")[1].lower()
        var_name = self._consume("IDENT")[1]
        if self._peek() and self._peek()[0] == "COLON":
            self._consume("COLON")
        z3_var = self._var(var_name)
        self._vars[var_name] = z3_var
        if self._peek() and self._peek()[0] == "LP":
            self._consume("LP")
            body = self._parse_formula()
            if self._peek() and self._peek()[0] == "RP":
                self._consume("RP")
        else:
            body = self._parse_formula()
        del self._vars[var_name]
        if q == "forall":
            return z3.ForAll([z3_var], body)
        else:
            return z3.Exists([z3_var], body)

    def _parse_biconditional(self) -> Any:
        left = self._parse_implication()
        while self._peek() and self._peek()[0] == "IFF":
            self._consume("IFF")
            right = self._parse_implication()
            left = z3.And(z3.Implies(left, right), z3.Implies(right, left))
        return left

    def _parse_implication(self) -> Any:
        left = self._parse_disjunction()
        while self._peek() and self._peek()[0] == "IMP":
            self._consume("IMP")
            right = self._parse_disjunction()
            left = z3.Implies(left, right)
        return left

    def _parse_disjunction(self) -> Any:
        left = self._parse_conjunction()
        while self._peek() and self._peek()[0] == "KW" and self._peek()[1].lower() == "or":
            self._consume("KW", "or")
            right = self._parse_conjunction()
            left = z3.Or(left, right)
        return left

    def _parse_conjunction(self) -> Any:
        left = self._parse_negation()
        while self._peek() and self._peek()[0] == "KW" and self._peek()[1].lower() == "and":
            self._consume("KW", "and")
            right = self._parse_negation()
            left = z3.And(left, right)
        return left

    def _parse_negation(self) -> Any:
        if self._peek() and self._peek()[0] == "KW" and self._peek()[1].lower() == "not":
            self._consume("KW", "not")
            return z3.Not(self._parse_negation())
        return self._parse_atom()

    def _parse_atom(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end in atom")
        if tok[0] == "LP":
            self._consume("LP")
            expr = self._parse_formula()
            if self._peek() and self._peek()[0] == "RP":
                self._consume("RP")
            return expr
        if tok[0] == "KW" and tok[1].lower() == "true":
            self._consume()
            return z3.BoolVal(True)
        if tok[0] == "KW" and tok[1].lower() == "false":
            self._consume()
            return z3.BoolVal(False)
        if tok[0] == "KW" and tok[1].lower() in ("forall", "exists"):
            return self._parse_quantifier()
        if tok[0] == "IDENT":
            name = self._consume("IDENT")[1]
            if self._peek() and self._peek()[0] == "LP":
                self._consume("LP")
                args = self._parse_arglist()
                if self._peek() and self._peek()[0] == "RP":
                    self._consume("RP")
                fn = self._fn(name, len(args))
                return fn(*args)
            if name in self._vars:
                raise _ParseError(f"Bare variable '{name}' used as predicate")
            return z3.Const(name + "_fact", z3.BoolSort())
        raise _ParseError(f"Unexpected token: {tok}")

    def _parse_arglist(self) -> list[Any]:
        args: list[Any] = []
        while True:
            tok = self._peek()
            if tok is None or tok[0] == "RP":
                break
            if tok[0] == "IDENT":
                var_name = self._consume("IDENT")[1]
                if var_name in self._vars:
                    args.append(self._vars[var_name])
                else:
                    const = z3.Const(var_name, self._sort_())
                    self._vars[var_name] = const
                    args.append(const)
            elif tok[0] == "KW":
                kw_name = self._consume("KW")[1]
                const = z3.Const(_sanitise(kw_name), self._sort_())
                args.append(const)
            else:
                break
            if self._peek() and self._peek()[0] == "COMMA":
                self._consume("COMMA")
            else:
                break
        return args

    _INFIX_OPS_RE = re.compile(
        r"(\w+)\s+(?:neq|in|notin|leq|geq|subset|subseteq)\s+(\w+)",
        re.IGNORECASE,
    )
    _EQ_RE = re.compile(r"(\w+)\s*=\s*(\w+)")

    def _preprocess(self, text: str) -> str:
        text = self._INFIX_OPS_RE.sub(r"rel(\1,\2)", text)
        text = self._EQ_RE.sub(r"eq(\1,\2)", text)
        return text

    def build(self, text: str) -> Any:
        text = self._preprocess(text)
        self._tokens = self._tokenise(text)
        self._pos = 0
        self._vars = {}
        expr = self._parse_formula()
        return expr


def _try_z3_compile(block: str, concept_name: str) -> Any | None:
    if not HAS_Z3:
        return None
    try:
        builder = _Z3Builder()
        return builder.build(block)
    except Exception:
        return None


def parse_to_z3(fol_text: str, concept_name: str) -> tuple[Any | None, bool]:
    if not HAS_Z3:
        return None, False

    block = extract_fol_block(fol_text) or fol_text.strip()
    if not block:
        return None, False

    if _CONTR_AND.search(block):
        E = z3.DeclareSort("Entity_contr")
        x = z3.Const("x", E)
        P = z3.Function("P_contr", E, z3.BoolSort())
        return z3.And(P(x), z3.Not(P(x))), True

    if _is_self_negation(block, concept_name):
        E = z3.DeclareSort("Entity_selfneg")
        x = z3.Const("x", E)
        C = z3.Function(_sanitise(concept_name), E, z3.BoolSort())
        iff = z3.And(z3.Implies(C(x), z3.Not(C(x))), z3.Implies(z3.Not(C(x)), C(x)))
        return z3.ForAll([x], iff), True

    expr = _try_z3_compile(block, concept_name)
    if expr is not None:
        return expr, True

    return None, False


def check_satisfiability(expr: Any, timeout_ms: int = 10_000) -> str:
    if not HAS_Z3 or expr is None:
        return "parse_fail"
    try:
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(expr)
        result = s.check()
        return str(result)
    except Exception:
        return "error"
