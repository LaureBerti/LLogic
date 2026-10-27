from __future__ import annotations

import re

REASON_EQUALITY = "equality_or_nominal"
REASON_UNIVERSAL = "universal_quantifier"
REASON_EXISTENTIAL = "existential_restriction"
REASON_NEGATION = "negation"
REASON_DISJUNCTION = "disjunction"
REASON_IMPLICATION = "implication"
REASON_DATA_PROPERTY = "data_property"
REASON_NARY_PREDICATE = "n_ary_predicate"
REASON_PROSE = "prose_not_formula"
REASON_EMPTY = "empty_body"
REASON_NO_BICONDITIONAL = "no_biconditional"

_UNARY = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]*)\s*\(\s*x\s*\)$")
_MEMBER = re.compile(r"^x\s+in\s+[*_`]*\s*([A-Za-z][A-Za-z0-9_\-]*)\s*[*_`]*$", re.IGNORECASE)


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and text:
        changed = False
        for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
            if text.startswith(opener) and text.endswith(closer) and _balanced(text[1:-1], opener, closer):
                text = text[1:-1].strip()
                changed = True
    return text


def _balanced(text: str, opener: str, closer: str) -> bool:
    depth = 0
    for char in text:
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _split_conjuncts(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    index = 0
    lowered = body.lower()
    while index < len(body):
        char = body[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if depth == 0:
            if char == "∧":
                parts.append(current)
                current = ""
                index += 1
                continue
            if lowered.startswith(" and ", index):
                parts.append(current)
                current = ""
                index += 5
                continue
        current += char
        index += 1
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def classify_definition(fol_block: str) -> tuple[list[str] | None, str]:
    if not fol_block or not fol_block.strip():
        return None, REASON_EMPTY
    body = extract_body(fol_block)
    if not body:
        return None, REASON_NO_BICONDITIONAL
    return translate_body(body)


def translate_body(body: str) -> tuple[list[str] | None, str]:
    if not body or not body.strip():
        return None, REASON_EMPTY

    text = " ".join(body.split())
    lowered = text.lower()

    if "->" in text or "→" in text or "implies" in lowered:
        return None, REASON_IMPLICATION
    if re.search(r"\bnot\b|¬|\!", lowered):
        return None, REASON_NEGATION
    if re.search(r"\bor\b|∨", lowered):
        return None, REASON_DISJUNCTION
    if re.search(r"\bforall\b|∀", lowered):
        return None, REASON_UNIVERSAL
    if re.search(r"\bexists\b|∃", lowered):
        return None, REASON_EXISTENTIAL
    if re.search(r"\w\.\w+\s*=", text):
        return None, REASON_DATA_PROPERTY
    if "=" in text or "≠" in text:
        return None, REASON_EQUALITY

    conjuncts = _split_conjuncts(_strip_wrappers(text))
    if not conjuncts:
        return None, REASON_EMPTY

    classes: list[str] = []
    for conjunct in conjuncts:
        piece = _strip_wrappers(conjunct)
        match = _UNARY.match(piece) or _MEMBER.match(piece)
        if match:
            classes.append(match.group(1))
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9_\-]*\s*\([^)]*,", piece):
            return None, REASON_NARY_PREDICATE
        return None, REASON_PROSE

    if not classes:
        return None, REASON_PROSE
    return classes, "ok"


_LATEX_SUBSTITUTIONS = [
    (r"\\leftrightarrow", " <-> "), (r"\\iff", " <-> "), (r"\\equiv", " <-> "),
    (r"\\rightarrow", " -> "), (r"\\Rightarrow", " -> "), (r"\\implies", " -> "),
    (r"\\to\b", " -> "),
    (r"\\land\b", " and "), (r"\\wedge", " and "),
    (r"\\lor\b", " or "), (r"\\vee", " or "),
    (r"\\lnot\b", " not "), (r"\\neg", " not "),
    (r"\\forall", " forall "), (r"\\exists", " exists "),
    (r"\\in\b", " in "), (r"\\neq", " != "),
    (r"\\text\s*\{([^}]*)\}", r"\1"), (r"\\mathrm\s*\{([^}]*)\}", r"\1"),
    (r"\\mathit\s*\{([^}]*)\}", r"\1"), (r"\\mathbf\s*\{([^}]*)\}", r"\1"),
    (r"\\left", ""), (r"\\right", ""),
    (r"\\[,;:!]", " "), (r"\\quad", " "), (r"\\qquad", " "),
    (r"\\\[", " "), (r"\\\]", " "), (r"\\\(", " "), (r"\\\)", " "),
    (r"\\\\", " "),
]

_UNICODE_SUBSTITUTIONS = [
    ("\u2194", " <-> "), ("\u2192", " -> "), ("\u2227", " and "),
    ("\u2228", " or "), ("\u00ac", " not "), ("\u2200", " forall "),
    ("\u2203", " exists "), ("\u2208", " in "), ("\u2260", " != "),
]


def normalise_notation(text: str) -> str:
    if not text:
        return ""
    for pattern, replacement in _LATEX_SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    for symbol, replacement in _UNICODE_SUBSTITUTIONS:
        text = text.replace(symbol, replacement)
    return " ".join(text.split())


def extract_body(fol_block: str) -> str:
    if not fol_block:
        return ""
    match = re.search(r"<->(.*)", normalise_notation(fol_block), re.DOTALL)
    return match.group(1).strip() if match else ""
