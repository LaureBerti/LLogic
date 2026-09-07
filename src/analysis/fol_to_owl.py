"""Translate a stored FOL definition body into an OWL class expression (the reasoner validation).

Why this is deliberately narrow. The question is whether a definition the Phase 1
filter accepts survives a real OWL reasoner. Answering that means asserting the
definition into an ontology, which means turning LLM-written pseudo-FOL into OWL. A
permissive translator would be the wrong tool: it would invent structure the model
never wrote, and any unsatisfiability it produced would be an artefact of translation
rather than evidence about the model. The design therefore restricts translation to
the fragment that maps onto OWL DL without guessing --- a conjunction of named classes
--- and refuses everything else with a recorded reason, so the not-translatable count
is published rather than hidden.

What is accepted:

    Isotope(x) and Radioactive(x)          ->  Isotope and Radioactive
    x in Bone and x in HumanBody           ->  Bone and HumanBody
    [Tissue(x) and Vascular(x)]            ->  Tissue and Vascular

What is refused, and why each refusal is the honest choice:

    x = LateralMeniscus        equality is a nominal; owl:oneOf changes the semantics
    forall z: P(z) -> Q(z)     universals need a property to quantify over; the LLM
                               text does not say which, so any choice would be ours
    exists y: R(x, y)          an existential restriction needs a property AND a
                               filler; these are frequently malformed in the corpus
    not P(x)                   complement flips satisfiability, the very quantity
                               being measured --- never inferred from loose syntax
    x.genus = "Xanthomonas"    a data property assertion, not a class expression
    prose                      "x is a substance with the following properties"

Every refusal returns a reason code so Section 4's denominator caveat can be reported
per cause instead of as one opaque total.

No LLM calls, no network. Pure string analysis over data already on disk.
"""

from __future__ import annotations

import re

#: Refusal reasons, ordered by how they are checked. Reported verbatim in results.
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
#: Text is present but states no biconditional. Mostly one-directional forms such as
#: "forall x (C(x) -> Body)", which assert a necessary condition rather than define the
#: concept, so there is no class expression to make equivalent to C. Kept separate from
#: empty_body because the distinction is the the: a weaker logical form is a
#: finding about the model, an absent one is a finding about the pipeline.
REASON_NO_BICONDITIONAL = "no_biconditional"

#: A unary predicate applied to the definition variable: Tissue(x)
_UNARY = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]*)\s*\(\s*x\s*\)$")
#: Set-membership spelling of the same thing: x in Tissue, x in *Tissue*
_MEMBER = re.compile(r"^x\s+in\s+[*_`]*\s*([A-Za-z][A-Za-z0-9_\-]*)\s*[*_`]*$", re.IGNORECASE)

def _strip_wrappers(text: str) -> str:
    """Remove matched outer brackets and markdown emphasis, repeatedly."""
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
    """Split on top-level ' and ' / ' AND ' / '∧', ignoring bracketed regions."""
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
            if char == "∧":                       # ∧
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
    """Translate a whole stored ``fol_block``, distinguishing the two absence cases.

    Separates "there is no text" from "there is text but no biconditional", which
    ``translate_body`` alone cannot tell apart once the body comes back empty.
    """
    if not fol_block or not fol_block.strip():
        return None, REASON_EMPTY
    body = extract_body(fol_block)
    if not body:
        return None, REASON_NO_BICONDITIONAL
    return translate_body(body)

def translate_body(body: str) -> tuple[list[str] | None, str]:
    """Return (list of named classes, "ok") or (None, reason).

    The returned list is the conjunction the definition asserts. An empty list is
    never returned as a success: a definition that constrains nothing is refused as
    prose, because asserting ``C equivalentTo owl:Thing`` would silently make every
    such concept trivially satisfiable and inflate the pass rate.
    """
    if not body or not body.strip():
        return None, REASON_EMPTY

    text = " ".join(body.split())
    lowered = text.lower()

    # Order matters: the cheapest, most decisive refusals first. Each of these
    # constructs changes satisfiability in a way loose syntax cannot pin down.
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
    if re.search(r"\w\.\w+\s*=", text):                # x.genus = "..."
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
        # A predicate over more than one argument is a relation, not a class.
        if re.match(r"^[A-Za-z][A-Za-z0-9_\-]*\s*\([^)]*,", piece):
            return None, REASON_NARY_PREDICATE
        return None, REASON_PROSE

    if not classes:
        return None, REASON_PROSE
    return classes, "ok"

#: LaTeX spellings of the logical operators, longest first so that \leftrightarrow is
#: consumed before \left. A large share of the corpus is written this way --- 3,479 of
#: 9,793 stored definitions carry no ASCII "<->" at all and would otherwise be counted
#: as having no body, turning a notation preference into a fake not-translatable rate.
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

#: Unicode operators, per the normalisation the FOL parser already applies.
_UNICODE_SUBSTITUTIONS = [
    ("\u2194", " <-> "), ("\u2192", " -> "), ("\u2227", " and "),
    ("\u2228", " or "), ("\u00ac", " not "), ("\u2200", " forall "),
    ("\u2203", " exists "), ("\u2208", " in "), ("\u2260", " != "),
]

def normalise_notation(text: str) -> str:
    """Rewrite LaTeX and Unicode logic notation into the ASCII forms used here.

    Purely a notation change: no operator is added, removed or reinterpreted, so a
    definition's logical content is untouched. This runs before body extraction so
    that a model's choice of LaTeX does not masquerade as a missing definition.
    """
    if not text:
        return ""
    for pattern, replacement in _LATEX_SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    for symbol, replacement in _UNICODE_SUBSTITUTIONS:
        text = text.replace(symbol, replacement)
    return " ".join(text.split())

def extract_body(fol_block: str) -> str:
    """The right-hand side of the biconditional in a stored fol_block."""
    if not fol_block:
        return ""
    match = re.search(r"<->(.*)", normalise_notation(fol_block), re.DOTALL)
    return match.group(1).strip() if match else ""
