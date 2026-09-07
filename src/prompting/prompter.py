"""Generate prompts for LLM querying over ontology concepts/relations."""
from __future__ import annotations

import re
from typing import Any

SYSTEM_PROMPT = (
    "You are a formal ontology expert. Answer concisely and formally. "
    "When asked for a logical definition, use first-order logic (FOL) notation: "
    "∀, ∃, →, ∧, ∨, ¬, =. Do not add examples or prose unless explicitly asked."
)

def concept_zero_shot(concept: dict) -> list[dict]:
    label = concept["label"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Define the concept '{label}' using first-order logic (FOL). "
            f"Give necessary and sufficient conditions. Use the form: "
            f"∀x: {label}(x) ↔ [conditions]"
        )},
    ]

def concept_cot(concept: dict) -> list[dict]:
    label = concept["label"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Define the concept '{label}' step by step:\n"
            f"Step 1: Identify its genus (superclass).\n"
            f"Step 2: Identify its differentia (distinguishing properties).\n"
            f"Step 3: State necessary conditions (∀x: {label}(x) → ...).\n"
            f"Step 4: State sufficient conditions (∀x: [conditions] → {label}(x)).\n"
            f"Step 5: Combine into a biconditional FOL statement."
        )},
    ]

def concept_tot(concept: dict) -> list[dict]:
    """Tree-of-Thoughts: explore multiple definition branches, then synthesise."""
    label = concept["label"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"I want to define '{label}' in FOL. Explore three alternative definitional framings:\n"
            f"Branch A: Define by structural/compositional properties.\n"
            f"Branch B: Define by functional/relational properties.\n"
            f"Branch C: Define by extensional examples and counter-examples.\n"
            f"Evaluate each branch: is it necessary? sufficient? consistent?\n"
            f"Synthesise the best elements into a single FOL biconditional: ∀x: {label}(x) ↔ ..."
        )},
    ]

def concept_self_consistency(concept: dict, n: int = 5) -> list[list[dict]]:
    """Return n independent zero-shot prompt chains for majority-vote aggregation."""
    return [concept_zero_shot(concept) for _ in range(n)]

def relation_consistency_prompt(relation: dict, concept_a: dict, concept_b: dict, concept_c: dict | None = None) -> list[dict]:
    """Probe LLM for transitivity/symmetry/domain-range assertions."""
    rel = relation["label"]
    a, b = concept_a["label"], concept_b["label"]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Is it true that '{a}' {rel} '{b}'? Answer YES or NO, then give a one-sentence justification in FOL."},
    ]
    if concept_c:
        c = concept_c["label"]
        msgs.append({"role": "user", "content": f"Given your answer above, is it also true that '{a}' {rel} '{c}'? (Transitivity check.)"})
    return msgs

def subclassof_zero_shot(triple: dict) -> list[dict]:
    """Phase 2b subClassOf transitivity probe, zero-shot.

    Kept byte-identical to the prompt used for the originally reported Phase 2b
    runs, so that previously collected zero-shot cells remain valid and do not
    need to be re-collected.
    """
    return [
        {"role": "system", "content": "You are a formal ontology expert."},
        {"role": "user", "content": (
            f"In an ontology, '{triple['c_label']}' is a subclass of "
            f"'{triple['b_label']}', and '{triple['b_label']}' is a subclass of "
            f"'{triple['a_label']}'. Is '{triple['c_label']}' also a subclass of "
            f"'{triple['a_label']}'? Answer YES or NO only."
        )},
    ]

# Small models drift into prose and never emit a bare YES/NO, which would be
# silently scored as "no violation". Demanding a fixed final token sequence, and
# parsing that sequence first, makes the verdict recoverable.
VERDICT_INSTRUCTION = ("Be brief. Finish your reply with a final line in exactly "
                       "this form:\nANSWER: YES\nor\nANSWER: NO")

FORCED_VERDICT_QUESTION = ("Based on your reasoning above, reply with exactly one "
                           "line: 'ANSWER: YES' or 'ANSWER: NO'.")

def subclassof_cot(triple: dict) -> list[dict]:
    c, b, a = triple["c_label"], triple["b_label"], triple["a_label"]
    return [
        {"role": "system", "content": "You are a formal ontology expert."},
        {"role": "user", "content": (
            f"In an ontology, '{c}' is a subclass of '{b}', and '{b}' is a "
            f"subclass of '{a}'. Reason step by step, in at most four short steps:\n"
            f"Step 1: What it means for '{c}' to be a subclass of '{b}'.\n"
            f"Step 2: What it means for '{b}' to be a subclass of '{a}'.\n"
            f"Step 3: Whether the subclass relation composes across the two steps.\n"
            f"Step 4: Conclude whether '{c}' is a subclass of '{a}'.\n"
            f"{VERDICT_INSTRUCTION}"
        )},
    ]

def subclassof_tot(triple: dict) -> list[dict]:
    c, b, a = triple["c_label"], triple["b_label"], triple["a_label"]
    return [
        {"role": "system", "content": "You are a formal ontology expert."},
        {"role": "user", "content": (
            f"In an ontology, '{c}' is a subclass of '{b}', and '{b}' is a "
            f"subclass of '{a}'. Consider three framings, one short sentence each:\n"
            f"Branch A: subclass as set inclusion over instances.\n"
            f"Branch B: subclass as inheritance of necessary properties.\n"
            f"Branch C: any counter-example where the chain fails.\n"
            f"Then give a single conclusion about whether '{c}' is a subclass of '{a}'.\n"
            f"{VERDICT_INSTRUCTION}"
        )},
    ]

def subclassof_self_consistency(triple: dict, n: int = 5) -> list[list[dict]]:
    """n independent zero-shot chains, to be sampled at temperature > 0."""
    return [subclassof_zero_shot(triple) for _ in range(n)]

def build_subclassof_prompts(triple: dict, strategy: str,
                             n_samples: int = 5) -> list[list[dict]] | list[dict]:
    """Strategy-aware Phase 2b prompt builder.

    The original Phase 2b implementation built one fixed prompt and used the
    strategy only as an output column, so all four strategies re-ran the same
    deterministic probe. This function is what makes the strategy factor real.
    """
    if strategy == "zero_shot":
        return subclassof_zero_shot(triple)
    if strategy == "cot":
        return subclassof_cot(triple)
    if strategy == "tot":
        return subclassof_tot(triple)
    if strategy == "self_consistency":
        return subclassof_self_consistency(triple, n=n_samples)
    raise ValueError(f"unknown strategy: {strategy!r}")

_ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(YES|NO)\b")

def parse_yes_no(answer: str) -> int | None:
    """Return 1 for NO (a transitivity violation), 0 for YES, None if unclear.

    Returning None matters: an unrecoverable verdict must be excluded from the
    denominator, not scored as "no violation", which would bias violation rates
    downwards for exactly those models that ramble.
    """
    if not answer:
        return None
    text = answer.strip().upper()

    # The requested explicit form, last occurrence first.
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return 1 if matches[-1] == "NO" else 0

    # Otherwise a line that is only a verdict, scanning from the end.
    for line in reversed(text.splitlines()):
        stripped = line.strip().strip(".*_:# ")
        if stripped in ("YES", "NO"):
            return 1 if stripped == "NO" else 0

    # Last resort: the whole reply mentions exactly one of the two.
    has_yes, has_no = "YES" in text, "NO" in text
    if has_yes != has_no:
        return 1 if has_no else 0
    return None

def reconstruction_prompt(conversation_history: list[dict], ontology_name: str) -> list[dict]:
    """Ask LLM to reconstruct the ontology as OWL Manchester syntax from its own conversation."""
    msgs = conversation_history.copy()
    msgs.append({"role": "user", "content": (
        f"Based on all the concepts and relations you have defined above, "
        f"reconstruct the '{ontology_name}' ontology in OWL Manchester Syntax. "
        f"Output ONLY the ontology text, starting with 'Ontology: <{ontology_name}>'. "
        f"Include SubClassOf, EquivalentTo, ObjectProperty, Domain, Range declarations."
    )})
    return msgs

def build_prompts(concept: dict, strategy: str, n_samples: int = 5) -> list[list[dict]] | list[dict]:
    """Dispatch to the correct prompting function."""
    if strategy == "zero_shot":
        return concept_zero_shot(concept)
    elif strategy == "cot":
        return concept_cot(concept)
    elif strategy == "tot":
        return concept_tot(concept)
    elif strategy == "self_consistency":
        return concept_self_consistency(concept, n=n_samples)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
