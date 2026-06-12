"""Generate prompts for LLM querying over ontology concepts/relations."""
from __future__ import annotations
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
