"""Compute consistency metrics from solver results and LLM responses."""
from __future__ import annotations
import re
from typing import Any
from collections import Counter

def compute_concept_metrics(results: list[dict]) -> dict:
    """
    Aggregate per-concept solver results into phase-1 metrics.
    Input: list of dicts with keys: concept, solver_result, fol_parse_success, is_tautology
    """
    n = len(results)
    if n == 0:
        return {}

    n_unsat = sum(1 for r in results if r.get("solver_result") == "unsat")
    n_cycle = sum(1 for r in results if r.get("is_tautology", False))
    n_parsed = sum(1 for r in results if r.get("fol_parse_success", False))

    return {
        "n_concepts": n,
        "contradiction_rate": round(n_unsat / n, 4),
        "cycle_rate": round(n_cycle / n, 4),
        "fol_parse_success_rate": round(n_parsed / n, 4),
        "tautology_rate": round(n_cycle / n, 4),
    }

def check_transitivity(llm_responses: dict[tuple, str], relation: dict) -> dict:
    """
    Check transitivity violations.
    llm_responses: {(a, b): "YES"/"NO", ...} from LLM for all pairs
    Returns transitivity_violation_rate.
    """
    if not relation.get("is_transitive", False):
        return {"transitivity_violation_rate": None, "n_triples_checked": 0}

    concepts = list({c for pair in llm_responses for c in pair})
    yes_pairs = {pair for pair, ans in llm_responses.items() if ans.strip().upper().startswith("YES")}

    violations = 0
    checked = 0
    for a in concepts:
        for b in concepts:
            if (a, b) not in yes_pairs:
                continue
            for c in concepts:
                if (b, c) not in yes_pairs:
                    continue
                checked += 1
                if (a, c) not in yes_pairs:
                    violations += 1

    rate = round(violations / checked, 4) if checked > 0 else 0.0
    return {"transitivity_violation_rate": rate, "n_triples_checked": checked, "n_violations": violations}

def check_symmetry(llm_responses: dict[tuple, str], relation: dict) -> dict:
    """Check symmetry violations for symmetric relations."""
    if not relation.get("is_symmetric", False):
        return {"symmetry_violation_rate": None, "n_pairs_checked": 0}

    yes_pairs = {pair for pair, ans in llm_responses.items() if ans.strip().upper().startswith("YES")}
    checked = 0
    violations = 0
    seen = set()
    for (a, b) in yes_pairs:
        if (b, a) in seen:
            continue
        seen.add((a, b))
        checked += 1
        if (b, a) not in yes_pairs:
            violations += 1

    rate = round(violations / checked, 4) if checked > 0 else 0.0
    return {"symmetry_violation_rate": rate, "n_pairs_checked": checked, "n_violations": violations}

def check_domain_range(llm_response: str, relation: dict, subject_concept: dict, object_concept: dict) -> dict:
    """Heuristic domain/range check: flag if subject/object are outside declared domain/range."""
    domain_iris = set(relation.get("domain", []))
    range_iris = set(relation.get("range", []))

    subject_iri = subject_concept.get("iri", "")
    object_iri = object_concept.get("iri", "")

    domain_ok = (not domain_iris) or (subject_iri in domain_iris)
    range_ok = (not range_iris) or (object_iri in range_iris)

    answered_yes = llm_response.strip().upper().startswith("YES")
    error = answered_yes and (not domain_ok or not range_ok)

    return {
        "domain_range_error": int(error),
        "domain_declared": bool(domain_iris),
        "range_declared": bool(range_iris),
    }

def majority_vote(responses: list[str]) -> str:
    """For self-consistency on YES/NO questions (Phase 2): return the majority vote."""
    votes = []
    for r in responses:
        if r.strip().upper().startswith("YES"):
            votes.append("YES")
        elif r.strip().upper().startswith("NO"):
            votes.append("NO")
    if not votes:
        return "ABSTAIN"
    return Counter(votes).most_common(1)[0][0]

def aggregate_fol_responses(responses: list[str]) -> str:
    """For self-consistency on FOL generation (Phase 1): majority vote on solver result.

    Runs the Z3 compiler on all n responses, takes the majority solver_result,
    and returns the longest response that matches the majority result.
    Ties broken by: unsat > sat > unknown > parse_fail (conservative).
    This is consistent with Wang et al. (2022) majority-vote SC semantics.
    """
    from src.parsing.fol_parser import extract_fol_block, parse_to_z3, check_satisfiability

    if not responses:
        return "ABSTAIN"

    evaluated: list[tuple[str, str]] = []  # (solver_result, response)
    for r in responses:
        if not r or not r.strip():
            continue
        block = extract_fol_block(r) or ""
        expr, _ = parse_to_z3(block, "")
        result = check_satisfiability(expr, timeout_ms=5_000)
        evaluated.append((result, r))

    if not evaluated:
        return responses[0] if responses else "ABSTAIN"

    # Majority vote on solver_result
    result_counts = Counter(res for res, _ in evaluated)
    majority_result, _ = result_counts.most_common(1)[0]

    # If tied, prefer: unsat > sat > unknown > parse_fail (conservative tie-break only)
    if len(result_counts) > 1:
        top_count = result_counts.most_common(1)[0][1]
        tied = [r for r, c in result_counts.items() if c == top_count]
        if len(tied) > 1:
            priority = {"unsat": 0, "sat": 1, "unknown": 2, "parse_fail": 3}
            majority_result = min(tied, key=lambda r: priority.get(r, 4))

    # Return longest response with the majority result (for logging/re-parsing)
    candidates = [(r, resp) for r, resp in evaluated if r == majority_result]
    return max(candidates, key=lambda x: len(x[1]))[1]
