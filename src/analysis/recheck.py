"""Re-check every stored Phase 1 definition under a corrected encoding.

Why this exists. The Phase 1 pipeline encodes a definition as

    forall x: C(x) <-> Body(x)

and asks Z3 whether that sentence is satisfiable. It essentially always is: if
Body is contradictory, the sentence is still satisfied by giving C an empty
extension. The injection experiment confirms this behaviour directly --
``C(x) <-> (P(x) and Q(x) and not P(x))`` is reported ``sat``, while the
syntactically adjacent ``C(x) <-> (P(x) and not P(x))`` is reported ``unsat``
only because a pattern matcher fires on it. Detection therefore tracked
syntactic adjacency rather than logic.

The corrected question is the one OWL reasoners actually ask about a class: is
the class satisfiable, i.e. can anything at all satisfy the definition's body?

    exists x: Body(x)

If that is unsatisfiable, the definition describes a necessarily empty concept,
which is the concept-level analogue of an unsatisfiable OWL class.

This runs entirely on ``fol_block`` values already stored on disk. No LLM calls.

Run:
    python src/analysis/recheck.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

# Definition connectives, most specific first. The parser normalises Unicode to
# ASCII, but stored blocks also contain raw LaTeX from some models.
_CONNECTIVES = ["<->", "<=>", r"\leftrightarrow", r"\equiv", "->", r"\rightarrow"]

def extract_body(block: str) -> tuple[str, str] | None:
    """Return (connective, body) for the definitional part of a block."""
    for connective in _CONNECTIVES:
        idx = block.find(connective)
        if idx != -1:
            body = block[idx + len(connective):].strip()
            body = body.strip("[]").strip()
            body = re.sub(r"\\[\[\]()]", "", body).strip()
            if body:
                return connective, body
    return None

def clopper_pearson(k: int, n: int, conf_level: float) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta
    alpha = 1.0 - conf_level
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lower, upper)

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = hydra.utils.get_original_cwd()
    os.chdir(root)
    out_dir = os.path.join(root, cfg.paths.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from src.parsing.fol_parser import parse_to_z3, check_satisfiability

    ontologies = set(cfg.grid.ontologies)
    models = set(cfg.grid.models)
    strategies = set(cfg.grid.strategies)
    timeout_ms = int(cfg.injection.solver_timeout_s * 1000)

    pattern = os.path.join(cfg.paths.results_dir, cfg.paths.phase1_glob)
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("ontology") in ontologies
                        and row.get("llm_model") in models
                        and row.get("strategy") in strategies):
                    rows.append(row)

    records = []
    for i, row in enumerate(rows):
        block = (row.get("fol_block") or "").strip()
        original = row.get("solver_result", "")
        parts = extract_body(block) if block else None
        if parts is None:
            verdict = "no_body_found"
        else:
            _, body = parts
            expr, ok = parse_to_z3(f"exists x: ({body})", row.get("concept_label", "C"))
            verdict = check_satisfiability(expr, timeout_ms=timeout_ms) if ok else "parse_fail"
        records.append({
            "ontology": row["ontology"], "llm_model": row["llm_model"],
            "strategy": row["strategy"], "concept_label": row.get("concept_label", ""),
            "original_solver_result": original,
            "corrected_verdict": verdict,
            "fol_block": block[:400],
        })
        if (i + 1) % 250 == 0:
            print(f"\rrechecked {i + 1}/{len(rows)}", end="", flush=True)
    print()

    with open(os.path.join(out_dir, "recheck_records.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    verdicts = Counter(r["corrected_verdict"] for r in records)
    # Evaluable = the corrected check actually reached a solver decision.
    evaluable = [r for r in records if r["corrected_verdict"] in ("sat", "unsat")]
    contradictory = [r for r in evaluable if r["corrected_verdict"] == "unsat"]
    lo, hi = clopper_pearson(len(contradictory), len(evaluable), cfg.stats.conf_level)

    by_model = defaultdict(lambda: [0, 0])
    for r in evaluable:
        by_model[r["llm_model"]][1] += 1
        if r["corrected_verdict"] == "unsat":
            by_model[r["llm_model"]][0] += 1

    newly = [r for r in contradictory if r["original_solver_result"] != "unsat"]

    summary: dict[str, Any] = {
        "n_rows_examined": len(records),
        "corrected_verdict_counts": dict(verdicts),
        "n_evaluable": len(evaluable),
        "n_contradictory": len(contradictory),
        "contradiction_rate_over_evaluable": (
            round(len(contradictory) / len(evaluable), 6) if evaluable else None),
        "contradiction_ci": [round(lo, 6), round(hi, 6)],
        "n_newly_detected_vs_original": len(newly),
        "by_model": {m: {"n_unsat": v[0], "n_evaluable": v[1],
                         "rate": round(v[0] / v[1], 4) if v[1] else None}
                     for m, v in sorted(by_model.items())},
        "examples_newly_detected": [
            {k: r[k] for k in ("ontology", "llm_model", "strategy",
                               "concept_label", "fol_block")}
            for r in newly[:15]],
        "encoding_note": (
            "original: satisfiability of 'forall x: C(x) <-> Body(x)', which an "
            "empty extension of C always satisfies. corrected: satisfiability of "
            "'exists x: Body(x)', the analogue of OWL class satisfiability."),
    }

    out_path = os.path.join(out_dir, "recheck_summary.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "examples_newly_detected"}, indent=2, sort_keys=True))
    print(f"\nwritten: {out_path}")

if __name__ == "__main__":
    main()
