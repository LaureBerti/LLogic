"""Contradiction-injection validation of the Phase 1 checker.

The manuscript reports zero Z3-detectable contradictions across every compilable
FOL definition. On its own that number is uninterpretable: a checker that never
fires would produce exactly the same result. the analysis asks for validation
against controlled contradictions.

This task takes definitions the models actually produced, injects contradictions
of known kinds at known positions, and measures how often the existing Phase 1
pipeline recovers them. It reports a sensitivity (true-positive rate) per
injection kind, and a false-positive rate on untouched controls.

Nothing here involves an LLM: the definitions already exist on disk and the
checker is local Z3. It is CPU work of the order of seconds per hundred formulas.

Run:
    python src/analysis/injection.py
    python src/analysis/injection.py injection.n_per_kind=500
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

_ATOM = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*([^()]*?)\s*\)")
_BICOND = re.compile(r"<->")

def split_definition(block: str) -> tuple[str, str] | None:
    """Split 'forall x: C(x) <-> BODY' into (prefix_including_arrow, body)."""
    match = _BICOND.search(block)
    if not match:
        return None
    head, body = block[:match.end()], block[match.end():]
    return (head, body.strip()) if body.strip() else None

def inject_explicit_negation(block: str, label: str, rng: random.Random) -> str | None:
    """Conjoin the negation of a conjunct the definition already asserts.

    Produces C(x) <-> (... and P(t) and not P(t)), which is unsatisfiable for
    any non-empty extension of C.
    """
    parts = split_definition(block)
    if not parts:
        return None
    head, body = parts
    atoms = [m.group(0) for m in _ATOM.finditer(body)
             if m.group(1).lower() not in {"forall", "exists", "not", "and", "or"}]
    if not atoms:
        return None
    atom = rng.choice(atoms)
    return f"{head} ({body}) and not {atom}"

def inject_fresh_contradiction(block: str, label: str, rng: random.Random) -> str | None:
    """Conjoin a self-contained contradiction on a predicate not otherwise used."""
    parts = split_definition(block)
    if not parts:
        return None
    head, body = parts
    fresh = f"Zq{rng.randrange(10_000)}"
    return f"{head} ({body}) and ({fresh}(x) and not {fresh}(x))"

def inject_self_negation(block: str, label: str, rng: random.Random) -> str | None:
    """Replace the body with the negation of the definiendum: C(x) <-> not C(x)."""
    parts = split_definition(block)
    if not parts:
        return None
    head, _ = parts
    safe = re.sub(r"\W+", "_", label).strip("_") or "C"
    return f"{head} not {safe}(x)"

INJECTORS = {
    "explicit_negation_of_own_conjunct": inject_explicit_negation,
    "fresh_predicate_contradiction": inject_fresh_contradiction,
    "self_negation": inject_self_negation,
}

def load_compilable(cfg: DictConfig) -> list[dict]:
    ontologies = set(cfg.grid.ontologies)
    models = set(cfg.grid.models)
    strategies = set(cfg.grid.strategies)
    pattern = os.path.join(cfg.paths.results_dir, cfg.paths.phase1_glob)
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("ontology") in ontologies
                        and row.get("llm_model") in models
                        and row.get("strategy") in strategies
                        and row.get("fol_parse_success") == "1"
                        and (row.get("fol_block") or "").strip()):
                    rows.append(row)
    return rows

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

    n_per_kind = int(cfg.injection.n_per_kind)
    timeout_ms = int(cfg.injection.solver_timeout_s * 1000)
    rng = random.Random(cfg.injection.seed)

    pool = load_compilable(cfg)
    if not pool:
        sys.exit("no compilable Phase 1 definitions found")
    print(f"compilable definitions available: {len(pool)}")

    def verdict(block: str, label: str) -> str:
        expr, ok = parse_to_z3(block, label)
        if not ok:
            return "parse_fail"
        return check_satisfiability(expr, timeout_ms=timeout_ms)

    records: list[dict] = []

    # Control arm: untouched definitions must not be flagged.
    controls = rng.sample(pool, min(n_per_kind, len(pool)))
    for row in controls:
        records.append({
            "kind": "control_unmodified",
            "ontology": row["ontology"], "llm_model": row["llm_model"],
            "strategy": row["strategy"], "concept_label": row["concept_label"],
            "verdict": verdict(row["fol_block"], row["concept_label"]),
        })
        print(f"\rcontrol {len(records)}/{len(controls)}", end="", flush=True)
    print()

    # Injection arms.
    for kind, injector in INJECTORS.items():
        chosen = rng.sample(pool, min(n_per_kind, len(pool)))
        done = 0
        for row in chosen:
            mutated = injector(row["fol_block"], row["concept_label"], rng)
            if mutated is None:
                records.append({
                    "kind": kind, "ontology": row["ontology"],
                    "llm_model": row["llm_model"], "strategy": row["strategy"],
                    "concept_label": row["concept_label"],
                    "verdict": "not_injectable",
                })
                continue
            records.append({
                "kind": kind, "ontology": row["ontology"],
                "llm_model": row["llm_model"], "strategy": row["strategy"],
                "concept_label": row["concept_label"],
                "verdict": verdict(mutated, row["concept_label"]),
            })
            done += 1
            print(f"\r{kind} {done}/{len(chosen)}", end="", flush=True)
        print()

    with open(os.path.join(out_dir, "injection_records.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, Any] = {
        "n_pool_compilable": len(pool),
        "n_per_kind": n_per_kind,
        "solver_timeout_s": cfg.injection.solver_timeout_s,
        "seed": cfg.injection.seed,
        "arms": {},
    }
    for kind in ["control_unmodified"] + list(INJECTORS):
        arm = [r for r in records if r["kind"] == kind]
        verdicts = Counter(r["verdict"] for r in arm)
        injectable = [r for r in arm if r["verdict"] != "not_injectable"]
        detected = sum(1 for r in injectable if r["verdict"] == "unsat")
        lo, hi = clopper_pearson(detected, len(injectable), cfg.stats.conf_level)
        entry = {
            "n_attempted": len(arm),
            "n_evaluated": len(injectable),
            "n_not_injectable": len(arm) - len(injectable),
            "verdicts": dict(verdicts),
            "n_flagged_unsat": detected,
        }
        if kind == "control_unmodified":
            entry["false_positive_rate"] = (round(detected / len(injectable), 6)
                                            if injectable else None)
            entry["interpretation"] = (
                "an unmodified definition flagged as unsat would be a false "
                "positive; this arm calibrates the checker's specificity")
        else:
            entry["detection_rate"] = (round(detected / len(injectable), 4)
                                       if injectable else None)
            entry["detection_ci"] = [round(lo, 4), round(hi, 4)]
            entry["interpretation"] = (
                "share of injected contradictions of this kind that the Phase 1 "
                "pipeline recovers; this is the sensitivity the zero-rate finding "
                "must be read against")
        summary["arms"][kind] = entry

    out_path = os.path.join(out_dir, "injection_summary.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwritten: {out_path}")

if __name__ == "__main__":
    main()
