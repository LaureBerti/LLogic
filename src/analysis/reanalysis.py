"""Post-hoc analyses over generated LLogic results.

Two tasks, selected with ``task=``:

``counts``
    Re-derives every Phase 1 count reported in the manuscript directly from the
    result CSVs, and separates well-formed rows from CSV-garbled ones. The
    manuscript, ``contributions.md`` and the data currently disagree; this task
    is the single authority that replaces all three.

``parse_audit``
    Characterises the FOL parse failures. Parse failure
    was previously treated as benign missingness. This task classifies each
    failure, reports the contradiction-rate bound under explicit alternative
    assumptions about the failures, and exports a stratified sample for manual
    audit.

Note on what is recoverable: ``phase1_concepts.py`` stores ``response_text``
truncated to 300 characters, but stores the extracted ``fol_block`` in full.
The audit therefore runs on the FOL block, which is the object whose parseability
is at issue; claims about the surrounding prose are out of reach and are not made.

Run:
    python src/analysis/reanalysis.py task=counts
    python src/analysis/reanalysis.py task=parse_audit audit.sample_n=100
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

# Result rows embed free-text FOL; fields can be large.
csv.field_size_limit(10 ** 9)

# --------------------------------------------------------------------------
# Config schema
# --------------------------------------------------------------------------
@dataclass
class PathsConfig:
    results_dir: str = "outputs/results"
    phase1_glob: str = "**/phase1_results.csv"
    phase2_glob: str = "**/phase2_results.csv"
    phase3_glob: str = "**/phase3_results.csv"
    out_dir: str = "outputs/analysis"

@dataclass
class GridConfig:
    ontologies: list = field(default_factory=list)
    models: list = field(default_factory=list)
    strategies: list = field(default_factory=list)

@dataclass
class StatsConfig:
    conf_level: float = 0.95

@dataclass
class AuditConfig:
    sample_n: int = 60
    seed: int = 42
    export_all_failures: bool = True

@dataclass
class AnalysisConfig:
    task: str = "counts"
    paths: PathsConfig = field(default_factory=PathsConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_phase1(cfg: DictConfig) -> tuple[list[dict], list[dict]]:
    """Return (well_formed_rows, garbled_rows).

    A row is well-formed when its three grid keys are all recognised values.
    Rows failing that test are records whose quoted fields contained raw
    newlines, so the CSV reader split them across records. They are returned
    rather than dropped so their number can be reported.
    """
    ontologies = set(cfg.grid.ontologies)
    models = set(cfg.grid.models)
    strategies = set(cfg.grid.strategies)

    pattern = os.path.join(cfg.paths.results_dir, cfg.paths.phase1_glob)
    good: list[dict] = []
    bad: list[dict] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source_file"] = path
                if (row.get("ontology") in ontologies
                        and row.get("llm_model") in models
                        and row.get("strategy") in strategies):
                    good.append(row)
                else:
                    bad.append(row)
    return good, bad

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, conf_level: float,
                    one_sided: bool = False) -> tuple[float, float]:
    """Exact binomial confidence interval, as proportions.

    ``one_sided=True`` puts the whole tail mass on the upper side, which is the
    convention behind the submitted manuscript's bound: with k=0 it reduces to
    the rule of three, 1 - alpha**(1/n). The two conventions differ materially
    at these n (see ``bound_pair``), so the manuscript must state which it uses.
    Returns (nan, nan) if scipy is unavailable, so the caller can say so.
    """
    if n == 0:
        return (0.0, 1.0)
    try:
        from scipy.stats import beta
    except ImportError:
        return (float("nan"), float("nan"))
    alpha = 1.0 - conf_level
    tail = alpha if one_sided else alpha / 2
    lower = 0.0 if (k == 0 or one_sided) else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return (lower, upper)

def bound_pair(k: int, n: int, conf_level: float) -> dict[str, Any]:
    """Both bound conventions for a rate, so the choice is explicit in the paper."""
    _, hi_two = clopper_pearson(k, n, conf_level, one_sided=False)
    _, hi_one = clopper_pearson(k, n, conf_level, one_sided=True)
    fmt = lambda v: None if v != v else round(100 * v, 4)  # noqa: E731  (NaN check)
    return {
        "k": k,
        "n": n,
        "point_estimate_pct": round(100 * k / n, 6) if n else None,
        "upper_bound_pct_one_sided": fmt(hi_one),
        "upper_bound_pct_two_sided": fmt(hi_two),
        "conf_level": conf_level,
    }

# --------------------------------------------------------------------------
# Task: counts
# --------------------------------------------------------------------------
def task_counts(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    good, bad = load_phase1(cfg)

    parsed = [r for r in good if r.get("fol_parse_success") == "1"]
    failed = [r for r in good if r.get("fol_parse_success") == "0"]
    contradictions = [r for r in good if r.get("contradiction") == "1"]
    solver = Counter(r.get("solver_result") for r in good)

    n_total = len(good)
    n_parsed = len(parsed)
    n_failed = len(failed)
    n_contra = len(contradictions)

    # The manuscript's headline bound: contradictions among COMPILABLE responses.
    contradiction_bound = bound_pair(n_contra, n_parsed, cfg.stats.conf_level)

    cells = Counter((r["ontology"], r["llm_model"], r["strategy"]) for r in good)
    per_model_cells = defaultdict(set)
    for onto, model, strategy in cells:
        per_model_cells[model].add((onto, strategy))

    expected = [(o, m, s)
                for o in cfg.grid.ontologies
                for m in cfg.grid.models
                for s in cfg.grid.strategies]
    missing = [c for c in expected if c not in cells]

    summary = {
        "n_rows_well_formed": n_total,
        "n_rows_csv_garbled": len(bad),
        "n_compilable": n_parsed,
        "n_parse_failures": n_failed,
        "parse_failure_rate": round(n_failed / n_total, 6) if n_total else None,
        "n_contradictions": n_contra,
        "contradiction_rate_over_compilable": (
            round(n_contra / n_parsed, 8) if n_parsed else None),
        "contradiction_bound_over_compilable": contradiction_bound,
        "solver_result_counts": dict(solver),
        "n_cells_filled": len(cells),
        "n_cells_expected": len(expected),
        "cells_per_model": {m: len(v) for m, v in sorted(per_model_cells.items())},
        "missing_cells": [list(c) for c in missing],
        "rows_per_ontology": dict(Counter(r["ontology"] for r in good)),
        "rows_per_model": dict(Counter(r["llm_model"] for r in good)),
        "rows_per_strategy": dict(Counter(r["strategy"] for r in good)),
        "conf_level": cfg.stats.conf_level,
    }

    # Per-cell table, for the paired/repeated-measures analyses that the analysis
    # (point 9) asks for: one row per (ontology, model, strategy) experimental unit.
    cell_rows = []
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in good:
        by_cell[(r["ontology"], r["llm_model"], r["strategy"])].append(r)
    for (onto, model, strategy), rows in sorted(by_cell.items()):
        n = len(rows)
        n_ok = sum(1 for r in rows if r.get("fol_parse_success") == "1")
        n_bad = sum(1 for r in rows if r.get("fol_parse_success") == "0")
        n_c = sum(1 for r in rows if r.get("contradiction") == "1")
        cell_rows.append({
            "ontology": onto,
            "llm_model": model,
            "strategy": strategy,
            "n_concepts": n,
            "n_compilable": n_ok,
            "n_parse_failures": n_bad,
            "parse_success_rate": round(n_ok / n, 6) if n else None,
            "n_contradictions": n_c,
            "contradiction_rate": round(n_c / n_ok, 8) if n_ok else None,
        })

    with open(os.path.join(out_dir, "phase1_cell_table.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cell_rows)

    if bad:
        with open(os.path.join(out_dir, "phase1_garbled_rows.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["_source_file", "ontology", "llm_model", "strategy"],
                extrasaction="ignore")
            writer.writeheader()
            writer.writerows(bad)

    return summary

# --------------------------------------------------------------------------
# Task: parse_audit
# --------------------------------------------------------------------------
# Categories below were derived by reading the failing FOL blocks, not assumed.
# Note that fol_parser.py normalises Unicode logic symbols to ASCII words before
# parsing, so set membership reaches this stage as the word " in ", not as U+2208.
_NL_CONNECTIVE = re.compile(
    r"\b(is a|are|has|have|refers to|such that|which|whose|that is|due to|"
    r"belongs to|used to|means)\b", re.I)
_APOSTROPHE_S = re.compile(r"\w's\b")
_HIGHER_ORDER = re.compile(r"\b(exists|forall)\s+[a-z]\w*\s*\.\s*\(?\s*forall|"
                           r"\b[fgh]\s*\(\s*x\s*\)\s*=", re.I)
_MULTIVAR_QUANT = re.compile(r"\b(exists|forall)\s+[a-z]\w*\s*,\s*[a-z]", re.I)
_SET_NOTATION = re.compile(
    r"[∈∉⊆⊂∪∩]|\\in\b|\bsubset\b|\bsubseteq\b|\bin\b\s*[\{\*\"']|"
    r"\s\bin\b\s|\{[^}]*\}")
_LATEX_MATH = re.compile(r"\\\(|\\\)|\\mathbb|\\text|\$")
# Angle-bracket placeholders use a letter-only class so that the biconditional
# operator "<->" is not mistaken for one.
_ELISION = re.compile(r"\.\.\.|\bP1\b|\bPn\b|\betc\b|<[A-Za-z_ ]+>")
_DOT_ATTR = re.compile(r"\b[a-z]\.\w+", re.I)
_DEGENERATE = re.compile(r"^\s*(forall|exists)\s+\w+\s*[:.]?\s*$", re.I)
_TRAILING_CUT = re.compile(r"[(\[,]\s*$|\b(and|or|not)\s*$", re.I)
_SPACED_PREDICATE = re.compile(r"\b[A-Za-z_]+\s+[A-Za-z_]+\s*\(\s*[a-z]\s*\)")
_QUANT_VARS = re.compile(r"\b(?:forall|exists)\s+((?:[a-z]\w*\s*,\s*)*[a-z]\w*)", re.I)
_PRED_ARGS = re.compile(r"\b[A-Za-z_]\w*\s*\(([^()]*)\)")

def _has_free_variable(text: str) -> bool:
    """True when a predicate takes a single-letter argument that is never bound.

    Catches the common ``Species(s) and s = "..."`` and ``WrittenBy(x, y)``
    shapes, where the model introduces a variable without a quantifier.
    """
    bound = set()
    for group in _QUANT_VARS.findall(text):
        bound.update(v.strip().lower() for v in group.split(","))
    for args in _PRED_ARGS.findall(text):
        for arg in args.split(","):
            arg = arg.strip().strip('"\'').lower()
            if (len(arg) <= 2 and arg.isalpha() and arg not in bound):
                return True
    return False

def classify_failure(fol_block: str) -> str:
    """Assign one substantive category to a non-compilable FOL block.

    Order matters: the first matching rule wins, most-specific first. The
    point of the taxonomy is to separate failures that reflect parser
    expressiveness limits or model underspecification from anything that could
    plausibly hide a logical contradiction.
    """
    text = (fol_block or "").strip()
    if not text:
        return "empty_block"
    if _DEGENERATE.match(text):
        return "degenerate_stub"
    if _TRAILING_CUT.search(text):
        return "truncated_expression"
    if _ELISION.search(text):
        return "elided_placeholder"
    if _LATEX_MATH.search(text):
        return "latex_math_notation"
    if _HIGHER_ORDER.search(text):
        return "higher_order_or_function_valued"
    if _SET_NOTATION.search(text):
        return "set_theoretic_notation"
    if _DOT_ATTR.search(text):
        return "dot_attribute_notation"
    if _APOSTROPHE_S.search(text) or _NL_CONNECTIVE.search(text):
        return "natural_language_predicate_body"
    if _has_free_variable(text):
        return "unbound_free_variable"
    if _SPACED_PREDICATE.search(text):
        return "space_in_predicate_name"
    if _MULTIVAR_QUANT.search(text):
        return "multivariable_quantifier"
    return "other_unparsed_syntax"

# Failure categories that are notation/expressiveness problems rather than
# anything that could conceal an inconsistent definition. Used only to report
# how much of the missingness is structural; it is not a claim of consistency.
NON_SEMANTIC_CATEGORIES = frozenset({
    "empty_block", "degenerate_stub", "truncated_expression",
    "elided_placeholder", "latex_math_notation", "set_theoretic_notation",
    "dot_attribute_notation", "space_in_predicate_name",
    "multivariable_quantifier", "higher_order_or_function_valued",
})

def task_parse_audit(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    good, _ = load_phase1(cfg)
    parsed = [r for r in good if r.get("fol_parse_success") == "1"]
    failed = [r for r in good if r.get("fol_parse_success") == "0"]
    n_total, n_parsed, n_failed = len(good), len(parsed), len(failed)
    n_contra = sum(1 for r in good if r.get("contradiction") == "1")

    for row in failed:
        row["_failure_category"] = classify_failure(row.get("fol_block", ""))

    categories = Counter(r["_failure_category"] for r in failed)
    cat_by_model = defaultdict(Counter)
    for row in failed:
        cat_by_model[row["llm_model"]][row["_failure_category"]] += 1

    n_non_semantic = sum(v for k, v in categories.items()
                         if k in NON_SEMANTIC_CATEGORIES)

    # Sensitivity analysis. The manuscript's bound conditions on compilable
    # responses. the analysis asks what happens under other assumptions about
    # the failures. These are stated as assumptions, not as measurements.
    conf = cfg.stats.conf_level
    scenarios = {
        "A_compilable_only_as_published": dict(
            assumption=("Failures excluded; the bound conditions on compilable "
                        "responses only. This is what the submitted manuscript "
                        "reported, and it is the narrowest reading."),
            **bound_pair(n_contra, n_parsed, conf)),
        "B_failures_assumed_consistent": dict(
            assumption=("Every parse failure is assumed non-contradictory "
                        "(best case for the claim)."),
            **bound_pair(n_contra, n_total, conf)),
        "C_failures_assumed_contradictory": dict(
            assumption=("Every parse failure is assumed contradictory "
                        "(worst case; a deliberately pessimistic bound)."),
            **bound_pair(n_contra + n_failed, n_total, conf)),
    }

    # Stratified manual-audit sample, by model x strategy.
    rng = random.Random(cfg.audit.seed)
    strata = defaultdict(list)
    for row in failed:
        strata[(row["llm_model"], row["strategy"])].append(row)
    per_stratum = max(1, cfg.audit.sample_n // max(1, len(strata)))
    sample = []
    for key in sorted(strata):
        rows = strata[key]
        sample.extend(rng.sample(rows, min(per_stratum, len(rows))))

    audit_cols = ["ontology", "llm_model", "strategy", "concept_label",
                  "_failure_category", "fol_block"]
    with open(os.path.join(out_dir, "parse_failure_audit_sample.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_cols + ["manual_verdict", "manual_note"],
                                extrasaction="ignore")
        writer.writeheader()
        for row in sample:
            out = {c: row.get(c, "") for c in audit_cols}
            out["manual_verdict"] = ""   # to be filled by a human
            out["manual_note"] = ""
            writer.writerow(out)

    if cfg.audit.export_all_failures:
        with open(os.path.join(out_dir, "parse_failures_all.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failed)

    return {
        "n_rows_well_formed": n_total,
        "n_compilable": n_parsed,
        "n_parse_failures": n_failed,
        "parse_failure_rate": round(n_failed / n_total, 6) if n_total else None,
        "failure_categories": dict(categories.most_common()),
        "n_failures_notation_or_expressiveness": n_non_semantic,
        "pct_failures_notation_or_expressiveness": (
            round(100 * n_non_semantic / n_failed, 2) if n_failed else None),
        "failure_categories_by_model": {m: dict(c.most_common())
                                        for m, c in sorted(cat_by_model.items())},
        "sensitivity_scenarios": scenarios,
        "audit_sample_size": len(sample),
        "audit_strata": len(strata),
        "note_on_response_text": (
            "response_text is stored truncated to 300 characters "
            "(src/phase1_concepts.py:117); fol_block is stored in full. "
            "This audit characterises fol_block only."),
    }

TASKS = {"counts": task_counts, "parse_audit": task_parse_audit}

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    if cfg.task not in TASKS:
        sys.exit(f"unknown task {cfg.task!r}; choose one of {sorted(TASKS)}")

    root = hydra.utils.get_original_cwd()
    out_dir = os.path.join(root, cfg.paths.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(root)

    summary = TASKS[cfg.task](cfg, out_dir)

    out_path = os.path.join(out_dir, f"{cfg.task}_summary.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwritten: {out_path}")

if __name__ == "__main__":
    main()
