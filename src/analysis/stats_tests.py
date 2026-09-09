"""Statistical analyses for LLogic.

Three analyses that are all about inference, not about data
collection, and therefore run entirely on the results already on disk:

``paired_phase1``
    An earlier analysis compared models with Kruskal-Wallis and Mann-Whitney,
    which treat cells as independent. They are not: every model is measured on
    the same ontology x strategy combinations. This task re-runs the comparisons
    as blocked, repeated-measures tests (Friedman + Holm-corrected Wilcoxon
    signed-rank), reports Kendall's W as an effect size, and prints the original
    unpaired test beside it so the change is visible.

``phase2_rates``
    Table 4 tested violation rates against a 50% random-guessing null while the
    text argued they exceed 0%. Those are different questions. This task reports
    each rate with an exact confidence interval and evaluates both nulls
    separately, labelling which is which.

``c6_mixed``
    The domain-moderation claim treated model/strategy cells as replicates of a
    domain, when the real replicate is the ontology (2 general vs 3 technical).
    This task fits concept_coverage with ontology as a random effect and reports
    the result whichever way it falls, next to the original Mann-Whitney.

Run:
    python src/analysis/stats_tests.py task=paired_phase1
    python src/analysis/stats_tests.py task=phase2_rates
    python src/analysis/stats_tests.py task=c6_mixed
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

csv.field_size_limit(10 ** 9)

# Domain assignment used for C6.
DOMAIN = {"book": "general", "agrovoc": "general",
          "anatomy": "technical", "cso": "technical", "mesh": "technical"}

# Phase 2 was originally run with a hardcoded SKOS relation list applied to every RDF
# ontology, which is wrong. Counted directly in the source
# files, the properties actually present are:
#
#   AGROVOC : skos:broader 42,269 ; skos:related 8,907 ; skos:narrower 0
#   GEMET   : skos:broader 5,685  ; skos:narrower 5,689
#   CSO     : superTopicOf 44,510 ; contributesTo 48,980 ; no skos:* at all
#
# and neither AGROVOC, GEMET nor CSO declares any transitive or symmetric
# property. A measured rate is therefore only interpretable when the property
# both EXISTS in the ontology and CARRIES the semantics being tested.
RELATION_PRESENT = {
    ("agrovoc", "broader"): True,
    ("agrovoc", "narrower"): False,   # AGROVOC contains no skos:narrower
    ("agrovoc", "related"): True,
    ("cso", "broader"): False,        # CSO contains no skos:broader
    ("cso", "narrower"): False,
    ("cso", "related"): False,        # CSO contains no skos:related
}

# Semantics the defining standard licenses for the tested property.
SEMANTICS_LICENSED = {
    ("broader", "transitivity"): False,   # skos:broader is not transitive
    ("narrower", "transitivity"): False,
    ("related", "symmetry"): True,        # skos:related is symmetric
}

def rate_validity(ontology: str, relation: str, kind: str) -> tuple[bool, str]:
    """Is a measured rate interpretable for this (ontology, relation, test)?"""
    if RELATION_PRESENT.get((ontology, relation)) is False:
        return False, (f"RETRACTED: '{relation}' does not occur in {ontology}; "
                       "the probe tested a property the ontology does not contain")
    licensed = SEMANTICS_LICENSED.get((relation, kind))
    if licensed is False:
        return False, (f"RETRACTED: {relation} carries no {kind} semantics in the "
                       "defining standard, so a violation is not defined")
    return True, "valid"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source_file"] = path
                rows.append(row)
    return rows

def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out

def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted

def kendalls_w(matrix: np.ndarray) -> float:
    """Kendall's W for a blocks x treatments matrix (effect size for Friedman)."""
    n_blocks, k = matrix.shape
    if n_blocks == 0 or k < 2:
        return float("nan")
    ranks = np.apply_along_axis(_rankdata, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    mean_rs = rank_sums.mean()
    s = ((rank_sums - mean_rs) ** 2).sum()
    return float(12 * s / (n_blocks ** 2 * (k ** 3 - k)))

def _rankdata(row: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(row)

def clopper_pearson(k: int, n: int, conf_level: float) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta
    alpha = 1.0 - conf_level
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lower, upper)

# --------------------------------------------------------------------------
# Task: paired_phase1
# --------------------------------------------------------------------------
def _blocked_matrix(cells: dict[tuple, float], treatments: list[str],
                    block_keys: list[tuple]) -> tuple[np.ndarray, list[tuple]]:
    """Build a complete blocks x treatments matrix, dropping incomplete blocks."""
    rows, kept = [], []
    for block in block_keys:
        values = [cells.get(block + (t,), float("nan")) for t in treatments]
        if not any(np.isnan(v) for v in values):
            rows.append(values)
            kept.append(block)
    return np.array(rows, dtype=float), kept

def _friedman_block(name: str, matrix: np.ndarray, treatments: list[str],
                    n_blocks_label: str) -> dict[str, Any]:
    from scipy.stats import friedmanchisquare, wilcoxon, kruskal
    result: dict[str, Any] = {
        "factor": name,
        "experimental_unit": "one (ontology, model, strategy) cell",
        "blocking": n_blocks_label,
        "n_blocks": int(matrix.shape[0]),
        "treatments": treatments,
        "treatment_means": {t: round(float(matrix[:, i].mean()), 4)
                            for i, t in enumerate(treatments)},
        "treatment_medians": {t: round(float(np.median(matrix[:, i])), 4)
                              for i, t in enumerate(treatments)},
    }
    if matrix.shape[0] < 3:
        result["error"] = "too few complete blocks for a repeated-measures test"
        return result

    stat, p = friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    result["friedman"] = {"chi2": round(float(stat), 4), "p": float(p),
                          "kendalls_w": round(kendalls_w(matrix), 4)}

    # The unpaired test the manuscript used, for side-by-side comparison.
    ks_stat, ks_p = kruskal(*[matrix[:, i] for i in range(matrix.shape[1])])
    result["kruskal_wallis_as_submitted"] = {
        "H": round(float(ks_stat), 4), "p": float(ks_p),
        "note": "treats cells as independent; reported only for comparison"}

    pairs, praw = [], []
    for i, j in combinations(range(len(treatments)), 2):
        a, b = matrix[:, i], matrix[:, j]
        if np.allclose(a, b):
            pairs.append((treatments[i], treatments[j], float("nan"), "identical"))
            praw.append(1.0)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_stat, w_p = wilcoxon(a, b)
        diff = a - b
        nz = diff[diff != 0]
        # Matched-pairs rank-biserial correlation as the effect size.
        rbc = float(np.sign(nz).mean()) if nz.size else float("nan")
        pairs.append((treatments[i], treatments[j], float(w_p), round(rbc, 4)))
        praw.append(float(w_p))

    adj = holm(praw)
    result["posthoc_wilcoxon_holm"] = [
        {"a": a, "b": b, "p_raw": (None if p != p else round(p, 6)),
         "p_holm": round(q, 6), "rank_biserial": rb}
        for (a, b, p, rb), q in zip(pairs, adj)
    ]
    return result

def task_paired_phase1(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    path = os.path.join(out_dir, "phase1_cell_table.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path}; run reanalysis.py task=counts first")
    cells_raw = list(csv.DictReader(open(path, newline="")))

    metric = "parse_success_rate"
    values = {(r["ontology"], r["llm_model"], r["strategy"]): as_float(r[metric])
              for r in cells_raw}

    models = list(cfg.grid.models)
    strategies = list(cfg.grid.strategies)
    ontologies = list(cfg.grid.ontologies)

    # Strategy effect: blocks are (ontology, model).
    strat_cells = {(o, m, s): values[(o, m, s)] for (o, m, s) in values}
    mat_s, kept_s = _blocked_matrix(
        {(o, m) + (s,): v for (o, m, s), v in strat_cells.items()},
        strategies, [(o, m) for o in ontologies for m in models])
    strategy_result = _friedman_block(
        "prompting strategy", mat_s, strategies,
        "blocked by (ontology, model)")

    # Model effect: blocks are (ontology, strategy).
    mat_m, kept_m = _blocked_matrix(
        {(o, s) + (m,): v for (o, m, s), v in strat_cells.items()},
        models, [(o, s) for o in ontologies for s in strategies])
    model_result = _friedman_block(
        "model", mat_m, models, "blocked by (ontology, strategy)")

    return {
        "metric": metric,
        "metric_note": ("fol_parse_success_rate is the Phase 1 outcome with "
                        "variance; contradiction_rate is 0 in every cell and "
                        "admits no comparative test."),
        "design": {
            "experimental_unit": "one (ontology, model, strategy) cell",
            "n_cells_total": len(values),
            "cells_per_model": len(ontologies) * len(strategies),
            "n_ontologies": len(ontologies),
            "n_models": len(models),
            "n_strategies": len(strategies),
        },
        "strategy_effect": strategy_result,
        "model_effect": model_result,
        "multiple_comparison_adjustment": "Holm-Bonferroni within each family",
    }

# --------------------------------------------------------------------------
# Task: phase2_rates
# --------------------------------------------------------------------------
def task_phase2_rates(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    from scipy.stats import binomtest
    pattern = os.path.join(cfg.paths.results_dir, cfg.paths.phase2_glob)
    rows = read_rows(pattern)
    conf = cfg.stats.conf_level

    specs = [
        ("symmetry", "n_symmetry_violations", "n_pairs_checked", "is_symmetric"),
        ("transitivity", "n_transitivity_violations", "n_triples_checked", "is_transitive"),
        ("domain_range", None, "n_dr_checked", None),
    ]

    out: dict[str, Any] = {
        "null_hypotheses": {
            "H0_zero": ("no violations occur (rate = 0). A binomial test against "
                        "0 is degenerate, so this is decided by whether the exact "
                        "confidence interval excludes 0."),
            "H0_random": ("the model answers at chance (rate = 0.5). Evaluated "
                          "with a two-sided exact binomial test."),
        },
        "note": ("These are different questions and are reported separately. A "
                 "rate can exclude 0 and still be far below 0.5, which means "
                 "violations are real but the model is much better than chance."),
        "rates": [],
    }

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["ontology"], row["relation_label"])].append(row)

    for (onto, rel), group in sorted(grouped.items()):
        for kind, vio_col, n_col, flag_col in specs:
            if flag_col is not None and group[0].get(flag_col) != "1":
                continue
            if vio_col is None:
                # Domain-range errors are stored as a rate, not a count.
                n = sum(int(as_float(r.get(n_col)) or 0) for r in group)
                k = int(round(sum(as_float(r.get("domain_range_error_rate")) *
                                  (as_float(r.get(n_col)) or 0) for r in group)))
            else:
                n = sum(int(as_float(r.get(n_col)) or 0) for r in group)
                k = sum(int(as_float(r.get(vio_col)) or 0) for r in group)
            if n == 0:
                valid, why = rate_validity(onto, rel, kind)
                out["rates"].append({
                    "ontology": onto, "relation": rel, "kind": kind,
                    "valid": valid, "validity_note": why,
                    "k": 0, "n": 0,
                    "verdict": "not testable: the model asserted no qualifying "
                               "triples/pairs, so no violation can be observed",
                })
                continue

            lo, hi = clopper_pearson(k, n, conf)
            p_random = float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
            valid, why = rate_validity(onto, rel, kind)
            out["rates"].append({
                "ontology": onto, "relation": rel, "kind": kind,
                "valid": valid, "validity_note": why,
                "k": k, "n": n,
                "rate": round(k / n, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "excludes_zero": bool(lo > 0),
                "p_vs_random_0.5": p_random,
                "below_random": bool(k / n < 0.5),
            })
    return out

# --------------------------------------------------------------------------
# Task: c6_mixed
# --------------------------------------------------------------------------
def task_c6_mixed(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    import pandas as pd
    from scipy.stats import mannwhitneyu

    pattern = os.path.join(cfg.paths.results_dir, cfg.paths.phase3_glob)
    rows = read_rows(pattern)
    records = []
    for row in rows:
        cov = as_float(row.get("concept_coverage"))
        if np.isnan(cov):
            continue
        records.append({
            "ontology": row["ontology"],
            "model": row["llm_model"],
            "strategy": row["strategy"],
            "domain": DOMAIN.get(row["ontology"], "unknown"),
            "concept_coverage": cov,
        })
    frame = pd.DataFrame.from_records(records)

    general = frame[frame.domain == "general"].concept_coverage.to_numpy()
    technical = frame[frame.domain == "technical"].concept_coverage.to_numpy()

    result: dict[str, Any] = {
        "n_observations": int(len(frame)),
        "n_ontologies": int(frame.ontology.nunique()),
        "ontologies_per_domain": frame.groupby("domain").ontology.nunique().to_dict(),
        "cells_per_domain": frame.domain.value_counts().to_dict(),
        "mean_coverage_by_domain": frame.groupby("domain").concept_coverage.mean().round(4).to_dict(),
        "mean_coverage_by_ontology": frame.groupby("ontology").concept_coverage.mean().round(4).to_dict(),
    }

    # The test as submitted: cells treated as independent observations.
    u_stat, u_p = mannwhitneyu(general, technical, alternative="greater")
    n1, n2 = len(general), len(technical)
    z = (u_stat - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    result["mann_whitney_as_submitted"] = {
        "U": float(u_stat), "p_one_tailed": float(u_p),
        "n_general_cells": n1, "n_technical_cells": n2,
        "effect_size_r": round(float(z / np.sqrt(n1 + n2)), 4),
        "note": ("treats each model/strategy cell as a replicate of its domain; "
                 "this is the pseudoreplication this analysis identifies"),
    }

    # The ontology-level test: one observation per ontology, which is the real
    # replicate. With 2 vs 3 ontologies this has almost no power, and saying so
    # is part of the answer.
    per_onto = frame.groupby(["ontology", "domain"]).concept_coverage.mean().reset_index()
    g_onto = per_onto[per_onto.domain == "general"].concept_coverage.to_numpy()
    t_onto = per_onto[per_onto.domain == "technical"].concept_coverage.to_numpy()
    u2, p2 = mannwhitneyu(g_onto, t_onto, alternative="greater")
    result["mann_whitney_ontology_level"] = {
        "U": float(u2), "p_one_tailed": float(p2),
        "n_general_ontologies": int(len(g_onto)),
        "n_technical_ontologies": int(len(t_onto)),
        "note": ("the ontology is the true replicate; with 2 vs 3 ontologies the "
                 "smallest attainable one-tailed p is 0.1, so this design cannot "
                 "reach significance at the ontology level whatever the data"),
        "min_attainable_p": 0.1,
    }

    # Leave-one-ontology-out: if the domain contrast depends on a single
    # ontology, it is an ontology effect wearing a domain label.
    loo = {}
    for dropped in sorted(frame.ontology.unique()):
        sub = frame[frame.ontology != dropped]
        g = sub[sub.domain == "general"].concept_coverage.to_numpy()
        t = sub[sub.domain == "technical"].concept_coverage.to_numpy()
        if len(g) == 0 or len(t) == 0:
            loo[dropped] = {"note": "dropping this ontology empties a domain"}
            continue
        u, p = mannwhitneyu(g, t, alternative="greater")
        loo[dropped] = {
            "U": float(u), "p_one_tailed": float(p),
            "mean_general": round(float(g.mean()), 4),
            "mean_technical": round(float(t.mean()), 4),
            "still_significant_at_0.05": bool(p < 0.05),
        }
    result["leave_one_ontology_out"] = loo

    # Mixed model: domain as fixed effect, ontology as random intercept.
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm("concept_coverage ~ domain", frame,
                                groups=frame["ontology"])
            fit = model.fit(reml=True, method="lbfgs")
        term = [t for t in fit.params.index if t.startswith("domain")]
        key = term[0] if term else None
        result["mixed_effects"] = {
            "formula": "concept_coverage ~ domain + (1 | ontology)",
            "fixed_effect_term": key,
            "coefficient": (None if key is None else round(float(fit.params[key]), 5)),
            "std_error": (None if key is None else round(float(fit.bse[key]), 5)),
            "z": (None if key is None else round(float(fit.tvalues[key]), 4)),
            "p": (None if key is None else float(fit.pvalues[key])),
            "group_variance": round(float(fit.cov_re.iloc[0, 0]), 6),
            "n_groups": int(frame.ontology.nunique()),
            "note": ("ontology enters as a random intercept, so the domain "
                     "contrast is evaluated against between-ontology variance "
                     "rather than against cell-to-cell noise"),
        }
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        result["mixed_effects"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "interpretation": (
                "domain is constant within each ontology, so it is a "
                "between-group covariate estimated from 5 groups (2 general, "
                "3 technical). The random-intercept model is not identifiable "
                "at this group count and the fit is singular. This is a "
                "property of the design, not a numerical accident: the data "
                "cannot separate a domain effect from an ontology effect."),
        }

    return result

# --------------------------------------------------------------------------
# Task: phase2b_strategy
# --------------------------------------------------------------------------
def task_phase2b_strategy(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    """Does prompting strategy move subClassOf transitivity violations?

    The manuscript asserts relational inconsistency is 'impervious to prompting'.
    Phase 2b is the only relational experiment that covers all four strategies,
    so it is the only place that assertion can be tested. Tested here as a
    blocked comparison, with the incompleteness of the grid stated explicitly.
    """
    pattern = os.path.join(cfg.paths.results_dir, "**/phase2b_summary.csv")
    rows = read_rows(pattern)

    cells: dict[tuple, float] = {}
    counts: dict[tuple, tuple[int, int]] = {}
    for row in rows:
        key = (row["onto"], row["model"], row["strategy"])
        cells[key] = as_float(row["subclassof_transitivity_violation_rate"])
        counts[key] = (int(as_float(row["n_violations"]) or 0),
                       int(as_float(row["n_triples"]) or 0))

    strategies = list(cfg.grid.strategies)
    blocks = sorted({(o, m) for (o, m, _) in cells})
    matrix, kept = _blocked_matrix(
        {(o, m, s): v for (o, m, s), v in cells.items()}, strategies, blocks)

    result = _friedman_block("prompting strategy (Phase 2b, subClassOf)",
                             matrix, strategies, "blocked by (ontology, model)")
    result["grid_completeness"] = {
        "n_cells_present": len(cells),
        "n_cells_for_full_grid": len(cfg.grid.ontologies) * len(cfg.grid.models) * len(strategies),
        "n_complete_blocks_used": len(kept),
        "blocks_used": [list(b) for b in kept],
        "note": ("Only complete (ontology, model) blocks enter the test. Cells "
                 "outside these blocks are not covered by any strategy "
                 "comparison and the manuscript must not generalise to them."),
    }

    # Per-cell rates with exact intervals, so the table can report CIs.
    per_cell = []
    for key in sorted(counts):
        k, n = counts[key]
        lo, hi = clopper_pearson(k, n, cfg.stats.conf_level)
        per_cell.append({
            "ontology": key[0], "model": key[1], "strategy": key[2],
            "k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        })
    result["per_cell_rates"] = per_cell
    result["scope_warning"] = (
        "This tests subClassOf only. The named-property experiment (symmetry, "
        "transitivity, domain-range) was run for zero_shot and cot only, so no "
        "claim about tot or self_consistency on named properties is supported.")
    return result

# --------------------------------------------------------------------------
# Task: phase3_precision
# --------------------------------------------------------------------------
def task_phase3_precision(cfg: DictConfig, out_dir: str) -> dict[str, Any]:
    """Add precision, F1 and hallucination rate to the Phase 3 evaluation.

    The submitted analysis reported concept_coverage only, which is recall:
    |GT ∩ LLM| / |GT|. It says nothing about how much the model invented. Both
    |GT| and |LLM| are stored per cell, and coverage is stored to 4 decimals over
    a 50-concept ground truth, so the intersection size is recoverable exactly
    (the rounding error is far below 0.5 of a concept). No re-runs are required.
    """
    pattern = os.path.join(cfg.paths.results_dir, cfg.paths.phase3_glob)
    rows = read_rows(pattern)
    conf = cfg.stats.conf_level

    per_cell, skipped = [], []
    for row in rows:
        recall = as_float(row.get("concept_coverage"))
        n_gt = as_float(row.get("n_gt_concepts"))
        n_llm = as_float(row.get("n_llm_concepts"))
        if np.isnan(recall) or np.isnan(n_gt) or n_gt <= 0:
            skipped.append({k: row.get(k) for k in ("ontology", "llm_model", "strategy")})
            continue
        inter = int(round(recall * n_gt))
        precision = (inter / n_llm) if n_llm and n_llm > 0 else float("nan")
        f1 = ((2 * precision * recall / (precision + recall))
              if (precision == precision and (precision + recall) > 0) else float("nan"))
        hallucinated = (n_llm - inter) if (n_llm and n_llm > 0) else float("nan")

        e_recall = as_float(row.get("edge_coverage"))
        n_gt_e = as_float(row.get("n_gt_edges"))
        n_llm_e = as_float(row.get("n_llm_edges"))
        e_inter = int(round(e_recall * n_gt_e)) if (not np.isnan(e_recall)
                                                    and not np.isnan(n_gt_e)) else None
        e_precision = ((e_inter / n_llm_e)
                       if (e_inter is not None and n_llm_e and n_llm_e > 0) else float("nan"))

        per_cell.append({
            "ontology": row["ontology"],
            "llm_model": row["llm_model"],
            "strategy": row["strategy"],
            "n_gt_concepts": int(n_gt),
            "n_llm_concepts": int(n_llm) if n_llm == n_llm else None,
            "n_intersect": inter,
            "concept_recall": round(recall, 4),
            "concept_precision": (None if precision != precision else round(precision, 4)),
            "concept_f1": (None if f1 != f1 else round(f1, 4)),
            "n_hallucinated_concepts": (None if hallucinated != hallucinated
                                        else int(hallucinated)),
            "edge_recall": (None if e_recall != e_recall else round(e_recall, 4)),
            "edge_precision": (None if e_precision != e_precision else round(e_precision, 4)),
            "n_gt_edges": (None if n_gt_e != n_gt_e else int(n_gt_e)),
            "n_llm_edges": (None if n_llm_e != n_llm_e else int(n_llm_e)),
        })

    with open(os.path.join(out_dir, "phase3_precision_recall.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_cell[0].keys()))
        writer.writeheader()
        writer.writerows(per_cell)

    def _mean(key: str) -> float | None:
        vals = [c[key] for c in per_cell if c[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    by_model: dict[str, dict] = {}
    for model in sorted({c["llm_model"] for c in per_cell}):
        sub = [c for c in per_cell if c["llm_model"] == model]
        by_model[model] = {
            "recall": round(float(np.mean([c["concept_recall"] for c in sub])), 4),
            "precision": round(float(np.mean([c["concept_precision"] for c in sub
                                              if c["concept_precision"] is not None])), 4),
            "f1": round(float(np.mean([c["concept_f1"] for c in sub
                                       if c["concept_f1"] is not None])), 4),
            "mean_hallucinated": round(float(np.mean([c["n_hallucinated_concepts"] for c in sub
                                                      if c["n_hallucinated_concepts"] is not None])), 2),
        }

    n_zero_gt_edges = sum(1 for c in per_cell if c["n_gt_edges"] in (0, None))
    return {
        "n_cells": len(per_cell),
        "n_cells_skipped": len(skipped),
        "cells_skipped": skipped,
        "overall": {
            "concept_recall": _mean("concept_recall"),
            "concept_precision": _mean("concept_precision"),
            "concept_f1": _mean("concept_f1"),
            "mean_hallucinated_concepts": _mean("n_hallucinated_concepts"),
        },
        "by_model": by_model,
        "edge_level_note": (
            f"{n_zero_gt_edges} of {len(per_cell)} cells have no ground-truth edges, "
            "so edge recall is undefined for them. This is the same limitation that "
            "made onto_sim_structural uninformative and must be stated, not averaged over."),
        "derivation_note": (
            "intersection = round(concept_coverage * n_gt_concepts); exact because "
            "coverage is stored to 4 decimals over a 50-concept ground truth."),
        "conf_level": conf,
    }

TASKS = {
    "paired_phase1": task_paired_phase1,
    "phase2_rates": task_phase2_rates,
    "c6_mixed": task_c6_mixed,
    "phase2b_strategy": task_phase2b_strategy,
    "phase3_precision": task_phase3_precision,
}

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    if cfg.task not in TASKS:
        sys.exit(f"unknown task {cfg.task!r}; choose one of {sorted(TASKS)}")
    root = hydra.utils.get_original_cwd()
    os.chdir(root)
    out_dir = os.path.join(root, cfg.paths.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary = TASKS[cfg.task](cfg, out_dir)
    out_path = os.path.join(out_dir, f"{cfg.task}_summary.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\nwritten: {out_path}")

if __name__ == "__main__":
    main()
