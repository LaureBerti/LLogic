"""Score Phase 3 concept recall against the concepts the model was actually shown.

Why. Phase 3 scores recall against the whole sampled ontology (36-50 concepts), while the
prompt carries only the last ``context_turns`` conversation turns -- two per concept, so
fifteen concepts at the default of 30. Recall against the full sample therefore conflates
what the model failed to recall with what it was never shown, and is a lower bound rather
than an estimate. The reconstruction target should correspond to information actually
available in the supplied context; this script computes that other denominator.

The context is deterministic: Phase 3 builds it by walking ``phase1_results.csv`` in order,
appending two turns per row, then truncating to the last ``context_turns``. The
reconstructions are stored as ``reconstruction.txt``. So no LLM call is needed -- this is a
reanalysis of files already on disk, which is precisely why it can be run at all.

Normalisation is not reimplemented: the same ``compute_coverage`` the pipeline uses is
called with the context concepts in place of the ground-truth set, so the two numbers differ
only in their denominator. (It deduplicates after normalising, so the denominator is the
distinct normalised label count, not the raw row count.)

The comparison is within-cell -- every cell yields both numbers from the same
reconstruction -- so the test is a paired Wilcoxon signed-rank, not an unpaired one.

Compute: CPU only, local, no network, no LLM calls. Seconds.

Run:
    python src/analysis/context_recall.py
    python src/analysis/context_recall.py context_recall.results_dir=outputs/results_seed123
    python src/analysis/context_recall.py --cfg job
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.metrics.ontology_similarity import (  # noqa: E402
    compute_coverage,
    parse_manchester_to_graph,
)

csv.field_size_limit(10 ** 9)


def context_concepts(phase1_csv: Path, context_turns: int) -> list[str]:
    """The concept labels the Phase 3 prompt actually carried.

    Mirrors ``run_phase3``: two turns per Phase 1 row in file order, truncated to the last
    ``context_turns``. Truncating turns rather than rows is what makes the count
    ``context_turns // 2``, and getting that wrong silently changes every denominator.
    """
    labels: list[str] = []
    with open(phase1_csv, newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            labels.append((row.get("concept_label") or "").strip())
    keep = max(context_turns // 2, 1)
    return labels[-keep:]


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    cr = cfg.context_recall
    results = root / str(cr.results_dir)
    turns = int(cr.context_turns)

    cells: list[dict] = []
    skipped: list[str] = []
    for p3 in sorted(results.glob("*/*/*/phase3_results.csv")):
        d = p3.parent
        p1 = d / "phase1_results.csv"
        rec = d / "reconstruction.txt"
        rows = list(csv.DictReader(open(p3, newline="", errors="replace")))
        with_ctx = [
            r for r in rows
            if (r.get("use_phase1_context") or "True").strip().lower()
            not in ("false", "0", "")
        ]
        if len(with_ctx) != 1 or not p1.exists() or not rec.exists():
            skipped.append(str(d.relative_to(root)))
            continue
        row = with_ctx[0]
        ctx = [c for c in context_concepts(p1, turns) if c]
        graph = parse_manchester_to_graph(rec.read_text(errors="replace"))
        llm = list(graph.nodes()) if graph else []
        cov = compute_coverage(ctx, llm, [], [])
        try:
            reported = float(row.get("concept_coverage") or "nan")
        except ValueError:
            reported = float("nan")
        cells.append({
            "ontology": row.get("ontology", d.parts[-3]),
            "model": row.get("llm_model", d.parts[-2]),
            "strategy": row.get("strategy", d.parts[-1]),
            "cc_full_sample": reported,
            "cc_context": cov["concept_coverage"],
            "n_context_concepts": cov["n_gt_concepts"],
            "n_llm_concepts": cov["n_llm_concepts"],
        })

    if not cells:
        raise SystemExit(f"no scorable Phase 3 cells under {results} -- nothing computed")

    full = [c["cc_full_sample"] for c in cells]
    ctxv = [c["cc_context"] for c in cells]

    print(f"\ncells scored: {len(cells)}" + (f"   skipped: {len(skipped)}" if skipped else ""))
    print(f"{'ontology':10}{'model':18}{'strategy':18}{'vs sample':>11}{'vs context':>12}{'n_ctx':>7}")
    for c in cells:
        print(f"{c['ontology']:10}{c['model']:18}{c['strategy']:18}"
              f"{c['cc_full_sample']:11.3f}{c['cc_context']:12.3f}{c['n_context_concepts']:7}")

    summary: dict = {
        "what": "Phase 3 concept recall against the context actually supplied, vs against "
                "the full sampled ontology. Reanalysis of stored reconstructions; no LLM calls.",
        "results_dir": str(cr.results_dir),
        "context_turns": turns,
        "n_cells": len(cells),
        "cc_full_sample": {
            "mean": round(statistics.mean(full), 4),
            "sd": round(statistics.pstdev(full), 4) if len(full) > 1 else 0.0,
            "min": round(min(full), 4), "max": round(max(full), 4),
        },
        "cc_context": {
            "mean": round(statistics.mean(ctxv), 4),
            "sd": round(statistics.pstdev(ctxv), 4) if len(ctxv) > 1 else 0.0,
            "min": round(min(ctxv), 4), "max": round(max(ctxv), 4),
        },
        "per_ontology": {},
        "cells": cells,
    }
    for onto in sorted({c["ontology"] for c in cells}):
        sub = [c for c in cells if c["ontology"] == onto]
        summary["per_ontology"][onto] = {
            "n": len(sub),
            "cc_full_sample": round(statistics.mean(x["cc_full_sample"] for x in sub), 4),
            "cc_context": round(statistics.mean(x["cc_context"] for x in sub), 4),
        }

    # Within-cell comparison: both numbers come from the same reconstruction.
    try:
        from scipy.stats import wilcoxon
        diffs = [b - a for a, b in zip(full, ctxv)]
        nz = [d for d in diffs if d != 0]
        if nz:
            st, p = wilcoxon(full, ctxv)
            summary["paired_wilcoxon"] = {
                "statistic": float(st), "p_value": float(p),
                "n_pairs": len(full), "n_nonzero": len(nz),
                "mean_difference_pp": round(100 * statistics.mean(diffs), 2),
            }
    except ImportError:
        summary["paired_wilcoxon"] = "scipy unavailable"

    out = root / str(cr.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))

    print(f"\n  vs full sample : {summary['cc_full_sample']['mean']:.3f} "
          f"+/- {summary['cc_full_sample']['sd']:.3f}  "
          f"[{summary['cc_full_sample']['min']:.3f}, {summary['cc_full_sample']['max']:.3f}]")
    print(f"  vs context     : {summary['cc_context']['mean']:.3f} "
          f"+/- {summary['cc_context']['sd']:.3f}  "
          f"[{summary['cc_context']['min']:.3f}, {summary['cc_context']['max']:.3f}]")
    pw = summary.get("paired_wilcoxon")
    if isinstance(pw, dict):
        print(f"  paired Wilcoxon: p={pw['p_value']:.3g} over {pw['n_pairs']} cells "
              f"({pw['n_nonzero']} non-tied), mean difference {pw['mean_difference_pp']:+.2f} pp")
    print(f"\nwritten: {out.relative_to(root)}")


if __name__ == "__main__":
    main()
