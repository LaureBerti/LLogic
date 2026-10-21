from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.metrics.ontology_similarity import (
    compute_coverage,
    parse_manchester_to_graph,
)
from src.phase3_reconstruct import _load_gt_graph

csv.field_size_limit(10 ** 9)


def context_concepts(phase1_csv: Path, context_turns: int) -> list[str]:
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
        ontology = row.get("ontology", d.parts[-3])
        _, gt_concepts, _ = _load_gt_graph(ontology, seed=int(cr.seed))
        reported = compute_coverage(gt_concepts, llm, [], [])["concept_coverage"]
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
