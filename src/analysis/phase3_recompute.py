from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

from src.metrics.ontology_similarity import (
    compute_coverage,
    parse_manchester_to_graph,
    semantic_similarity,
    structural_similarity,
)
from src.phase3_reconstruct import RESULT_COLS, _load_gt_graph


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    spec = cfg.phase3_recompute
    write = bool(spec.write)
    with_semantic = bool(spec.semantic)

    rows: list[dict] = []
    for results_dir in spec.results_dirs:
        base = root / results_dir
        for text_path in sorted(base.glob("**/reconstruction.txt")):
            cell = text_path.parent
            parts = cell.relative_to(base).parts
            if any(p in set(spec.exclude) for p in parts):
                continue
            if len(parts) >= 4 and parts[-4].startswith("seed"):
                seed_dir, ontology, model_dir, strategy = parts[-4:]
                seed = int("".join(c for c in seed_dir if c.isdigit()))
            elif len(parts) >= 3:
                seed_dir = f"seed{spec.default_seed}"
                ontology, model_dir, strategy = parts[-3:]
                seed = int(spec.default_seed)
            else:
                continue

            text = text_path.read_text(encoding="utf-8", errors="replace")
            graph = parse_manchester_to_graph(text)
            nodes = list(graph.nodes()) if graph else []
            edges = list(graph.edges()) if graph else []

            gt_graph, gt_concepts, gt_edges = _load_gt_graph(ontology, seed=seed)
            coverage = compute_coverage(gt_concepts, nodes, gt_edges, edges)
            structural = structural_similarity(gt_graph, graph)
            semantic = (semantic_similarity(gt_concepts, nodes,
                                            model_name=cfg.metrics.embed_model)
                        if with_semantic else float("nan"))

            csv_path = cell / "phase3_results.csv"
            previous: dict = {}
            if csv_path.exists() and csv_path.stat().st_size > 0:
                with open(csv_path, newline="", errors="replace") as handle:
                    existing = list(csv.DictReader(handle))
                if existing:
                    previous = existing[-1]

            row = {
                "ontology": ontology,
                "llm_model": previous.get("llm_model", model_dir),
                "strategy": strategy,
                "use_phase1_context": previous.get("use_phase1_context", True),
                "onto_sim_structural": structural,
                "onto_sim_semantic": semantic,
                **coverage,
                "reconstruction_text_chars": len(text),
                "time_search_s": previous.get("time_search_s", ""),
                "time_eval_s": previous.get("time_eval_s", ""),
                "n_candidates": previous.get("n_candidates", 1),
                "time_condition_s": previous.get("time_condition_s", ""),
            }
            rows.append({
                "_path": str(csv_path.relative_to(root)),
                "_seed": seed_dir,
                "_prev_concepts": int(previous.get("n_llm_concepts") or 0),
                "_prev_cov": float(previous.get("concept_coverage") or 0.0),
                **row,
            })

            if write:
                with open(csv_path, "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=RESULT_COLS)
                    writer.writeheader()
                    writer.writerow({k: row[k] for k in RESULT_COLS})

    regressions = [r for r in rows if r["n_llm_concepts"] < r["_prev_concepts"]]
    gained = [r for r in rows if r["n_llm_concepts"] > r["_prev_concepts"]]
    zero_before = sum(1 for r in rows if r["_prev_concepts"] == 0)
    zero_after = sum(1 for r in rows if r["n_llm_concepts"] == 0)

    print(f"cells: {len(rows)}")
    print(f"  concepts recovered: improved={len(gained)} regressed={len(regressions)}")
    print(f"  cells with zero concepts: before={zero_before} after={zero_after}")
    print(f"  cells with a non-empty ground-truth edge set: "
          f"{sum(1 for r in rows if r['n_gt_edges'] > 0)}")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["llm_model"], r["strategy"])].append(r)
    print("\nconcept_coverage")
    for key in sorted(grouped):
        vals = [r["concept_coverage"] for r in grouped[key]]
        print(f"  {key[0]:16s} {key[1]:10s} n={len(vals):2d} "
              f"mean={statistics.mean(vals):.4f} "
              f"sd={statistics.stdev(vals) if len(vals) > 1 else 0.0:.4f} "
              f"max={max(vals):.4f}")
    print("\nedge_coverage over cells with ground-truth edges")
    for key in sorted(grouped):
        vals = [r["edge_coverage"] for r in grouped[key] if r["n_gt_edges"] > 0]
        if not vals:
            print(f"  {key[0]:16s} {key[1]:10s} no cell has ground-truth edges")
            continue
        print(f"  {key[0]:16s} {key[1]:10s} n={len(vals):2d} "
              f"mean={statistics.mean(vals):.4f} max={max(vals):.4f}")

    if spec.report_json:
        out = root / spec.report_json
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [{k: (str(v) if isinstance(v, Path) else v)
                    for k, v in r.items()} for r in rows]
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nreport -> {spec.report_json}")

    if not write:
        print("\nwrite=false — no phase3_results.csv was modified. "
              "Re-run with phase3_recompute.write=true to persist.")


if __name__ == "__main__":
    main()
