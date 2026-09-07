"""Phase 2: relationship-level consistency testing (transitivity, symmetry, domain/range)."""
from __future__ import annotations
import csv
import json
import itertools
import time
from pathlib import Path
from typing import Any

from src.prompting.prompter import relation_consistency_prompt
from src.prompting.llm_client import get_client, call_llm
from src.metrics.consistency_metrics import check_transitivity, check_symmetry, check_domain_range

RESULT_COLS = [
    "ontology", "llm_model", "strategy", "relation_label",
    "is_transitive", "is_symmetric",
    "transitivity_violation_rate", "n_triples_checked", "n_transitivity_violations",
    "symmetry_violation_rate", "n_pairs_checked", "n_symmetry_violations",
    "domain_range_error_rate", "n_dr_checked",
    "time_search_s", "time_eval_s", "n_candidates", "time_condition_s",
]

def run_phase2(cfg: Any, out_dir: Path) -> None:
    concepts_path = Path("data/samples") / f"concepts_{cfg.ontology.name}_seed{cfg.sampling.seed}.json"
    edges_path = Path("data/samples") / f"edges_{cfg.ontology.name}_seed{cfg.sampling.seed}.json"

    if not concepts_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Run phase=0 first to generate samples.")

    with open(concepts_path) as f:
        concepts = json.load(f)
    with open(edges_path) as f:
        relations = json.load(f)

    results_path = out_dir / "phase2_results.csv"
    completed_keys: set[str] = set()

    if results_path.exists() and results_path.stat().st_size > 0:
        with open(results_path) as f:
            for row in csv.DictReader(f):
                completed_keys.add(row["relation_label"])
        print(f"  Resume mode: {len(completed_keys)} relations already done.")
        write_mode, write_header = "a", False
    else:
        write_mode, write_header = "w", True

    client = get_client(cfg)
    model = cfg.llm.model
    ontology = cfg.ontology.name
    strategy = cfg.prompting.strategy
    t0_total = time.time()
    timing_rows: list[dict] = []

    # Use a small subset of concepts for pair probing (max 10, → up to 90 pairs)
    probe_concepts = concepts[:10]
    pairs = list(itertools.combinations(probe_concepts, 2))

    with open(results_path, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()

        for relation in relations:
            rel_label = relation["label"]
            if rel_label in completed_keys:
                continue

            t0_rel = time.time()
            llm_responses: dict[tuple, str] = {}
            dr_errors = 0

            t_search_start = time.time()
            for (ca, cb) in pairs:
                messages = relation_consistency_prompt(relation, ca, cb)
                response = call_llm(client, model, messages,
                                    temperature=cfg.llm.temperature,
                                    max_tokens=256)
                llm_responses[(ca["label"], cb["label"])] = response

                dr = check_domain_range(response, relation, ca, cb)
                dr_errors += dr["domain_range_error"]
            t_search = round(time.time() - t_search_start, 3)

            t_eval_start = time.time()
            trans = check_transitivity(llm_responses, relation)
            sym = check_symmetry(llm_responses, relation)
            t_eval = round(time.time() - t_eval_start, 3)

            n_dr = len(pairs)
            dr_rate = round(dr_errors / n_dr, 4) if n_dr > 0 else 0.0
            t_condition = round(time.time() - t0_rel, 3)

            row = {
                "ontology": ontology,
                "llm_model": model,
                "strategy": strategy,
                "relation_label": rel_label,
                "is_transitive": int(relation.get("is_transitive", False)),
                "is_symmetric": int(relation.get("is_symmetric", False)),
                "transitivity_violation_rate": trans.get("transitivity_violation_rate"),
                "n_triples_checked": trans.get("n_triples_checked", 0),
                "n_transitivity_violations": trans.get("n_violations", 0),
                "symmetry_violation_rate": sym.get("symmetry_violation_rate"),
                "n_pairs_checked": sym.get("n_pairs_checked", 0),
                "n_symmetry_violations": sym.get("n_violations", 0),
                "domain_range_error_rate": dr_rate,
                "n_dr_checked": n_dr,
                "time_search_s": t_search,
                "time_eval_s": t_eval,
                "n_candidates": len(pairs),
                "time_condition_s": t_condition,
            }
            writer.writerow(row)
            f.flush()
            timing_rows.append({
                "ontology": ontology, "llm": model, "strategy": strategy,
                "relation": rel_label,
                "time_condition_s": t_condition, "time_search_s": t_search,
                "time_eval_s": t_eval, "n_candidates": len(pairs),
            })

            print(f"  {rel_label:<25} | trans_viol={trans.get('transitivity_violation_rate', 'N/A')} "
                  f"| sym_viol={sym.get('symmetry_violation_rate', 'N/A')} "
                  f"| dr_err={dr_rate} | {t_condition:.2f}s")

    if timing_rows:
        _write_timing(timing_rows, out_dir)

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"Phase 2 complete — {ontology} × {model} × {strategy}")
    print(f"  Relations processed: {len(timing_rows)}")
    print(f"  Total wall time    : {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"{'─'*60}")

def _write_timing(timing_rows: list[dict], out_dir: Path) -> None:
    import csv as _csv
    tc_path = out_dir / "timing_per_condition.csv"
    with open(tc_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)
