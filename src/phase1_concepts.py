"""Phase 1: concept-level FOL generation + solver-based contradiction detection."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.prompting.prompter import build_prompts
from src.prompting.llm_client import get_client, call_llm, call_self_consistency
from src.parsing.fol_parser import extract_fol_block, parse_to_z3, check_satisfiability
from src.metrics.consistency_metrics import compute_concept_metrics, majority_vote, aggregate_fol_responses

RESULT_COLS = [
    "ontology", "llm_model", "strategy", "concept_iri", "concept_label",
    "concept_depth", "concept_stratum",
    "fol_block", "fol_parse_success", "solver_result", "is_tautology",
    "contradiction", "response_text",
    "time_search_s", "time_eval_s", "n_candidates", "time_condition_s",
]

def run_phase1(cfg: Any, out_dir: Path) -> None:
    samples_path = Path("data/samples") / f"concepts_{cfg.ontology.name}_seed{cfg.sampling.seed}.json"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Concept sample file not found: {samples_path}\n"
            f"Run phase=0 first: python src/main.py phase=0 ontology.name={cfg.ontology.name}"
        )

    with open(samples_path) as f:
        concepts = json.load(f)

    results_path = out_dir / "phase1_results.csv"
    completed_keys: set[str] = set()

    # Resume logic — never truncate existing results
    if results_path.exists() and results_path.stat().st_size > 0:
        with open(results_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_keys.add(f"{row['concept_iri']}|{row['strategy']}")
        print(f"  Resume mode: {len(completed_keys)} concepts already done.")
        write_mode = "a"
        write_header = False
    else:
        write_mode = "w"
        write_header = True

    client = get_client(cfg)
    strategy = cfg.prompting.strategy
    model = cfg.llm.model
    ontology = cfg.ontology.name
    t0_total = time.time()
    all_rows: list[dict] = []
    timing_rows: list[dict] = []

    with open(results_path, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()

        for concept in concepts:
            key = f"{concept['iri']}|{strategy}"
            if key in completed_keys:
                continue

            t0_concept = time.time()
            label = concept["label"]

            # LLM call
            t_search_start = time.time()
            if strategy == "self_consistency":
                prompt_chains = build_prompts(concept, strategy, n_samples=cfg.llm.n_samples)
                responses = call_self_consistency(client, model, prompt_chains,
                                                  temperature=0.7, max_tokens=cfg.llm.max_tokens,
                                                  timeout=getattr(cfg.llm, "timeout_s", 120.0))
                response_text = aggregate_fol_responses(responses)
                n_candidates = len(responses)
            else:
                messages = build_prompts(concept, strategy)
                response_text = call_llm(client, model, messages,
                                         temperature=cfg.llm.temperature,
                                         max_tokens=cfg.llm.max_tokens,
                                         timeout=getattr(cfg.llm, "timeout_s", 120.0))
                if not response_text.strip():
                    # A reasoning model can spend the whole budget thinking and emit
                    # no answer: qwen3:30b-a3b returned finish_reason="length" with
                    # 7,883 characters of reasoning and empty content at 2,048 tokens.
                    # The client refuses to hand truncated reasoning to the parser --
                    # a verdict must not be read out of unfinished thought -- so the
                    # only honest recovery is to give the model room to finish. This
                    # is the Phase 1 counterpart of the retry in phase2b_subclassof.
                    response_text = call_llm(
                        client, model, messages,
                        temperature=cfg.llm.temperature,
                        max_tokens=int(getattr(cfg.llm, "max_tokens_retry", 8192)),
                        timeout=float(getattr(cfg.llm, "timeout_retry_s", 1800.0)))
                n_candidates = 1
            t_search = round(time.time() - t_search_start, 3)

            # FOL parsing + solving
            t_eval_start = time.time()
            fol_block = extract_fol_block(response_text) or ""
            z3_expr, parse_ok = parse_to_z3(fol_block, label)
            solver_result = check_satisfiability(z3_expr, timeout_ms=cfg.solver.timeout_s * 1000)
            is_tautology = (fol_block != "" and label.lower() in fol_block.lower()
                            and fol_block.count(label.lower()) >= 2
                            and "<->" in fol_block and fol_block.count("<->") == 1
                            and fol_block.split("<->")[0].strip().endswith(f"{label}(x)")
                            and fol_block.split("<->")[1].strip().startswith(label))
            contradiction = int(solver_result == "unsat")
            t_eval = round(time.time() - t_eval_start, 3)
            t_condition = round(time.time() - t0_concept, 3)

            row = {
                "ontology": ontology,
                "llm_model": model,
                "strategy": strategy,
                "concept_iri": concept["iri"],
                "concept_label": label,
                "concept_depth": concept.get("depth", -1),
                "concept_stratum": concept.get("stratum", "unknown"),
                "fol_block": fol_block[:500],  # truncate for CSV readability
                "fol_parse_success": int(parse_ok),
                "solver_result": solver_result,
                "is_tautology": int(is_tautology),
                "contradiction": contradiction,
                # Earlier runs stored only the first 300 characters of a reply.
                # A Chain-of-Thought reply spends that budget on Steps 1-3, so the
                # final biconditional at Step 5 was truncated away and it became
                # impossible to tell whether an extracted "Step 3: ..." scaffold
                # line meant the model failed or the extractor picked the wrong
                # line. Store the full response so that is decidable.
                "response_text": response_text,
                "time_search_s": t_search,
                "time_eval_s": t_eval,
                "n_candidates": n_candidates,
                "time_condition_s": t_condition,
            }
            writer.writerow(row)
            f.flush()
            all_rows.append(row)
            timing_rows.append({
                "ontology": ontology, "llm": model, "strategy": strategy,
                "concept": label, "stratum": concept.get("stratum", "?"),
                "time_condition_s": t_condition, "time_search_s": t_search,
                "time_eval_s": t_eval, "n_candidates": n_candidates,
            })

            print(f"  {label:<30} | solver={solver_result:<7} | contradiction={contradiction} | "
                  f"search={t_search:.2f}s eval={t_eval:.2f}s total={t_condition:.2f}s")

    # Save timing files
    if timing_rows:
        _write_timing(timing_rows, out_dir)

    agg = compute_concept_metrics(all_rows)

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"Phase 1 complete — {ontology} × {model} × {strategy}")
    print(f"  Concepts processed: {len(all_rows)}")
    print(f"  Contradiction rate: {agg.get('contradiction_rate', 'N/A')}")
    print(f"  FOL parse success:  {agg.get('fol_parse_success_rate', 'N/A')}")
    print(f"  Total wall time   : {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Results → {results_path}")
    print(f"{'─'*60}")

def _write_timing(timing_rows: list[dict], out_dir: Path) -> None:
    import csv as _csv
    tc_path = out_dir / "timing_per_condition.csv"
    with open(tc_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)

    # Per-dataset (= per-ontology here)
    td_path = out_dir / "timing_per_dataset.csv"
    total_time = sum(r["time_condition_s"] for r in timing_rows)
    with open(td_path, "w", newline="") as f:
        _csv.writer(f).writerow(["ontology", "total_time_s"])
        _csv.writer(f).writerow([timing_rows[0]["ontology"], round(total_time, 3)])
