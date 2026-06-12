"""Phase 3: LLM-driven ontology reconstruction + similarity metrics."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.prompting.prompter import reconstruction_prompt
from src.prompting.llm_client import get_client, call_llm
from src.metrics.ontology_similarity import (
    parse_manchester_to_graph,
    structural_similarity,
    semantic_similarity,
    compute_coverage,
)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


RESULT_COLS = [
    "ontology", "llm_model", "strategy", "use_phase1_context",
    "onto_sim_structural", "onto_sim_semantic",
    "concept_coverage", "edge_coverage",
    "n_gt_concepts", "n_llm_concepts", "n_gt_edges", "n_llm_edges",
    "reconstruction_text_chars",
    "time_search_s", "time_eval_s", "n_candidates", "time_condition_s",
]


def _load_gt_graph(ontology_name: str, seed: int = 42) -> tuple[Any, list[str], list[tuple]]:
    """Load ground-truth ontology as networkx graph from the concept sample (proxy)."""
    samples_path = Path("data/samples") / f"concepts_{ontology_name}_seed{seed}.json"
    edges_path = Path("data/samples") / f"edges_{ontology_name}_seed{seed}.json"

    with open(samples_path) as f:
        concepts = json.load(f)

    gt_concepts = [c["label"] for c in concepts]
    gt_edges = []
    for c in concepts:
        for parent_iri in c.get("parents", []):
            parent_label = parent_iri.split("/")[-1].split("#")[-1]
            if parent_label:
                gt_edges.append((c["label"], parent_label))

    if not HAS_NX:
        return None, gt_concepts, gt_edges

    g = nx.DiGraph()
    for (child, parent) in gt_edges:
        g.add_edge(child, parent, rel="subClassOf")
    return g, gt_concepts, gt_edges


def run_phase3(cfg: Any, out_dir: Path) -> None:
    use_ctx: bool = bool(cfg.phase3.use_phase1_context)

    # Build conversation history from phase 1 results (or leave empty for control)
    conversation_history: list[dict] = []
    if use_ctx:
        phase1_path = out_dir / "phase1_results.csv"
        if not phase1_path.exists():
            raise FileNotFoundError("Run phase=1 first (phase1_results.csv needed to build conversation history).")
        with open(phase1_path) as f:
            for row in csv.DictReader(f):
                concept_label = row["concept_label"]
                fol_block = row.get("fol_block", "")
                conversation_history.append({"role": "user", "content": f"Define '{concept_label}' in FOL."})
                conversation_history.append({"role": "assistant", "content": fol_block or "[no FOL block]"})
        # Truncate to last 30 turns to stay within context window
        if len(conversation_history) > 30:
            conversation_history = conversation_history[-30:]
    else:
        print("  Phase 3 control mode: no Phase 1 context injected.")

    client = get_client(cfg)
    model = cfg.llm.model
    ontology = cfg.ontology.name
    strategy = cfg.prompting.strategy

    # Resume: skip if a row with the same use_phase1_context value already exists
    results_path = out_dir / "phase3_results.csv"
    if results_path.exists() and results_path.stat().st_size > 0:
        with open(results_path) as f:
            existing = list(csv.DictReader(f))
        for existing_row in existing:
            row_ctx = existing_row.get("use_phase1_context", "True")
            # Normalise string booleans from CSV
            row_ctx_bool = row_ctx.strip().lower() not in ("false", "0", "")
            if row_ctx_bool == use_ctx:
                ctx_label = "with-context" if use_ctx else "no-context"
                print(f"  SKIP {ontology} × {model} × {strategy} ({ctx_label}) — phase3 already done.")
                return

    t0 = time.time()
    messages = reconstruction_prompt(conversation_history, ontology)

    t_search_start = time.time()
    reconstruction_text = call_llm(client, model, messages,
                                   temperature=cfg.llm.temperature,
                                   max_tokens=2048,
                                   timeout=float(getattr(cfg.llm, "timeout_s", 120.0)))
    t_search = round(time.time() - t_search_start, 3)

    # Save raw reconstruction
    (out_dir / "reconstruction.txt").write_text(reconstruction_text)

    t_eval_start = time.time()
    llm_graph = parse_manchester_to_graph(reconstruction_text)
    gt_graph, gt_concepts, gt_edges = _load_gt_graph(ontology, seed=int(cfg.sampling.seed))

    llm_concepts = list(llm_graph.nodes()) if llm_graph else []
    llm_edges = list(llm_graph.edges()) if llm_graph else []

    sim_struct = structural_similarity(gt_graph, llm_graph)
    sim_sem = semantic_similarity(gt_concepts, llm_concepts, model_name=cfg.metrics.embed_model)
    coverage = compute_coverage(gt_concepts, llm_concepts, gt_edges, llm_edges)
    t_eval = round(time.time() - t_eval_start, 3)
    t_total = round(time.time() - t0, 3)

    results_path = out_dir / "phase3_results.csv"
    row = {
        "ontology": ontology,
        "llm_model": model,
        "strategy": strategy,
        "use_phase1_context": use_ctx,
        "onto_sim_structural": sim_struct,
        "onto_sim_semantic": sim_sem,
        **coverage,
        "reconstruction_text_chars": len(reconstruction_text),
        "time_search_s": t_search,
        "time_eval_s": t_eval,
        "n_candidates": 1,
        "time_condition_s": t_total,
    }

    write_header = not (results_path.exists() and results_path.stat().st_size > 0)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n{'─'*60}")
    print(f"Phase 3 complete — {ontology} × {model} × {strategy}")
    print(f"  Structural similarity : {sim_struct}")
    print(f"  Semantic similarity   : {sim_sem}")
    print(f"  Concept coverage      : {coverage['concept_coverage']}")
    print(f"  Edge coverage         : {coverage['edge_coverage']}")
    print(f"  Total wall time       : {t_total:.1f}s")
    print(f"{'─'*60}")
