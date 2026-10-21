from __future__ import annotations

import csv
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

from src.metrics.ontology_similarity import (
    _norm_label,
    parse_manchester_to_graph,
)
from src.phase3_reconstruct import _load_gt_graph

COLS = [
    "ontology", "llm_model", "strategy",
    "n_gt_concepts", "n_llm_concepts", "n_intersect",
    "concept_recall", "concept_precision", "concept_f1",
    "n_hallucinated_concepts",
    "edge_recall", "edge_precision", "n_gt_edges", "n_llm_edges",
]

DISPLAY = {
    "gemma2_2b": "gemma2:2b",
    "llama3.2_latest": "llama3.2:latest",
    "mistral_7b": "mistral:7b",
    "qwen2.5_7b": "qwen2.5:7b",
    "gemma4_latest": "gemma4:latest",
    "qwen3_8b": "qwen3:8b",
}


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    spec = cfg.phase3_pr
    base = root / spec.results_dir
    seed = int(spec.seed)
    keep = set(spec.models)

    rows = []
    for text_path in sorted(base.glob("*/*/*/reconstruction.txt")):
        cell = text_path.parent
        ontology, model_dir, strategy = cell.relative_to(base).parts[-3:]
        if keep and model_dir not in keep:
            continue
        graph = parse_manchester_to_graph(
            text_path.read_text(encoding="utf-8", errors="replace"))
        nodes = {_norm_label(n) for n in (graph.nodes() if graph else [])}
        edges = {(_norm_label(a), _norm_label(b))
                 for a, b in (graph.edges() if graph else [])}

        _, gt_concepts, gt_edges = _load_gt_graph(ontology, seed=seed)
        gt_nodes = {_norm_label(c) for c in gt_concepts}
        gt_edge_set = {(_norm_label(a), _norm_label(b)) for a, b in gt_edges}

        hit = len(nodes & gt_nodes)
        precision = hit / len(nodes) if nodes else 0.0
        recall = hit / len(gt_nodes) if gt_nodes else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        edge_hit = len(edges & gt_edge_set)

        rows.append({
            "ontology": ontology,
            "llm_model": DISPLAY.get(model_dir, model_dir),
            "strategy": strategy,
            "n_gt_concepts": len(gt_nodes),
            "n_llm_concepts": len(nodes),
            "n_intersect": hit,
            "concept_recall": round(recall, 4),
            "concept_precision": round(precision, 4),
            "concept_f1": round(f1, 4),
            "n_hallucinated_concepts": len(nodes - gt_nodes),
            "edge_recall": round(edge_hit / len(gt_edge_set), 4) if gt_edge_set else "",
            "edge_precision": round(edge_hit / len(edges), 4) if edges else "",
            "n_gt_edges": len(gt_edge_set),
            "n_llm_edges": len(edges),
        })

    out = root / spec.out_csv
    print(f"cells: {len(rows)}")
    degenerate = sum(1 for r in rows
                     if r["concept_precision"] + r["concept_recall"] == 0)
    print(f"cells with no scorable concept overlap: {degenerate}")
    print(f"mean precision {sum(r['concept_precision'] for r in rows)/len(rows):.4f}  "
          f"mean recall {sum(r['concept_recall'] for r in rows)/len(rows):.4f}")

    if not spec.write:
        print(f"write=false — {spec.out_csv} not modified.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {spec.out_csv}")


if __name__ == "__main__":
    main()
