from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

from src.metrics.ontology_similarity import compute_coverage, parse_manchester_to_graph
from src.phase3_reconstruct import _load_gt_graph

DISPLAY = {
    "gemma2_2b": "gemma2:2b",
    "llama3.2_latest": "llama3.2:latest",
    "mistral_7b": "mistral:7b",
    "qwen2.5_7b": "qwen2.5:7b",
}


def read_arm(base: Path, seed: int, keep: set[str]) -> dict:
    out = {}
    for text_path in sorted(base.glob("*/*/*/reconstruction.txt")):
        cell = text_path.parent
        ontology, model_dir, strategy = cell.relative_to(base).parts[-3:]
        if keep and model_dir not in keep:
            continue
        graph = parse_manchester_to_graph(
            text_path.read_text(encoding="utf-8", errors="replace"))
        nodes = list(graph.nodes()) if graph else []
        _, gt_concepts, _ = _load_gt_graph(ontology, seed=seed)
        cov = compute_coverage(gt_concepts, nodes, [], [])
        out[(ontology, model_dir, strategy)] = {
            "coverage": cov["concept_coverage"],
            "n_llm_concepts": len(nodes),
        }
    return out


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    spec = cfg.phase3_fullctx
    keep = set(spec.models)
    seed = int(spec.seed)

    published = read_arm(root / spec.published_dir, seed, keep)
    fullctx = read_arm(root / spec.fullctx_dir, seed, keep)
    keys = sorted(set(published) & set(fullctx))

    a = [published[k]["coverage"] for k in keys]
    b = [fullctx[k]["coverage"] for k in keys]
    deltas = [y - x for x, y in zip(a, b)]

    both = [k for k in keys
            if published[k]["n_llm_concepts"] > 0 and fullctx[k]["n_llm_concepts"] > 0]
    a2 = [published[k]["coverage"] for k in both]
    b2 = [fullctx[k]["coverage"] for k in both]

    zero_pub = [k for k in keys if published[k]["n_llm_concepts"] == 0]
    zero_full = [k for k in keys if fullctx[k]["n_llm_concepts"] == 0]

    per_onto = {}
    for ontology in sorted({k[0] for k in keys}):
        sub = [k for k in keys if k[0] == ontology]
        per_onto[ontology] = {
            "n": len(sub),
            "published": round(statistics.mean(published[k]["coverage"] for k in sub), 4),
            "fullctx": round(statistics.mean(fullctx[k]["coverage"] for k in sub), 4),
        }

    report = {
        "n_paired": len(keys),
        "published_mean": round(statistics.mean(a), 4),
        "fullctx_mean": round(statistics.mean(b), 4),
        "published_sd": round(statistics.stdev(a), 4),
        "fullctx_sd": round(statistics.stdev(b), 4),
        "delta": round(statistics.mean(b) - statistics.mean(a), 4),
        "median_delta": round(statistics.median(deltas), 4),
        "wilcoxon_p": float(wilcoxon(a, b).pvalue),
        "untied": sum(1 for d in deltas if d != 0),
        "improved": sum(1 for d in deltas if d > 0),
        "unchanged": sum(1 for d in deltas if d == 0),
        "worse": sum(1 for d in deltas if d < 0),
        "zero_class_cells_published": len(zero_pub),
        "zero_class_cells_fullctx": len(zero_full),
        "parse_matched": {
            "n": len(both),
            "published_mean": round(statistics.mean(a2), 4),
            "fullctx_mean": round(statistics.mean(b2), 4),
            "delta": round(statistics.mean(b2) - statistics.mean(a2), 4),
            "wilcoxon_p": float(wilcoxon(a2, b2).pvalue),
        },
        "per_ontology": per_onto,
        "cells": [
            {
                "ontology": k[0],
                "model": DISPLAY.get(k[1], k[1]),
                "strategy": k[2],
                "published": published[k]["coverage"],
                "fullctx": fullctx[k]["coverage"],
                "published_n_concepts": published[k]["n_llm_concepts"],
                "fullctx_n_concepts": fullctx[k]["n_llm_concepts"],
            }
            for k in keys
        ],
    }

    print(f"paired cells: {report['n_paired']}")
    print(f"  full grid : {report['published_mean']} -> {report['fullctx_mean']} "
          f"(delta {report['delta']:+.4f}), Wilcoxon p={report['wilcoxon_p']:.4g}, "
          f"{report['untied']} untied")
    print(f"  improved {report['improved']}  unchanged {report['unchanged']}  "
          f"worse {report['worse']}")
    print(f"  zero-class cells: {report['zero_class_cells_published']} published, "
          f"{report['zero_class_cells_fullctx']} full history")
    pm = report["parse_matched"]
    print(f"  parse-matched ({pm['n']} cells): {pm['published_mean']} -> "
          f"{pm['fullctx_mean']} (delta {pm['delta']:+.4f}), p={pm['wilcoxon_p']:.4g}")

    if not spec.write:
        print(f"write=false — {spec.out_json} not modified.")
        return
    out = root / spec.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"written: {spec.out_json}")


if __name__ == "__main__":
    main()
