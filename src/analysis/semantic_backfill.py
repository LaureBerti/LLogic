"""Backfill ``onto_sim_semantic`` for the Phase 3 cells.

Why this exists. LLogic reports structural similarity for Phase 3 but
leaves semantic similarity empty for every row, because ``sentence-transformers``
could not be imported in the run environment. This backfills it, because a structural-only result is
thin: Jaccard over node and edge sets punishes a reconstruction that names the same
concept differently, so it cannot separate "wrong ontology" from "same ontology,
other words".

Nothing here calls an LLM. Phase 3 already wrote each reconstruction to
``reconstruction.txt`` next to its results file, so the reconstructed ontology is
parsed from disk and only the embedding model runs. That keeps this an analysis of
the runs already reported, not a new generation with a different model version.

Cells whose ``reconstruction.txt`` is missing keep ``nan`` and are counted in the
summary rather than dropped, so the denominator stays visible.

Compute: CPU only, local. ``all-MiniLM-L6-v2`` is a 22M-parameter encoder and the
inputs are short concept labels; a GPU would buy nothing.

Run (needs the Python 3.12 environment that has sentence-transformers):
    .venv-sem/bin/python src/analysis/semantic_backfill.py
    .venv-sem/bin/python src/analysis/semantic_backfill.py semantic.dry_run=false
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

from src.metrics.ontology_similarity import (  # noqa: E402
    parse_manchester_to_graph,
    semantic_similarity,
)
from src.ontology_loader import ONTOLOGY_DOMAIN  # noqa: E402

def _gt_concepts(ontology: str, samples_dir: Path, seed: int) -> list[str]:
    """Ground-truth concept labels for an ontology, from the Phase 0 sample."""
    path = samples_dir / f"concepts_{ontology}_seed{seed}.json"
    if not path.exists():
        return []
    try:
        concepts = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [str(c.get("label", "")).strip() for c in concepts if str(c.get("label", "")).strip()]

def _llm_concepts(cell_dir: Path) -> list[str] | None:
    """Concept labels from the stored reconstruction, or None if absent/unparsable."""
    raw = cell_dir / "reconstruction.txt"
    if not raw.exists():
        return None
    graph = parse_manchester_to_graph(raw.read_text(errors="replace"))
    if graph is None:
        return None
    return [str(n) for n in graph.nodes()]

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    sem = cfg.semantic
    samples_dir = root / sem.samples_dir
    model_name = sem.model_name
    dry_run = bool(sem.dry_run)

    result_files = sorted(
        str(p) for pattern in sem.results_dirs
        for p in (root / pattern).glob("**/phase3_results.csv")
    )

    filled = missing_text = unparsable = no_gt = 0
    per_ontology: dict[str, list[float]] = {}

    for results_path in result_files:
        path = Path(results_path)
        seed = sem.seed_by_dir.get(path.parts[-5] if len(path.parts) >= 5 else "", sem.default_seed)
        with open(path, newline="", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        fieldnames = list(rows[0].keys())

        changed = False
        for row in rows:
            current = (row.get("onto_sim_semantic") or "").strip().lower()
            if current and current not in ("nan", "none", ""):
                continue
            ontology = (row.get("ontology") or "").strip()
            gt = _gt_concepts(ontology, samples_dir, seed)
            if not gt:
                no_gt += 1
                continue
            llm = _llm_concepts(path.parent)
            if llm is None:
                missing_text += 1
                continue
            if not llm:
                unparsable += 1
                continue
            score = semantic_similarity(gt, llm, model_name=model_name)
            if score != score:            # nan
                unparsable += 1
                continue
            row["onto_sim_semantic"] = score
            per_ontology.setdefault(ontology, []).append(score)
            filled += 1
            changed = True

        if changed and not dry_run:
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    print(f"phase3 result files: {len(result_files)}")
    print(f"filled:      {filled}")
    print(f"no reconstruction.txt: {missing_text}")
    print(f"unparsable/empty graph: {unparsable}")
    print(f"no ground-truth sample: {no_gt}")
    if per_ontology:
        print(f"\n{'ontology':12s} n   mean   min   max   domain")
        for ontology in sorted(per_ontology):
            values = per_ontology[ontology]
            print(f"{ontology:12s} {len(values):3d} "
                  f"{sum(values)/len(values):6.3f} {min(values):5.3f} {max(values):5.3f}   "
                  f"{ONTOLOGY_DOMAIN.get(ontology, '?')}")
    if dry_run:
        print("\nDRY RUN — no CSV written. Re-run with semantic.dry_run=false to persist.")

if __name__ == "__main__":
    main()
