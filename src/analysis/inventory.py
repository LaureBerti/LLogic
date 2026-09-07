"""Report what experimental results exist, computed from the result files.

A long experimental campaign spreads results across many directories and seeds, and the
state of it cannot reliably be held in a summary or in memory. This reports, from the files
themselves:

  - every cell that exists -- (ontology x model x strategy) -- with row counts and
    completeness against that cell's own expected size. Expected sizes differ per ontology
    (book samples 36 concepts, anatomy 48, the others 50), so assuming a single number
    makes complete cells look short and short ones look complete;
  - empty or failed rows, duplicate keys, and the distribution of each recorded verdict;
  - which analysis artifacts exist and what they contain.

Run it before drawing any conclusion about coverage, and before launching a run to fill a
gap: the gap may not be where it is assumed to be.

Compute: CPU only, local, seconds. No LLM calls, no network.

Run:
    python src/analysis/inventory.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

def sample_sizes(root: Path) -> dict[tuple[str, str], int]:
    """(ontology, seed) -> number of concepts actually sampled.

    A cell is not "short" because it has fewer than 50 rows; it is short relative to
    its own sample. Treating 50 as universal is what made 36-row book cells and
    48-row anatomy cells look incomplete.
    """
    sizes: dict[tuple[str, str], int] = {}
    for path in glob.glob(str(root / "data/samples/concepts_*_seed*.json")):
        base = os.path.basename(path)
        if "INVALID" in base:
            continue
        parts = base.replace(".json", "").split("_")
        try:
            sizes[(parts[1], parts[2].replace("seed", ""))] = len(json.load(open(path)))
        except (OSError, ValueError, IndexError):
            continue
    return sizes

def seed_of(results_dir: str) -> str:
    name = os.path.basename(results_dir.rstrip("/"))
    return name.replace("results_seed", "") if name.startswith("results_seed") else "42"

def scan_phase1(root: Path, sizes: dict[tuple[str, str], int]) -> list[dict]:
    rows = []
    for results_dir in sorted(glob.glob(str(root / "outputs/results*"))):
        if not os.path.isdir(results_dir):
            continue
        seed = seed_of(results_dir)
        for path in sorted(glob.glob(results_dir + "/*/*/*/phase1_results.csv")):
            parts = path.split("/")
            onto, model, strategy = parts[-4], parts[-3], parts[-2]
            data = list(csv.DictReader(open(path, errors="replace")))
            empty = sum(1 for r in data if not (r.get("response_text") or "").strip())
            iris = Counter(r.get("concept_iri", "") for r in data)
            dupes = sum(v - 1 for v in iris.values() if v > 1)
            verdicts = Counter((r.get("solver_result") or "none") for r in data)
            accepted = sum(1 for r in data
                           if str(r.get("contradiction", "")).strip() not in ("1", "true", "yes"))
            expected = sizes.get((onto, seed))
            rows.append({
                "seed": seed, "ontology": onto, "model": model, "strategy": strategy,
                "rows": len(data), "expected": expected,
                "complete": expected is not None and len(data) >= expected,
                "empty": empty, "duplicate_rows": dupes,
                "filter_accepted": accepted, "solver": dict(verdicts),
            })
    return rows

def scan_phase2b(root: Path) -> list[dict]:
    rows = []
    for results_dir in sorted(glob.glob(str(root / "outputs/results*"))):
        if not os.path.isdir(results_dir):
            continue
        for path in sorted(glob.glob(results_dir + "/*/*/*/phase2b_results.csv")):
            parts = path.split("/")
            data = list(csv.DictReader(open(path, errors="replace")))
            rows.append({"seed": seed_of(results_dir), "ontology": parts[-4],
                         "model": parts[-3], "strategy": parts[-2], "rows": len(data)})
    return rows

def scan_arms(root: Path) -> dict:
    out = {}
    for path in sorted(glob.glob(str(root / "outputs/analysis/reasoner_arm_*.json"))):
        try:
            out[os.path.basename(path)] = json.load(open(path))
        except (OSError, ValueError):
            out[os.path.basename(path)] = "unreadable"
    return out

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    sizes = sample_sizes(root)

    p1 = scan_phase1(root, sizes)
    print("=" * 78)
    print("PHASE 1 — cells on disk")
    print("=" * 78)
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for r in p1:
        by_seed[r["seed"]].append(r)
    for seed in sorted(by_seed, key=lambda s: int(s)):
        cells = by_seed[seed]
        done = sum(1 for c in cells if c["complete"])
        print(f"\n  seed {seed}: {len(cells)} cells, {done} complete, "
              f"{sum(c['rows'] for c in cells)} rows, "
              f"{sum(c['empty'] for c in cells)} empty, "
              f"{sum(c['duplicate_rows'] for c in cells)} duplicate rows")
        ontos = sorted({c["ontology"] for c in cells})
        for onto in ontos:
            sub = [c for c in cells if c["ontology"] == onto]
            models = sorted({c["model"] for c in sub})
            short = [c for c in sub if not c["complete"]]
            flag = "" if not short else f"   SHORT: {len(short)}"
            print(f"     {onto:<11s} {len(sub):3d} cells  models={len(models)}{flag}")
            for c in short:
                print(f"        {c['rows']:3d}/{c['expected']}  {c['model']}/{c['strategy']}")

    p2 = scan_phase2b(root)
    print("\n" + "=" * 78)
    print("PHASE 2B — cells on disk")
    print("=" * 78)
    if not p2:
        print("  none")
    else:
        per = Counter((r["seed"], r["ontology"]) for r in p2)
        for (seed, onto), n in sorted(per.items()):
            rows = sum(r["rows"] for r in p2 if r["seed"] == seed and r["ontology"] == onto)
            print(f"  seed {seed:<5s} {onto:<11s} {n:3d} cells  {rows:5d} rows")
        covered = sorted({r["ontology"] for r in p2})
        print(f"\n  ontologies covered: {covered}")

    print("\n" + "=" * 78)
    print("the reasoner validation ARMS — results files present")
    print("=" * 78)
    arms = scan_arms(root)
    if not arms:
        print("  none")
    for name, payload in arms.items():
        if isinstance(payload, dict) and "results" in payload:
            scored = sum(r.get("n_scored", 0) for r in payload["results"])
            unsat = sum(r.get("unsatisfiable", 0) for r in payload["results"])
            onts = [r.get("ontology") for r in payload["results"]]
            print(f"  {name:<24s} ontologies={onts} scored={scored} unsatisfiable={unsat}")
        elif isinstance(payload, dict):
            keys = {k: payload[k] for k in list(payload)[:6]}
            print(f"  {name:<24s} {keys}")
        else:
            print(f"  {name:<24s} {payload}")

    out_path = root / cfg.paths.out_dir / "experiment_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"phase1": p1, "phase2b": p2, "reasoner_arms": list(arms)}, indent=2, default=str))
    print(f"\nwritten: {out_path.relative_to(root)}")

if __name__ == "__main__":
    main()
