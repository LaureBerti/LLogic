"""Recompute every C1 number from a single stated grid, and write it to an artifact.

Why this exists. The C1 figures in the manuscript (2,579 valid / 1,165 parse failures /
3,744 evaluated) matched no file in `outputs/analysis/`. They were computed
once and hardcoded, and the live results tree could not reproduce them: 82.9% of the
submitted-grid rows there still store replies truncated at 300 characters, so the extractor
cannot be re-applied to them.

The run that produced them survives as `outputs/archive/collapse_vm_20260906/`, recovered
from a VM before it was deleted. It reproduces the denominator exactly (3,744) and the rate
to 0.1 pp, and it stores full replies for the rows the live tree truncates. This script
recomputes C1 from it so that every number in the paper traces to a file.

The grid is stated explicitly rather than assumed, because the manuscript's two number
families used different denominators (3,744 over the submitted grid; 4,489 over the whole
corpus) without saying which was which.

Compute: CPU only, local, seconds. No LLM calls, no network.

Run:
    python src/analysis/c1_recompute.py
    python src/analysis/c1_recompute.py c1.source=outputs/results
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

# The grid an earlier version reports on: the four original models over the five
# ontologies that carry usable concept labels. Stated here so the denominator is a
# decision, not an accident of which directories happened to exist.
GRID_ONTOLOGIES = ("book", "anatomy", "agrovoc", "cso", "mesh")
GRID_MODELS = ("gemma2_2b", "llama3.2_latest", "mistral_7b", "qwen2.5_7b")

def clopper_pearson_upper(k: int, n: int, conf: float, one_sided: bool) -> float:
    from scipy.stats import beta
    if n == 0:
        return float("nan")
    alpha = (1.0 - conf) if one_sided else (1.0 - conf) / 2
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))

def collect(source: Path) -> tuple[list[dict], int]:
    """Rows of the stated grid, plus the count dropped as CSV-garbled.

    A garbled row has a strategy field carrying a fragment of another field, from
    unescaped newlines in an early run. They are excluded from every analysis and the
    count is reported rather than passed over (an earlier defect).
    """
    rows, garbled = [], 0
    for path in sorted(glob.glob(str(source / "*/*/*/phase1_results.csv"))):
        parts = path.split("/")
        onto, model, strategy = parts[-4], parts[-3], parts[-2]
        if onto not in GRID_ONTOLOGIES or model not in GRID_MODELS:
            continue
        for row in csv.DictReader(open(path, errors="replace")):
            if (row.get("strategy") or "").strip() != strategy:
                garbled += 1
                continue
            rows.append(row)
    return rows, garbled

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    source = root / cfg.get("c1", {}).get(
        "source", "outputs/archive/collapse_vm_20260906/results")
    conf = float(cfg.stats.conf_level)

    rows, garbled = collect(source)
    n = len(rows)
    compiled = sum(1 for r in rows if str(r.get("fol_parse_success", "")).strip() == "1")
    failures = n - compiled
    contradictions = sum(1 for r in rows
                         if str(r.get("contradiction", "")).strip() in ("1", "true", "yes"))
    truncated = sum(1 for r in rows if 0 < len((r.get("response_text") or "")) <= 300)
    empty = sum(1 for r in rows if not (r.get("response_text") or "").strip())

    out = {
        "source": str(source.relative_to(root)),
        "grid": {"ontologies": list(GRID_ONTOLOGIES), "models": list(GRID_MODELS),
                 "strategies": 4, "cells_expected": 80},
        "n_evaluated": n,
        "n_csv_garbled_excluded": garbled,
        "n_compiled": compiled,
        "n_parse_failures": failures,
        "parse_failure_rate": round(failures / n, 4) if n else None,
        "n_contradictions": contradictions,
        "contradiction_rate_over_compiled": round(contradictions / compiled, 6) if compiled else None,
        "bound_over_compiled": {
            "k": contradictions, "n": compiled,
            "upper_one_sided_pct": round(100 * clopper_pearson_upper(contradictions, compiled, conf, True), 4),
            "upper_two_sided_pct": round(100 * clopper_pearson_upper(contradictions, compiled, conf, False), 4),
        },
        "bound_all_failures_contradictory": {
            "k": failures, "n": n,
            "upper_one_sided_pct": round(100 * clopper_pearson_upper(failures, n, conf, True), 4),
        },
        "response_text_truncated_at_300": truncated,
        "response_text_empty": empty,
        "verdicts": dict(Counter(r.get("solver_result", "") for r in rows)),
    }

    out_path = root / cfg.paths.out_dir / "c1_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"source: {out['source']}")
    print(f"grid  : {len(GRID_MODELS)} models x {len(GRID_ONTOLOGIES)} ontologies x 4 strategies")
    print(f"\nevaluated responses      : {n}")
    print(f"  excluded as CSV-garbled: {garbled}")
    print(f"  compiled to checkable   : {compiled}")
    print(f"  parse failures          : {failures}  ({100*failures/n:.1f}%)")
    print(f"  contradictions          : {contradictions}")
    print(f"\n95% upper bound over compiled responses:")
    print(f"  one-sided {out['bound_over_compiled']['upper_one_sided_pct']:.4f}%"
          f"   two-sided {out['bound_over_compiled']['upper_two_sided_pct']:.4f}%")
    print(f"worst case (all failures contradictory): "
          f"{out['bound_all_failures_contradictory']['upper_one_sided_pct']:.1f}%")
    print(f"\nstored replies truncated at 300 chars: {truncated} "
          f"({100*truncated/n:.1f}%)   empty: {empty}")
    print(f"\nwritten: {out_path.relative_to(root)}")

if __name__ == "__main__":
    main()
