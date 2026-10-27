from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

from scipy.stats import beta, binomtest

SRC = Path(__file__).resolve().parent.parent.parent
OUT = SRC / "outputs/paper_ready/analysis/phase2b_intervals.json"
STRATEGY = "zero_shot"

csv.field_size_limit(10 ** 9)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> list[float]:
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [round(100 * lo, 1), round(100 * hi, 1)]


def main() -> None:
    cells = {}
    for p in sorted(glob.glob(str(SRC / f"outputs/results/*/*/{STRATEGY}/phase2b_results.csv"))):
        parts = p.split("/")
        onto, model = parts[-4], parts[-3]
        rows = list(csv.DictReader(open(p, errors="replace")))
        if not rows:
            continue
        col = next((c for c in rows[0] if "violat" in c.lower()), None)
        if col is None:
            continue
        usable = [r for r in rows if (r.get("is_unclear") or "").strip() != "1"]
        unclear = len(rows) - len(usable)
        vals = [(r.get(col) or "").strip() for r in usable]
        n = sum(1 for v in vals if v in ("0", "1"))
        k = sum(1 for v in vals if v == "1")
        if not n:
            print(f"  dropped {onto}/{model}: {unclear}/{len(rows)} probes unanswered")
            continue
        lo, hi = clopper_pearson(k, n)
        cells[f"{onto}/{model}"] = {
            "n": n, "violations": k, "rate_pct": round(100 * k / n, 1),
            "unclear_dropped": unclear,
            "ci95_clopper_pearson": [lo, hi],
            "excludes_zero": bool(lo > 0),
            "p_vs_random_half": float(binomtest(k, n, 0.5).pvalue),
            "direction_vs_random": "below" if k / n < 0.5 else "above",
        }

    below = [c for c in cells.values()
             if c["direction_vs_random"] == "below" and c["p_vs_random_half"] < 0.05]
    above = [c for c in cells.values()
             if c["direction_vs_random"] == "above" and c["p_vs_random_half"] < 0.05]
    art = {"strategy": STRATEGY,
           "note": ("Both nulls reported per cell. H0=0 is answered by whether the 95% "
                    "Clopper-Pearson interval excludes zero, not by a p-value; H0=0.5 is a "
                    "two-sided exact binomial with the direction stated."),
           "n_cells": len(cells),
           "n_significantly_below_random": len(below),
           "n_significantly_above_random": len(above),
           "cells": cells}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2))

    print(f"{len(cells)} cells  |  significantly below chance: {len(below)}  "
          f"|  significantly above chance: {len(above)}")
    for key in sorted(cells):
        v = cells[key]
        print("  %-26s %2d/%2d = %5.1f%%  CI [%4.1f, %5.1f]  zero:%-9s chance:%s p=%.3g"
              % (key, v["violations"], v["n"], v["rate_pct"], *v["ci95_clopper_pearson"],
                 "excluded" if v["excludes_zero"] else "included",
                 v["direction_vs_random"], v["p_vs_random_half"]))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
