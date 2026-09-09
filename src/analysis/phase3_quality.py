"""Phase 3 reconstruction quality: precision, recall, F1 and invented concepts, per model.

Precision, F1 and hallucinated-concept counts are reported alongside recall, since recall alone
rewards a model that simply emits more concepts.

This exists because the earlier summary averaged the three statistics over DIFFERENT
denominators. `concept_f1` is written blank in the per-cell file whenever precision and recall
are both zero, so averaging the column silently dropped the cells that failed completely --
between 2 and 5 of the 10 cells per model -- while precision and recall kept them. The result
was a mean F1 ABOVE the harmonic mean of the reported precision and recall, which is
arithmetically impossible and is exactly the kind of value a careful reader recomputes.

Here every statistic is a macro-average over all 10 cells of a model, with F1 taken as 0 when
precision and recall are both 0 (the standard convention for a degenerate cell). The invariant
F1 <= 2PR/(P+R) is asserted, so this cannot regress silently.

Compute: CPU only, local, seconds. No network.

Run:
    python src/analysis/phase3_quality.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path(".")
CELLS = SRC / "outputs/analysis/phase3_precision_recall.csv"
OUT = SRC / "outputs/analysis/phase3_quality.json"


def f(row: dict, key: str) -> float:
    return float((row.get(key) or "").strip() or 0.0)


def main() -> None:
    by = defaultdict(list)
    for r in csv.DictReader(open(CELLS, errors="replace")):
        by[r["llm_model"]].append(r)

    per_model, overall = {}, defaultdict(float)
    for model, rows in sorted(by.items()):
        n = len(rows)
        P = [f(r, "concept_precision") for r in rows]
        R = [f(r, "concept_recall") for r in rows]
        # F1 = 0 for a cell that produced nothing usable; dropping it inflates the mean.
        F = [0.0 if p + q == 0 else 2 * p * q / (p + q) for p, q in zip(P, R)]
        H = [f(r, "n_hallucinated_concepts") for r in rows]
        mp, mr, mf, mh = (sum(x) / n for x in (P, R, F, H))
        cap = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
        assert mf <= cap + 1e-9, (
            f"{model}: mean F1 {mf:.4f} exceeds 2PR/(P+R) = {cap:.4f}; "
            "the three statistics are not on the same denominator")
        # SD across the model's cells: the cells are the replicates, so this is the
        # dispersion a reader needs to judge whether the means separate at all.
        sd = lambda v: round(statistics.stdev(v), 3) if len(v) > 1 else 0.0
        per_model[model] = {
            "cells": n,
            "cells_with_degenerate_f1": sum(1 for p, q in zip(P, R) if p + q == 0),
            "precision": round(mp, 3), "recall": round(mr, 3), "f1": round(mf, 3),
            "precision_sd": sd(P), "recall_sd": sd(R), "f1_sd": sd(F),
            "max_possible_f1_from_means": round(cap, 3),
            "invented_concepts_per_cell": round(mh, 1),
            "invented_sd": round(statistics.stdev(H), 1) if len(H) > 1 else 0.0,
        }
        for k, v in (("precision", mp), ("recall", mr), ("f1", mf),
                     ("invented_concepts_per_cell", mh)):
            overall[k] += v / len(by)

    allP, allR, allF, allH = [], [], [], []
    for rows in by.values():
        P = [f(r, "concept_precision") for r in rows]
        R = [f(r, "concept_recall") for r in rows]
        allP += P
        allR += R
        allF += [0.0 if p + q == 0 else 2 * p * q / (p + q) for p, q in zip(P, R)]
        allH += [f(r, "n_hallucinated_concepts") for r in rows]
    overall_sd = {"precision_sd": round(statistics.stdev(allP), 3),
                  "recall_sd": round(statistics.stdev(allR), 3),
                  "f1_sd": round(statistics.stdev(allF), 3),
                  "invented_sd": round(statistics.stdev(allH), 1),
                  "cells": len(allP)}

    art = {"note": ("Macro-average over all cells per model. F1 is 0 for a cell with zero "
                    "precision and recall; the earlier summary dropped those cells from F1 "
                    "only, producing an impossible mean."),
           "by_model": per_model,
           "all_models": {k: round(v, 3) for k, v in overall.items()},
           "all_models_sd": overall_sd}
    OUT.write_text(json.dumps(art, indent=2))

    print("%-18s %5s %5s %5s %6s  %s" % ("model", "P", "R", "F1", "inv/cell", "degenerate cells"))
    for m, v in per_model.items():
        print("  %-16s %.3f %.3f %.3f %6.1f  %d of %d"
              % (m, v["precision"], v["recall"], v["f1"],
                 v["invented_concepts_per_cell"], v["cells_with_degenerate_f1"], v["cells"]))
    print("  %-16s %.3f %.3f %.3f %6.1f" % ("ALL", *(art["all_models"][k] for k in
          ("precision", "recall", "f1", "invented_concepts_per_cell"))))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
