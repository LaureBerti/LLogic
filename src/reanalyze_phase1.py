"""Re-analyze existing phase1_results.csv files with the improved FOL parser.

Reads every phase1_results.csv under outputs/results/, re-parses the fol_block
column with the updated parse_to_z3 + check_satisfiability, and writes an updated
CSV alongside the original (phase1_results_reanalyzed.csv).

Usage:
    python src/reanalyze_phase1.py [--dry-run] [--inplace]

--dry-run   Print what would change, but don't write files.
--inplace   Overwrite phase1_results.csv instead of writing a separate file.
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.parsing.fol_parser import parse_to_z3, check_satisfiability

TIMEOUT_MS = 10_000

def reanalyze_file(path: Path, dry_run: bool = False, inplace: bool = False) -> dict:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {"path": str(path), "rows": 0, "changed": 0}

    changed = 0
    new_rows = []
    for row in rows:
        fol_block = row.get("fol_block", "")
        concept_label = row.get("concept_label", "")

        z3_expr, parse_ok = parse_to_z3(fol_block, concept_label)
        new_solver_result = check_satisfiability(z3_expr, timeout_ms=TIMEOUT_MS)
        new_contradiction = int(new_solver_result == "unsat")
        new_parse_success = int(parse_ok)

        if (row.get("solver_result") != new_solver_result
                or str(row.get("contradiction")) != str(new_contradiction)
                or str(row.get("fol_parse_success")) != str(new_parse_success)):
            changed += 1

        new_row = dict(row)
        new_row["solver_result"] = new_solver_result
        new_row["contradiction"] = new_contradiction
        new_row["fol_parse_success"] = new_parse_success
        new_rows.append(new_row)

    if not dry_run:
        out_path = path if inplace else path.with_name("phase1_results_reanalyzed.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
            writer.writeheader()
            writer.writerows(new_rows)

    solver_counts = Counter(r["solver_result"] for r in new_rows)
    contradiction_rate = sum(r["contradiction"] == 1 or r["contradiction"] == "1"
                             for r in new_rows) / len(new_rows)

    return {
        "path": str(path),
        "rows": len(rows),
        "changed": changed,
        "solver_counts": dict(solver_counts),
        "contradiction_rate": round(contradiction_rate * 100, 2),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    results_root = Path("outputs/results")
    csvs = sorted(results_root.rglob("phase1_results.csv"))

    if not csvs:
        print("No phase1_results.csv files found under outputs/results/")
        return

    print(f"Found {len(csvs)} phase1_results.csv files")
    print(f"Mode: {'dry-run' if args.dry_run else ('inplace' if args.inplace else 'write reanalyzed')}")
    print()

    total_changed = 0
    for path in csvs:
        info = reanalyze_file(path, dry_run=args.dry_run, inplace=args.inplace)
        label = f"{path.parts[-4]}/{path.parts[-3]}/{path.parts[-2]}"
        print(f"  {label:<45}  rows={info['rows']:4d}  changed={info['changed']:4d}  "
              f"contra={info['contradiction_rate']:5.1f}%  "
              f"solver={info['solver_counts']}")
        total_changed += info["changed"]

    print()
    print(f"Total rows changed: {total_changed}")
    if not args.dry_run:
        suffix = "phase1_results.csv" if args.inplace else "phase1_results_reanalyzed.csv"
        print(f"Output written to: {suffix} alongside each input CSV")

if __name__ == "__main__":
    main()
