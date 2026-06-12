#!/usr/bin/env python3
"""
analyze_results.py — aggregate the per-condition CSVs in outputs/results/ into the
headline numbers reported in the paper (C1-C4).

Run after the pipeline has produced results:
    python3 scripts/analyze_results.py
    python3 scripts/analyze_results.py --base outputs/results

Reports, for whatever cells are present:
  C1  contradiction rate + FOL parse-success rate (per model)
  C2  Kruskal-Wallis across models (format compliance)
  C3  subClassOf transitivity violation rate (Phase 2b) per ontology
  C4  Phase-3 concept recall + general-vs-technical domain test
"""
from __future__ import annotations
import argparse
import csv
import statistics
from pathlib import Path

try:
    from scipy.stats import kruskal, mannwhitneyu
    _SCIPY = True
except ImportError:
    _SCIPY = False

MODELS = ["llama3.2_latest", "gemma2_2b", "mistral_7b", "qwen2.5_7b"]
STRATS = ["zero_shot", "cot", "tot", "self_consistency"]
GENERAL = {"book", "agrovoc"}
TECHNICAL = {"anatomy", "cso", "mesh"}
ALL_ONTOS = sorted(GENERAL | TECHNICAL)


def _uniq_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    seen, out = set(), []
    for r in csv.DictReader(open(path)):
        k = r.get("concept_iri", "")
        if k and k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _is_true(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1")


def phase1(base: Path) -> None:
    print("\n" + "=" * 60)
    print(" C1 — Contradiction rate & format compliance (Phase 1)")
    print("=" * 60)
    total_eval = total_comp = total_contra = 0
    model_psr: dict[str, list[float]] = {m: [] for m in MODELS}
    for onto in ALL_ONTOS:
        for m in MODELS:
            for s in STRATS:
                rows = _uniq_rows(base / onto / m / s / "phase1_results.csv")
                if not rows:
                    continue
                n = len(rows)
                comp = sum(1 for r in rows if _is_true(r.get("fol_parse_success", "")))
                contra = sum(1 for r in rows if _is_true(r.get("contradiction", "")))
                total_eval += n
                total_comp += comp
                total_contra += contra
                if comp:
                    model_psr[m].append(comp / n)
    if total_eval == 0:
        print("  (no Phase 1 results found)")
        return
    cr = total_contra / total_eval * 100
    print(f"  Concepts evaluated : {total_eval}")
    print(f"  Compiled to Z3     : {total_comp}")
    print(f"  Contradictions     : {total_contra}  ->  CR = {cr:.3f}%")
    print(f"  Format compliance (mean PSR per model):")
    psr_lists = []
    for m in MODELS:
        if model_psr[m]:
            mean = statistics.mean(model_psr[m]) * 100
            psr_lists.append(model_psr[m])
            print(f"    {m:18s}  {mean:5.1f}%  (n={len(model_psr[m])} cells)")
    if _SCIPY and len([p for p in psr_lists if p]) >= 2:
        H, p = kruskal(*[p for p in psr_lists if p])
        print(f"  C2 Kruskal-Wallis across models: H={H:.2f}, p={p:.5f}")
    elif not _SCIPY:
        print("  (install scipy for the Kruskal-Wallis test)")


def phase2b(base: Path) -> None:
    print("\n" + "=" * 60)
    print(" C3 — subClassOf transitivity violation rate (Phase 2b)")
    print("=" * 60)
    found = False
    for onto in ALL_ONTOS:
        rates = []
        for m in MODELS:
            p = base / onto / m / "zero_shot" / "phase2b_results.csv"
            if not p.exists():
                continue
            rows = list(csv.DictReader(open(p)))
            if not rows:
                continue
            v = sum(1 for r in rows if r.get("is_violation", "") == "1")
            rates.append(v / len(rows) * 100)
        if rates:
            found = True
            tag = "T" if onto in TECHNICAL else "G"
            print(f"  {onto:8s} ({tag})  mean TVR = {statistics.mean(rates):5.1f}%  "
                  f"(cells: {', '.join(f'{r:.0f}%' for r in rates)})")
    if not found:
        print("  (no Phase 2b results found)")


def phase3(base: Path) -> None:
    print("\n" + "=" * 60)
    print(" C4 — Concept recall & domain moderation (Phase 3)")
    print("=" * 60)

    def cc(onto: str, m: str, s: str):
        p = base / onto / m / s / "phase3_results.csv"
        if not p.exists():
            return None
        rows = list(csv.DictReader(open(p)))
        if not rows:
            return None
        try:
            return float(rows[0].get("concept_coverage", ""))
        except ValueError:
            return None

    gen = [cc(o, m, s) for o in GENERAL for m in MODELS for s in ("zero_shot", "cot")]
    tec = [cc(o, m, s) for o in TECHNICAL for m in MODELS for s in ("zero_shot", "cot")]
    gen = [v for v in gen if v is not None]
    tec = [v for v in tec if v is not None]
    if not (gen or tec):
        print("  (no Phase 3 results found)")
        return
    allv = gen + tec
    print(f"  Cells: {len(allv)}   mean concept_coverage = {statistics.mean(allv):.3f}  "
          f"(range {min(allv):.2f}-{max(allv):.2f})")
    if gen and tec:
        gm, tm = statistics.mean(gen), statistics.mean(tec)
        print(f"  General  (n={len(gen)}): {gm:.3f}   Technical (n={len(tec)}): {tm:.3f}   "
              f"ratio {gm / tm:.2f}x" if tm else "")
        if _SCIPY:
            U, p = mannwhitneyu(gen, tec, alternative="two-sided")
            print(f"  Mann-Whitney U={U:.0f}, p={p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/results", help="results directory")
    args = ap.parse_args()
    base = Path(args.base)
    print(f"Analyzing results under: {base.resolve()}")
    if not base.exists():
        print("  Directory not found — run the pipeline first (see README.md).")
        return
    phase1(base)
    phase2b(base)
    phase3(base)
    print()


if __name__ == "__main__":
    main()
