"""Generalisation experiment: current-generation models, and MoE versus dense.

Model generations turn over faster than evaluation protocols, so the study adds newer
checkpoints and varies architecture and training regime. This computes both from the result tree and writes a durable artifact, so
the figures in the paper and the letter trace to a file rather than to a run someone remembers.

Two things worth stating, because they bound what the numbers support:

  The newer models are a SEPARATE grid, not an extension of the core one, and are never pooled
  with the core four. In the seed-42 tree they ran on a different ontology slate, so those rates
  are not comparable cell-for-cell; in the seed-123 tree they ran on the SAME slate as the core
  four, which is where the matched generation contrast is computed.

  The MoE-vs-dense contrast is matched, and only on the ontologies the two Qwen 3 variants
  actually share. Comparing them across their full coverage would confound architecture with
  ontology, since their slates differ.

`fol_parse_success` is the compile gate: a response that yields a formula the translator can
render. It is NOT "the definition is correct" -- see the reasoner-validation experiment for
what a pass does and does not certify.

Compute: CPU only, local, seconds. No network.

Run:
    python src/analysis/r1_newer_models.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

SRC = Path(".")
OUT = SRC / "outputs/analysis/r1_newer_models.json"

CORE = ("gemma2_2b", "llama3.2_latest", "mistral_7b", "qwen2.5_7b")
NEW = ("gemma4_latest", "qwen3_8b", "qwen3_30b-a3b")
DENSE, MOE = "qwen3_8b", "qwen3_30b-a3b"

csv.field_size_limit(10 ** 9)


def load(tree: str) -> dict:
    """(model, ontology, strategy, concept_iri) -> row, first occurrence wins (resume key)."""
    out = {}
    for p in glob.glob(str(SRC / tree / "*/*/*/phase1_results.csv")):
        parts = p.split("/")
        onto, model, strat = parts[-4], parts[-3], parts[-2]
        for r in csv.DictReader(open(p, errors="replace")):
            if (r.get("strategy") or "").strip() != strat:
                continue                      # garbled row carrying another field
            out.setdefault((model, onto, strat, r.get("concept_iri", "")), r)
    return out


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [float("nan")] * 2
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs."""
    n = b + c
    if not n:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def rate(rows) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if (r.get("fol_parse_success") or "").strip() == "1")
    e = sum(1 for r in rows if not (r.get("response_text") or "").strip())
    return {"n": n, "compiled": k,
            "compile_rate_pct": round(100 * k / n, 1) if n else None,
            "ci95": wilson(k, n), "empty": e,
            "empty_pct": round(100 * e / n, 1) if n else None}


def main() -> None:
    trees = {"seed42": load("outputs/results"), "seed123": load("outputs/results_seed123")}

    per_model = {}
    for tname, data in trees.items():
        by = defaultdict(list)
        for (model, onto, _s, _c), r in data.items():
            by[(model, onto)].append(r)
        for model in CORE + NEW:
            ontos = sorted({o for (m, o) in by if m == model})
            if not ontos:
                continue
            rows = [r for (m, o), rs in by.items() if m == model for r in rs]
            per_model[f"{tname}/{model}"] = {
                "ontologies": ontos, "family": "new" if model in NEW else "core", **rate(rows)}

    # MoE vs dense: matched on the concepts both variants answered, on shared ontologies only.
    arch = {}
    for tname, data in trees.items():
        shared = ({o for (m, o, _s, _c) in data if m == DENSE}
                  & {o for (m, o, _s, _c) in data if m == MOE})
        keys = [k for k in data
                if k[0] == DENSE and k[1] in shared
                and (MOE, k[1], k[2], k[3]) in data]
        if not keys:
            continue
        b = c = 0                       # b: dense only, c: MoE only
        dn = mn = de = me = 0
        for k in keys:
            d, m = data[k], data[(MOE, k[1], k[2], k[3])]
            dok = (d.get("fol_parse_success") or "").strip() == "1"
            mok = (m.get("fol_parse_success") or "").strip() == "1"
            dn += dok
            mn += mok
            de += not (d.get("response_text") or "").strip()
            me += not (m.get("response_text") or "").strip()
            b += dok and not mok
            c += mok and not dok
        n = len(keys)
        # Excluding the MoE's empty replies isolates "failed to finish" from "answered badly".
        nz = [k for k in keys if (data[(MOE, k[1], k[2], k[3])].get("response_text") or "").strip()]
        mnz = sum(1 for k in nz
                  if (data[(MOE, k[1], k[2], k[3])].get("fol_parse_success") or "").strip() == "1")
        arch[tname] = {
            "shared_ontologies": sorted(shared), "matched_concepts": n,
            "dense_model": DENSE, "moe_model": MOE,
            "dense_compile_pct": round(100 * dn / n, 1), "dense_ci95": wilson(dn, n),
            "moe_compile_pct": round(100 * mn / n, 1), "moe_ci95": wilson(mn, n),
            "discordant_dense_only": b, "discordant_moe_only": c,
            "mcnemar_p": mcnemar_exact(b, c),
            "dense_empty_pct": round(100 * de / n, 1),
            "moe_empty_pct": round(100 * me / n, 1),
            "moe_compile_pct_excl_empty": round(100 * mnz / len(nz), 1) if nz else None,
            "moe_n_excl_empty": len(nz),
        }

    # Newer vs core, matched. In the seed-123 tree Gemma 4 and Qwen 3-8B were run on the same
    # ontology slate as the four core models, so the generation contrast can be made on the
    # concepts every model answered rather than across differing slates.
    gen = {}
    data = trees["seed123"]
    newer = [m for m in ("gemma4_latest", DENSE)
             if any(k[0] == m for k in data)]
    for nm in newer:
        for cm in CORE:
            keys = [k for k in data if k[0] == nm and (cm, k[1], k[2], k[3]) in data]
            if not keys:
                continue
            b = c = nk = ck = 0
            for k in keys:
                nok = (data[k].get("fol_parse_success") or "").strip() == "1"
                cok = (data[(cm, k[1], k[2], k[3])].get("fol_parse_success") or "").strip() == "1"
                nk += nok
                ck += cok
                b += nok and not cok
                c += cok and not nok
            n = len(keys)
            gen[f"{nm} vs {cm}"] = {
                "matched_concepts": n,
                "newer_pct": round(100 * nk / n, 1), "core_pct": round(100 * ck / n, 1),
                "delta_pp": round(100 * (nk - ck) / n, 1),
                "discordant_newer_only": b, "discordant_core_only": c,
                "mcnemar_p": mcnemar_exact(b, c),
            }

    art = {"note": ("Generalisation experiment. The newer models form a separate "
                    "grid and are never pooled with the four core models; the MoE/dense contrast "
                    "is matched on shared ontologies only."),
           "compile_gate": "fol_parse_success == 1",
           "per_model": per_model, "architecture_moe_vs_dense": arch,
           "newer_vs_core_matched_seed123": gen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2))

    print("per-model compile rates")
    for k in sorted(per_model, key=lambda k: (per_model[k]["family"], k)):
        v = per_model[k]
        print("  %-26s %-4s n=%4d  %5.1f%% [%.1f, %.1f]  empty %4.1f%%  %s"
              % (k, v["family"], v["n"], v["compile_rate_pct"], *v["ci95"],
                 v["empty_pct"], ",".join(o[:4] for o in v["ontologies"])))
    print("\nMoE vs dense (matched)")
    for t, v in arch.items():
        print("  %s: n=%d on %s" % (t, v["matched_concepts"], ", ".join(v["shared_ontologies"])))
        print("     dense %.1f%% %s vs MoE %.1f%% %s | discordant %d/%d | exact McNemar p=%.3g"
              % (v["dense_compile_pct"], v["dense_ci95"], v["moe_compile_pct"], v["moe_ci95"],
                 v["discordant_dense_only"], v["discordant_moe_only"], v["mcnemar_p"]))
        print("     MoE empty %.1f%% (dense %.1f%%); MoE excl. empty %.1f%% on n=%d"
              % (v["moe_empty_pct"], v["dense_empty_pct"],
                 v["moe_compile_pct_excl_empty"], v["moe_n_excl_empty"]))
    print("\nnewer vs core, matched on the seed-123 slate")
    for k, v in art["newer_vs_core_matched_seed123"].items():
        print("  %-34s n=%3d  newer %5.1f%% vs core %5.1f%%  (%+.1f pp)  McNemar p=%.3g"
              % (k, v["matched_concepts"], v["newer_pct"], v["core_pct"],
                 v["delta_pp"], v["mcnemar_p"]))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
