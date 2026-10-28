from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from math import comb
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

DENSE = "qwen3_8b"
SPARSE = "qwen3_30b-a3b"


def load(root: Path, seed: int, model: str) -> dict:
    out = {}
    for path in root.glob(f"seed{seed}/*/{model}/*/phase1_results.csv"):
        strategy = path.parts[-2]
        with open(path, newline="", errors="replace") as handle:
            for row in csv.DictReader(handle):
                key = (row["concept_iri"], strategy)
                out[key] = row["fol_parse_success"].strip().lower() in ("true", "1")
    return out


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def mcnemar(a: dict, b: dict, keys: list) -> tuple[int, int, float]:
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    n = only_a + only_b
    if not n:
        return only_a, only_b, 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    return only_a, only_b, min(1.0, tail / 2 ** n * 2)


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    spec = cfg.qwen_budget
    seeds = [int(s) for s in spec.seeds]
    arms = {"2048": root / spec.arm_2048, "8192": root / spec.arm_8192}

    report = {"seeds": seeds, "arms": {k: _rel(v, root) for k, v in arms.items()},
              "dense": DENSE, "sparse": SPARSE, "architecture_contrast": {},
              "budget_effect": {}}

    for label, tree in arms.items():
        per_seed, gaps = {}, []
        for seed in seeds:
            d, s = load(tree, seed, DENSE), load(tree, seed, SPARSE)
            keys = sorted(set(d) & set(s))
            if not keys:
                continue
            dr = sum(d[k] for k in keys) / len(keys)
            sr = sum(s[k] for k in keys) / len(keys)
            oa, ob, p = mcnemar(d, s, keys)
            gaps.append(dr - sr)
            per_seed[seed] = {"n": len(keys), "dense": round(dr, 4), "sparse": round(sr, 4),
                              "gap_pp": round((dr - sr) * 100, 2),
                              "dense_only": oa, "sparse_only": ob, "mcnemar_p": p}
        report["architecture_contrast"][label] = {
            "per_seed": per_seed,
            "mean_gap_pp": round(statistics.mean(gaps) * 100, 2) if gaps else None,
            "sd_gap_pp": round(statistics.stdev(gaps) * 100, 2) if len(gaps) > 1 else None,
        }

    lo, hi = sorted(arms)[0], sorted(arms)[-1]
    for model in (DENSE, SPARSE):
        per_seed, effects = {}, []
        for seed in seeds:
            a, b = load(arms[lo], seed, model), load(arms[hi], seed, model)
            keys = sorted(set(a) & set(b))
            if not keys:
                continue
            ar = sum(a[k] for k in keys) / len(keys)
            br = sum(b[k] for k in keys) / len(keys)
            oa, ob, p = mcnemar(a, b, keys)
            effects.append(br - ar)
            per_seed[seed] = {"n": len(keys), lo: round(ar, 4), hi: round(br, 4),
                              "effect_pp": round((br - ar) * 100, 2),
                              "discordant": oa + ob, "mcnemar_p": p}
        report["budget_effect"][model] = {
            "per_seed": per_seed,
            "mean_effect_pp": round(statistics.mean(effects) * 100, 2) if effects else None,
        }

    for label, block in report["architecture_contrast"].items():
        print(f"architecture contrast, {label}: mean gap {block['mean_gap_pp']:+.2f} pp "
              f"(SD {block['sd_gap_pp']})")
        for seed, v in block["per_seed"].items():
            print(f"    seed {seed}: dense {v['dense']*100:.2f}%  sparse {v['sparse']*100:.2f}%  "
                  f"gap {v['gap_pp']:+.2f} pp  p={v['mcnemar_p']:.3g}  n={v['n']}")
    for model, block in report["budget_effect"].items():
        print(f"budget effect, {model}: mean {block['mean_effect_pp']:+.2f} pp")
        for seed, v in block["per_seed"].items():
            print(f"    seed {seed}: {v['effect_pp']:+.2f} pp  p={v['mcnemar_p']:.3g}  "
                  f"discordant={v['discordant']}")

    if not spec.write:
        print(f"write=false — {spec.out_json} not modified.")
        return
    out = root / spec.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written: {spec.out_json}")


if __name__ == "__main__":
    main()
