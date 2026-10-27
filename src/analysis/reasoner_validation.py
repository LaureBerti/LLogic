from __future__ import annotations

import csv
import glob
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

sys.setrecursionlimit(100000)

from src.analysis.fol_to_owl import classify_definition


def clopper_pearson(successes: int, trials: int, conf: float = 0.95) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta
    except ImportError:
        return (float("nan"), float("nan"))
    alpha = 1.0 - conf
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (float(lower), float(upper))


def load_filter_accepted(cfg: DictConfig, root: Path) -> list[dict]:
    rows: list[dict] = []
    for pattern in cfg.reasoner.results_dirs:
        for path in sorted((root / pattern).glob("**/phase1_results.csv")):
            for row in csv.DictReader(open(path, newline="", errors="replace")):
                contradiction = str(row.get("contradiction", "")).strip().lower()
                if contradiction in ("1", "true", "yes"):
                    continue
                classes, reason = classify_definition(row.get("fol_block") or "")
                rows.append({
                    "ontology": (row.get("ontology") or "").strip(),
                    "model": (row.get("llm_model") or "").strip(),
                    "strategy": (row.get("strategy") or "").strip(),
                    "concept_label": (row.get("concept_label") or "").strip(),
                    "concept_iri": (row.get("concept_iri") or "").strip(),
                    "classes": classes,
                    "reason": reason,
                })
    return rows


def _reason_over(ontology_path: Path, definitions: list[dict],
                 timeout_s: int) -> dict[tuple[str, str, str], str]:
    import owlready2

    verdicts: dict[tuple[str, str, str], str] = {}
    errors: list[tuple[str, str]] = []
    for definition in definitions:
        workdir = Path(tempfile.mkdtemp(prefix="r4_10f_"))
        try:
            local = workdir / ontology_path.name
            shutil.copy(ontology_path, local)
            world = owlready2.World()
            onto = world.get_ontology(local.as_uri()).load()

            with onto:
                target = onto.search_one(label=definition["concept_label"])
                if not (isinstance(target, owlready2.ThingClass)
                        and not isinstance(target, owlready2.PropertyClass)):
                    target = owlready2.types.new_class(
                        _safe_name(definition["concept_label"]), (owlready2.Thing,))
                conjuncts = [_get_or_make_class(onto, name) for name in definition["classes"]]
                expression = conjuncts[0]
                for extra in conjuncts[1:]:
                    expression = expression & extra
                target.equivalent_to.append(expression)

            try:
                with onto:
                    owlready2.sync_reasoner(world, debug=0)
                unsat = _is_unsatisfiable(world, target)
                verdicts[_key(definition)] = "unsatisfiable" if unsat else "satisfiable"
            except owlready2.OwlReadyInconsistentOntologyError:
                verdicts[_key(definition)] = "inconsistent"
            except Exception as exc:
                verdicts[_key(definition)] = f"error:{type(exc).__name__}"
                errors.append((definition["concept_label"], f"{type(exc).__name__}: {exc}"[:160]))
        except Exception as exc:
            verdicts[_key(definition)] = f"setup_error:{type(exc).__name__}"
            errors.append((definition["concept_label"], f"setup {type(exc).__name__}: {exc}"[:160]))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    if errors:
        print(f"\n  first errors ({len(errors)} total):")
        for label, message in errors[:4]:
            print(f"    {label[:32]:34s} {message}")
    return verdicts


def _is_unsatisfiable(world, target) -> bool:
    import owlready2
    try:
        if owlready2.Nothing in list(getattr(target, "equivalent_to", [])):
            return True
    except Exception:
        pass
    try:
        for cls in world.inconsistent_classes():
            if cls is target:
                return True
    except TypeError:
        pass
    return False


def _key(definition: dict) -> tuple[str, str, str]:
    return (definition.get("model", ""), definition.get("strategy", ""),
            definition.get("concept_label", ""))


def _safe_name(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label) or "Concept"


def _get_or_make_class(onto, name: str):
    import owlready2
    existing = onto.search_one(iri=f"*{name}")
    if (existing is not None
            and isinstance(existing, owlready2.ThingClass)
            and not isinstance(existing, owlready2.PropertyClass)):
        return existing
    return owlready2.types.new_class(_safe_name(name), (owlready2.Thing,))


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    arm = cfg.reasoner.arm
    out_dir = root / cfg.paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_filter_accepted(cfg, root)
    translatable = [r for r in rows if r["classes"]]
    reasons = Counter(r["reason"] for r in rows)

    print(f"arm: {arm}")
    print(f"filter-accepted definitions : {len(rows)}")
    print(f"translatable to OWL         : {len(translatable)} "
          f"({100 * len(translatable) / max(len(rows), 1):.1f}%)")
    print("\nnot translatable, by cause (published, not hidden):")
    for reason, count in reasons.most_common():
        if reason == "ok":
            continue
        print(f"  {reason:26s} {count:5d}  {100 * count / max(len(rows),1):5.1f}%")

    allowed = set(cfg.reasoner.arm_ontologies.get(arm, []))
    if allowed:
        before = len(translatable)
        translatable = [r for r in translatable if r["ontology"] in allowed]
        print(f"\narm '{arm}' restricted to {sorted(allowed)}: "
              f"{len(translatable)} of {before} translatable definitions")
        if not translatable:
            print(f"NOT RUN: no definitions exist for {sorted(allowed)}. "
                  f"Phase 1 over those concepts is the prerequisite.")
            return

    by_ontology: dict[str, list[dict]] = defaultdict(list)
    for row in translatable:
        by_ontology[row["ontology"]].append(row)

    limit = int(cfg.reasoner.limit)
    summary = []
    for ontology, definitions in sorted(by_ontology.items()):
        override = (cfg.reasoner.get("arm_file_override", {}) or {}).get(arm, {}) or {}
        file_key = override.get(ontology, ontology)
        path = root / cfg.reasoner.ontology_files.get(file_key, "")
        if file_key != ontology:
            print(f"\n{ontology}: asserting into '{file_key}' ({path.name}) per arm_file_override")
        if not path or not path.exists():
            print(f"\n{ontology}: ontology file not found, skipped")
            continue
        sample = definitions[:limit]
        verdicts = _reason_over(path, sample, int(cfg.reasoner.timeout_s))
        counts = Counter(verdicts.values())
        unsat = counts["unsatisfiable"]
        scored = unsat + counts["satisfiable"]
        n_error = sum(v for k, v in counts.items() if k.startswith(("error", "setup_error")))
        low, high = clopper_pearson(unsat, scored, float(cfg.stats.conf_level))
        summary.append({
            "arm": arm, "ontology": ontology, "n_asserted": len(sample),
            "n_scored": scored, "unsatisfiable": unsat,
            "false_pass_rate": (unsat / scored) if scored else float("nan"),
            "ci_low": low, "ci_high": high,
            "errors": n_error, "inconsistent": counts["inconsistent"],
        })
        print(f"\n{ontology}: asserted={len(sample)} scored={scored} "
              f"unsat={unsat} errors={n_error}")

    out_path = out_dir / f"r4_10f_{arm}.json"
    out_path.write_text(json.dumps(
        {"arm": arm, "not_translatable": dict(reasons), "results": summary}, indent=2))
    print(f"\nwritten: {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
