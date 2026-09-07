"""Phase 2b: subClassOf transitivity probing via full ontology parent map.

For each sampled concept C, we enumerate triples (C, B, A) where:
  C subClassOf B  and  B subClassOf A  (from the full ontology, not just the sample).
We then ask the LLM: "Is C also a subclass of A?"  A "NO" answer = transitivity violation.

Phase 2 probes only named object properties
(Anatomy has 2, making C4's ≥3 qualifier hollow). subClassOf applies to all OWL
ontologies and produces much larger sample sizes.

Supported ontologies:
  - anatomy, cso  (OWL/rdf — owlready2/rdflib, uses is_a parents or cso:superTopicOf)
  - agrovoc       (SKOS — rdflib, uses skos:broader; loads slowly ~120s)
  - book          (OWL — flat hierarchy, all concepts → owl:Thing; 0 triples, self-skips)
  - gemet         (SKOS — numeric IRIs, no English labels in RDF; self-skips)
"""
from __future__ import annotations

import sys
import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

# Allow `from src.X import Y` whether invoked as a script or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Optional heavy imports — graceful fallback
try:
    import owlready2 as owl
except ImportError:
    owl = None  # type: ignore

try:
    from rdflib import Graph, URIRef
    from rdflib.namespace import SKOS
    _RDFLIB_AVAILABLE = True
except ImportError:
    _RDFLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Ontology metadata
# ---------------------------------------------------------------------------

ONTOLOGY_FILES = {
    "book":    ("book.rdf",           "owl"),
    "anatomy": ("anatomy.owl",        "owl"),
    "agrovoc": ("agrovoc_core.rdf",   "skos"),
    "gemet":   ("gemet.rdf",          "skos"),
    "cso":     ("cso.owl",            "cso"),   # custom schema: cso:superTopicOf
    "mesh":    ("mesh.ttl",           "mesh"),  # BioPortal: owl:Class + rdfs:subClassOf
}

_CSO_SUPER_TOPIC_OF = "http://cso.kmi.open.ac.uk/schema/cso#superTopicOf"

MAX_TRIPLES = 50
SAMPLE_SEED = 42

# Reasoning strategies need room to reason before their final verdict line; the
# zero-shot probe answers in one word. Using 16 tokens for every strategy would
# truncate CoT and ToT before they reach a verdict.
STRATEGY_MAX_TOKENS = {
    "zero_shot": 16,
    "cot": 200,
    "tot": 250,
    "self_consistency": 16,
}

# CPU inference on a laptop: a 200-token reasoning reply can take minutes, and
# the client default is far too short. Sized per strategy from a measured run.
STRATEGY_TIMEOUT_S = {
    "zero_shot": 120.0,
    "cot": 420.0,
    "tot": 480.0,
    "self_consistency": 120.0,
}

# Reasoning models (qwen3, deepseek-r1, …) spend the token budget thinking before
# they emit a verdict, and Ollama counts those thinking tokens against max_tokens.
# At the budgets above they hit finish_reason="length" mid-thought and return an
# empty string, which is why the first Phase 2b run scored qwen3 100% unclear and
# gemma4 93.5%. Measured on deepseek-r1:1.5b for one anatomy triple: 16 and 200
# tokens both return "", 800 returns a clean "YES".
#
# The escalation is applied on an empty reply rather than from a hardcoded model
# list, so a model that has never been seen before is handled too, and a model
# that answers within its normal budget costs nothing extra.
EMPTY_REPLY_MAX_TOKENS = 2000
EMPTY_REPLY_TIMEOUT_S = 900.0

# Self-consistency must sample, so it cannot run at the deterministic
# temperature used for the other strategies. Matches Phase 1, which samples
# self-consistency at 0.7.
SELF_CONSISTENCY_TEMPERATURE = 0.7

RESULT_COLS = [
    "onto", "model", "strategy",
    "triple_c", "triple_b", "triple_a",
    "c_label", "b_label", "a_label",
    "llm_answer", "is_violation", "is_unclear",
]

SUMMARY_COLS = [
    "onto", "model", "strategy",
    "n_triples", "n_violations", "subclassof_transitivity_violation_rate",
    "n_unclear",
]

# ---------------------------------------------------------------------------
# Parent-map builders (full ontology — not restricted to the sample)
# ---------------------------------------------------------------------------

def _build_parent_map_owl(onto_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (parent_map, label_map) from an OWL ontology via owlready2."""
    if owl is None:
        raise ImportError("owlready2 is required for OWL ontologies. pip install owlready2")
    onto = owl.get_ontology(str(onto_path)).load()
    parent_map: dict[str, list[str]] = {}
    label_map: dict[str, str] = {}
    for cls in onto.classes():
        iri = str(cls.iri)
        label_map[iri] = str(cls.label.first() or cls.name)
        parents = [str(p.iri) for p in cls.is_a if hasattr(p, "iri")]
        parent_map[iri] = parents
    return parent_map, label_map

def _build_parent_map_skos(onto_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (parent_map, label_map) using skos:broader as parent relation."""
    if not _RDFLIB_AVAILABLE:
        raise ImportError("rdflib is required for SKOS ontologies. pip install rdflib")
    g = Graph()
    g.parse(str(onto_path))

    # Build broader map: subject → list of broader concepts (= parents in hierarchy)
    parent_map: dict[str, list[str]] = {}
    for s, _, o in g.triples((None, SKOS.broader, None)):
        parent_map.setdefault(str(s), []).append(str(o))

    # Build English prefLabel map
    label_map: dict[str, str] = {}
    for s, _, o in g.triples((None, SKOS.prefLabel, None)):
        lang = getattr(o, "language", None)
        s_str = str(s)
        if lang == "en" or s_str not in label_map:
            label_map[s_str] = str(o)

    # Fallback: SKOS-XL prefLabel -> literalForm. AGROVOC publishes ZERO plain
    # skos:prefLabel triples, so without this every concept is unlabelled, the
    # transitivity probe finds nothing, and the run reports "no transitivity triples
    # ... may have a flat hierarchy". That reading is wrong: AGROVOC has 42,269
    # skos:broader triples and 41,261 A->B->C chains. `ontology_loader.py` already
    # had this fallback; Phase 2b did not, which is why Phase 2b silently covered
    # only 2 of the 4 study ontologies.
    if not label_map:
        from rdflib import Namespace
        SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
        for s, _, xl in g.triples((None, SKOSXL.prefLabel, None)):
            for lit in g.objects(xl, SKOSXL.literalForm):
                lang = getattr(lit, "language", None)
                s_str = str(s)
                if lang == "en" or s_str not in label_map:
                    label_map[s_str] = str(lit)

    return parent_map, label_map

def _build_parent_map_cso(onto_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (parent_map, label_map) from CSO using cso:superTopicOf as hierarchy.

    superTopicOf(B, A) means B is a sub-topic of A, i.e. A is the parent of B.
    We invert: parent_map[B] = [A, ...].
    """
    if not _RDFLIB_AVAILABLE:
        raise ImportError("rdflib is required for CSO. pip install rdflib")
    from rdflib import RDFS as _RDFS
    g = Graph()
    g.parse(str(onto_path))

    super_topic = URIRef(_CSO_SUPER_TOPIC_OF)
    parent_map: dict[str, list[str]] = {}
    # superTopicOf(subject=B, object=A) → A is parent of B
    for s, _, o in g.triples((None, super_topic, None)):
        parent_map.setdefault(str(s), []).append(str(o))

    label_map: dict[str, str] = {}
    for s, _, o in g.triples((None, _RDFS.label, None)):
        label_map[str(s)] = str(o)

    return parent_map, label_map

def _build_parent_map_mesh(onto_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (parent_map, label_map) from MeSH BioPortal Turtle.

    Hierarchy: rdfs:subClassOf among D-class descriptors (transitive IS-A).
    Labels: skos:prefLabel (English).
    """
    if not _RDFLIB_AVAILABLE:
        raise ImportError("rdflib is required for MeSH. pip install rdflib")
    from rdflib import RDFS as _RDFS
    from rdflib.namespace import SKOS as _SKOS
    print("  Loading full MeSH ontology for parent map (may take ~10 min) …")
    g = Graph()
    g.parse(str(onto_path), format="turtle")

    parent_map: dict[str, list[str]] = {}
    for s, _, o in g.triples((None, _RDFS.subClassOf, None)):
        ss, oo = str(s), str(o)
        if "/MESH/" in ss and "/MESH/" in oo:
            parent_map.setdefault(ss, []).append(oo)

    label_map: dict[str, str] = {}
    for s, _, o in g.triples((None, _SKOS.prefLabel, None)):
        lang = getattr(o, "language", None)
        s_str = str(s)
        if lang == "en" or s_str not in label_map:
            label_map[s_str] = str(o)
    return parent_map, label_map

def _cache_path(onto_name: str, onto_path: Path) -> Path:
    """Cache file keyed by ontology size and mtime, so a changed file misses."""
    stat = onto_path.stat()
    key = f"{onto_name}_{stat.st_size}_{int(stat.st_mtime)}"
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The key includes a loader version. The key is otherwise (size, mtime) of the
    # source file, so a fix to the LOADER never invalidates it: after the SKOS-XL
    # fallback was added, AGROVOC kept loading a cached label_map of 0 entries in
    # 0.1s and every cell still reported "no transitivity triples". Bump this
    # whenever the parent/label extraction changes.
    return cache_dir / f"parentmap_{key}_v2.json"

def build_parent_and_label_maps(
    onto_name: str, onto_dir: Path
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Dispatch to the correct parser and return (parent_map, label_map).

    The result is cached on disk. MeSH is an 803 MB Turtle file whose parent map
    takes about ten minutes to build, and Phase 2b invokes this once per cell, so
    without a cache the same graph would be re-parsed sixteen times. JSON is used
    rather than pickle: the payload is plain string data and JSON carries no
    code-execution risk.
    """
    if onto_name not in ONTOLOGY_FILES:
        raise ValueError(f"Unknown ontology '{onto_name}'. Choose from: {list(ONTOLOGY_FILES)}")
    fname, fmt = ONTOLOGY_FILES[onto_name]
    onto_path = onto_dir / fname
    if not onto_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {onto_path}")

    cache_file = _cache_path(onto_name, onto_path)
    if cache_file.exists():
        try:
            with open(cache_file) as handle:
                cached = json.load(handle)
            print(f"  Parent map loaded from cache ({cache_file.name})")
            return cached["parent_map"], cached["label_map"]
        except (json.JSONDecodeError, KeyError, OSError):
            print("  Parent-map cache unreadable; rebuilding")

    if fmt == "owl":
        result = _build_parent_map_owl(onto_path)
    elif fmt == "skos":
        result = _build_parent_map_skos(onto_path)
    elif fmt == "cso":
        result = _build_parent_map_cso(onto_path)
    elif fmt == "mesh":
        result = _build_parent_map_mesh(onto_path)
    else:
        result = None

    if result is not None:
        tmp = cache_file.with_suffix(".tmp")
        with open(tmp, "w") as handle:
            json.dump({"parent_map": result[0], "label_map": result[1]}, handle)
        tmp.replace(cache_file)   # atomic, so a killed run leaves no half-file
        print(f"  Parent map cached to {cache_file.name}")
        return result
    else:
        raise ValueError(f"Unknown format: {fmt}")

# ---------------------------------------------------------------------------
# Triple enumeration
# ---------------------------------------------------------------------------

def enumerate_triples(
    concepts: list[dict],
    parent_map: dict[str, list[str]],
    label_map: dict[str, str],
    max_triples: int = MAX_TRIPLES,
    seed: int = SAMPLE_SEED,
) -> list[dict]:
    """Enumerate subClassOf transitivity triples (C, B, A) where C⊑B and B⊑A.

    C must be a sampled concept; B and A come from the full ontology label_map.
    Returns at most `max_triples` triples, random-sampled with `seed`.
    """
    all_triples: list[dict] = []
    for concept in concepts:
        c_iri = concept["iri"]
        c_label = concept["label"]
        for b_iri in parent_map.get(c_iri, []):
            b_label = label_map.get(b_iri)
            if not b_label:
                continue  # B not in ontology label map — skip
            for a_iri in parent_map.get(b_iri, []):
                if a_iri == c_iri:
                    continue  # avoid trivial self-loops
                a_label = label_map.get(a_iri)
                if not a_label:
                    continue
                all_triples.append({
                    "triple_c": c_iri,
                    "triple_b": b_iri,
                    "triple_a": a_iri,
                    "c_label": c_label,
                    "b_label": b_label,
                    "a_label": a_label,
                })

    if len(all_triples) <= max_triples:
        return all_triples

    rng = random.Random(seed)
    return rng.sample(all_triples, max_triples)

# ---------------------------------------------------------------------------
# Core phase runner
# ---------------------------------------------------------------------------

def run_phase2b(cfg: Any, out_dir: Path) -> None:
    """Entry point called from main.py (phase="2b" or standalone)."""
    onto_name = cfg.ontology.name
    seed = cfg.sampling.seed
    model = cfg.llm.model
    strategy = cfg.prompting.strategy

    concepts_path = Path("data/samples") / f"concepts_{onto_name}_seed{seed}.json"
    if not concepts_path.exists():
        raise FileNotFoundError(f"Run phase=0 first: {concepts_path}")

    with open(concepts_path) as f:
        concepts = json.load(f)

    onto_dir = Path(cfg.ontology.path)
    print(f"  Loading full ontology for '{onto_name}' to build parent map …")
    t_load = time.time()
    parent_map, label_map = build_parent_and_label_maps(onto_name, onto_dir)
    print(f"  Loaded in {time.time() - t_load:.1f}s  "
          f"({len(parent_map)} nodes with parents, {len(label_map)} labels)")

    triples = enumerate_triples(concepts, parent_map, label_map,
                                max_triples=getattr(cfg, "max_triples", MAX_TRIPLES),
                                seed=seed)
    print(f"  {len(triples)} transitivity triples to probe")

    if not triples:
        print(f"  WARNING: no transitivity triples found for '{onto_name}'. "
              "This ontology may have a flat hierarchy (e.g. book) or missing labels. Skipping.")
        return

    results_path = out_dir / "phase2b_results.csv"
    summary_path = out_dir / "phase2b_summary.csv"

    # Resume logic: key = (triple_c, triple_b, triple_a, model, strategy)
    completed_keys: set[tuple[str, str, str, str, str]] = set()
    if results_path.exists() and results_path.stat().st_size > 0:
        with open(results_path) as f:
            for row in csv.DictReader(f):
                key = (row["triple_c"], row["triple_b"], row["triple_a"],
                       row["model"], row["strategy"])
                completed_keys.add(key)
        print(f"  Resume mode: {len(completed_keys)} triples already done.")
        write_mode, write_header = "a", False
    else:
        write_mode, write_header = "w", True

    from src.prompting.llm_client import get_client, call_llm  # noqa: E402 (path set at module top)
    from src.prompting.prompter import (  # noqa: E402
        build_subclassof_prompts, parse_yes_no, FORCED_VERDICT_QUESTION)
    client = get_client(cfg)

    n_violations = 0
    n_done = 0
    n_unclear = 0

    t0 = time.time()
    with open(results_path, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()

        for triple in triples:
            key = (triple["triple_c"], triple["triple_b"], triple["triple_a"], model, strategy)
            if key in completed_keys:
                continue

            # The prompt now depends on the strategy. An earlier implementation
            # was a fixed prompt and `strategy` was only an output column, so at
            # temperature 0 all four strategies re-ran one identical probe and
            # produced bit-identical counts. See build_subclassof_prompts().
            max_tokens = STRATEGY_MAX_TOKENS[strategy]
            timeout_s = STRATEGY_TIMEOUT_S[strategy]

            if strategy == "self_consistency":
                chains = build_subclassof_prompts(
                    triple, strategy, n_samples=getattr(cfg.llm, "n_samples", 5))
                votes = []
                for chain in chains:
                    sample = call_llm(client, model, chain,
                                      temperature=SELF_CONSISTENCY_TEMPERATURE,
                                      max_tokens=max_tokens, timeout=timeout_s)
                    if not sample.strip():
                        sample = call_llm(client, model, chain,
                                          temperature=SELF_CONSISTENCY_TEMPERATURE,
                                          max_tokens=EMPTY_REPLY_MAX_TOKENS,
                                          timeout=EMPTY_REPLY_TIMEOUT_S)
                    vote = parse_yes_no(sample)
                    if vote is not None:
                        votes.append(vote)
                n_candidates = len(votes)
                # Majority vote; a tie counts as unclear rather than as agreement.
                if not votes or sum(votes) * 2 == len(votes):
                    verdict = None
                else:
                    verdict = int(sum(votes) * 2 > len(votes))
                answer = f"SELF_CONSISTENCY VOTES={votes}"
            else:
                messages = build_subclassof_prompts(triple, strategy)
                answer = call_llm(client, model, messages,
                                  temperature=cfg.llm.temperature,
                                  max_tokens=max_tokens, timeout=timeout_s)
                if not answer.strip():
                    # Thinking consumed the whole budget; retry once with room to
                    # finish. See EMPTY_REPLY_MAX_TOKENS.
                    answer = call_llm(client, model, messages,
                                      temperature=cfg.llm.temperature,
                                      max_tokens=EMPTY_REPLY_MAX_TOKENS,
                                      timeout=EMPTY_REPLY_TIMEOUT_S)
                verdict = parse_yes_no(answer)
                if verdict is None:
                    # One cheap forced-choice follow-up before giving up, so that
                    # a rambling reply is not silently scored as "no violation".
                    followup = messages + [
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": FORCED_VERDICT_QUESTION},
                    ]
                    retry = call_llm(client, model, followup,
                                     temperature=cfg.llm.temperature,
                                     max_tokens=12, timeout=STRATEGY_TIMEOUT_S["zero_shot"])
                    if not retry.strip():
                        # 12 tokens cannot hold a reasoning model's thinking, so the
                        # forced question came back empty too, giving "|| FORCED:".
                        retry = call_llm(client, model, followup,
                                         temperature=cfg.llm.temperature,
                                         max_tokens=EMPTY_REPLY_MAX_TOKENS,
                                         timeout=EMPTY_REPLY_TIMEOUT_S)
                    verdict = parse_yes_no(retry)
                    answer = f"{answer} || FORCED: {retry}"
                n_candidates = 1

            # An unrecoverable verdict is excluded from the rate entirely; it is
            # neither a violation nor a compliance.
            unclear = verdict is None
            is_violation = int(verdict == 1)
            if unclear:
                n_unclear += 1
            n_violations += is_violation
            n_done += 1

            row = {
                "onto": onto_name,
                "model": model,
                "strategy": strategy,
                "triple_c": triple["triple_c"],
                "triple_b": triple["triple_b"],
                "triple_a": triple["triple_a"],
                "c_label": triple["c_label"],
                "b_label": triple["b_label"],
                "a_label": triple["a_label"],
                # Collapse whitespace: a raw newline inside a quoted CSV field
                # splits the record across physical lines and corrupts later parsing
                # (unescaped newlines here corrupt downstream CSV parsing).
                "llm_answer": " ".join(answer.split()).upper()[:300],
                "is_violation": is_violation,
                "is_unclear": int(unclear),
            }
            writer.writerow(row)
            f.flush()

            if n_done % 10 == 0 or n_done == len(triples) - len(completed_keys):
                elapsed = time.time() - t0
                vrate = n_violations / n_done if n_done else 0.0
                print(f"  [{n_done}/{len(triples)}] violation_rate={vrate:.3f}  "
                      f"elapsed={elapsed:.1f}s")

    # Recount from full CSV for the summary (covers resumed runs)
    total_triples, total_violations, total_unclear = 0, 0, 0
    if results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if row["model"] == model and row["strategy"] == strategy and row["onto"] == onto_name:
                    if int(row.get("is_unclear", 0) or 0):
                        total_unclear += 1
                        continue
                    total_triples += 1
                    total_violations += int(row["is_violation"])

    viol_rate = round(total_violations / total_triples, 4) if total_triples else 0.0

    # Append summary row (overwrite summary for this (onto, model, strategy) combination)
    _update_summary(summary_path, {
        "onto": onto_name,
        "model": model,
        "strategy": strategy,
        "n_triples": total_triples,
        "n_violations": total_violations,
        "subclassof_transitivity_violation_rate": viol_rate,
        "n_unclear": total_unclear,
    })

    elapsed_total = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"Phase 2b complete — {onto_name} × {model} × {strategy}")
    print(f"  Triples: {total_triples}  Violations: {total_violations}  Rate: {viol_rate:.4f}")
    print(f"  Wall time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"{'─'*60}")

def _update_summary(summary_path: Path, new_row: dict) -> None:
    """Write or update a summary CSV row for (onto, model, strategy)."""
    rows: list[dict] = []
    key = (new_row["onto"], new_row["model"], new_row["strategy"])
    found = False

    if summary_path.exists() and summary_path.stat().st_size > 0:
        with open(summary_path) as f:
            for row in csv.DictReader(f):
                if (row["onto"], row["model"], row["strategy"]) == key:
                    rows.append(new_row)
                    found = True
                else:
                    rows.append(row)

    if not found:
        rows.append(new_row)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# Standalone CLI (for dry-run and direct invocation)
# ---------------------------------------------------------------------------

def _make_dummy_cfg(onto_name: str, model: str, strategy: str,
                    seed: int, api_base: str,
                    max_triples: int = MAX_TRIPLES) -> Any:
    """Build a minimal config namespace for standalone use."""
    from types import SimpleNamespace
    return SimpleNamespace(
        ontology=SimpleNamespace(name=onto_name, path="data/ontologies/"),
        sampling=SimpleNamespace(seed=seed),
        llm=SimpleNamespace(
            model=model,
            api_base=api_base,
            temperature=0.0,
            max_tokens=16,
        ),
        prompting=SimpleNamespace(strategy=strategy),
        max_triples=max_triples,
        output=SimpleNamespace(dir="outputs/results"),
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2b — subClassOf transitivity probing")
    parser.add_argument("--ontology", default="book",
                        choices=list(ONTOLOGY_FILES),
                        help="Ontology to probe (default: book)")
    parser.add_argument("--model", default="llama3.2:latest",
                        help="LLM model identifier (default: llama3.2:latest)")
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "cot", "tot", "self_consistency"],
                        help="Prompting strategy (default: zero_shot)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--api_base", default="http://localhost:11434/v1",
                        help="LLM API base URL")
    parser.add_argument("--out_dir", default=None,
                        help="Override output root. REQUIRED when varying --seed: the "
                             "resume key is (triple, model, strategy) and does NOT "
                             "include the seed, so a second seed written to the same "
                             "directory would be skipped as already done.")
    parser.add_argument("--max_triples", type=int, default=MAX_TRIPLES,
                        help=f"Triples probed per cell (default: {MAX_TRIPLES})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print first 3 triples without calling the LLM then exit")
    args = parser.parse_args()

    cfg = _make_dummy_cfg(
        max_triples=args.max_triples,
        onto_name=args.ontology,
        model=args.model,
        strategy=args.strategy,
        seed=args.seed,
        api_base=args.api_base,
    )

    # Load concept sample
    concepts_path = Path("data/samples") / f"concepts_{args.ontology}_seed{args.seed}.json"
    if not concepts_path.exists():
        print(f"ERROR: {concepts_path} not found. Run phase=0 first.")
        return

    with open(concepts_path) as f:
        concepts = json.load(f)

    onto_dir = Path(cfg.ontology.path)
    print(f"Loading full ontology '{args.ontology}' …")
    try:
        parent_map, label_map = build_parent_and_label_maps(args.ontology, onto_dir)
    except Exception as e:
        print(f"ERROR building parent map: {e}")
        return

    triples = enumerate_triples(concepts, parent_map, label_map,
                                max_triples=args.max_triples, seed=args.seed)
    print(f"Total transitivity triples: {len(triples)}")

    if not triples:
        print(f"No triples found for '{args.ontology}' — hierarchy may be flat.")
        return

    if args.dry_run:
        print("\n--- Dry run: first 3 triples ---")
        for i, t in enumerate(triples[:3], 1):
            print(f"  [{i}] C='{t['c_label']}' ⊑ B='{t['b_label']}' ⊑ A='{t['a_label']}'")
            print(f"       Question: Is '{t['c_label']}' also a subclass of '{t['a_label']}'?")
        return

    model_safe = args.model.replace(":", "_")
    root = args.out_dir if args.out_dir else cfg.output.dir
    out_dir = Path(root) / args.ontology / model_safe / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    run_phase2b(cfg, out_dir)

if __name__ == "__main__":
    main()
