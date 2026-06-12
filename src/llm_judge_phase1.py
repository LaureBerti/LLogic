"""LLM-as-judge for Phase 1: semantic contradiction detection.

For each row in phase1_results.csv, calls a local LLM (default: llama3.2:latest)
to judge whether the FOL definition is logically self-contradictory.

Adds column `llm_judge_contradiction` (1=yes, 0=no, -1=unsure/failed) to a new
file phase1_results_judged.csv. Original CSV is never modified.

The judge sees only the FOL block, not the solver result, so this is an independent signal.

Usage:
    python src/llm_judge_phase1.py [--judge-model llama3.2:latest] [--api-base http://localhost:11434/v1]
    python src/llm_judge_phase1.py --dry-run  # show first 3 prompts, no API calls
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

JUDGE_PROMPT_SYSTEM = """\
You are a formal logic checker. Your only task is to determine whether a first-order \
logic (FOL) formula is unsatisfiable due to an explicit syntactic contradiction \
in its own symbols.\
"""

JUDGE_PROMPT_TEMPLATE = """\
Does this FOL formula contain an explicit logical contradiction that makes it unsatisfiable?

Formula: {fol_block}

A formula is contradictory ONLY if it literally contains BOTH P(x) AND not P(x) applied \
to the same predicate and variable, or asserts C(x) <-> not C(x).

Examples:
  "forall x: C(x) <-> (P(x) and not P(x))"  →  YES  (P and not P in body)
  "forall x: C(x) <-> not C(x)"              →  YES  (self-negation)
  "forall x: TechReport(x)"                  →  NO   (satisfiable; no negation)
  "forall x: C(x) <-> (exists y: R(y, x))"   →  NO   (normal definition)
  "forall x: C(x) <-> (P(x) or not P(x))"   →  NO   (tautology, not contradiction)

Do NOT consider real-world plausibility. Only look at the formula's own symbols.
If there is no explicit P(x) and not P(x) pattern, answer NO.

Reason step by step, then end your response with exactly one line: YES, NO, or UNSURE.\
"""


def build_judge_messages(concept_label: str, fol_block: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_PROMPT_SYSTEM},
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
            concept_label=concept_label, fol_block=fol_block
        )},
    ]


def parse_judge_response(response: str) -> int:
    """Return 1=yes, 0=no, -1=unsure.

    Scans from the last line upward so reasoning chains don't confuse the parser.
    DeepSeek-R1 ends its response with the final verdict after the <think> block
    has been stripped by call_llm().
    """
    if not response:
        return -1
    for line in reversed(response.strip().splitlines()):
        token = line.strip().upper()
        if token in ("YES", "YES.", "**YES**"):
            return 1
        if token in ("NO", "NO.", "**NO**"):
            return 0
        if token in ("UNSURE", "UNSURE."):
            return -1
        # also accept lines that are purely YES/NO with punctuation
        if token.startswith("YES") and len(token) <= 5:
            return 1
        if token.startswith("NO") and len(token) <= 4:
            return 0
    # fallback: search anywhere in last 200 chars
    tail = response[-200:].upper()
    if "\nYES" in tail or tail.endswith("YES") or tail.endswith("YES."):
        return 1
    if "\nNO" in tail or tail.endswith("NO") or tail.endswith("NO."):
        return 0
    return -1


def judge_file(path: Path, client: object, model: str, dry_run: bool = False) -> Path:
    from src.prompting.llm_client import call_llm

    rows = list(csv.DictReader(open(path)))
    if not rows:
        return path

    out_path = path.with_name("phase1_results_judged.csv")
    already_judged: set[str] = set()

    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                already_judged.add(f"{row['concept_iri']}|{row['strategy']}")

    new_rows = []
    judged = 0
    skipped = 0

    for row in rows:
        key = f"{row['concept_iri']}|{row['strategy']}"
        new_row = dict(row)

        if key in already_judged:
            # preserve existing judge result (will be merged on write)
            skipped += 1
            new_rows.append(new_row)
            continue

        fol_block = row.get("fol_block", "").strip()
        concept_label = row.get("concept_label", "")

        if not fol_block:
            new_row["llm_judge_contradiction"] = -1
            new_row["llm_judge_response"] = "NO_FOL_BLOCK"
            new_rows.append(new_row)
            continue

        if dry_run:
            msgs = build_judge_messages(concept_label, fol_block)
            print(f"\n--- Prompt for: {concept_label} ---")
            print(msgs[-1]["content"][:300])
            new_row["llm_judge_contradiction"] = -1
            new_row["llm_judge_response"] = "DRY_RUN"
        else:
            msgs = build_judge_messages(concept_label, fol_block)
            response = call_llm(client, model, msgs, temperature=0.0, max_tokens=512, timeout=120.0)
            verdict = parse_judge_response(response)
            new_row["llm_judge_contradiction"] = verdict
            new_row["llm_judge_response"] = response.strip()[:200]
            label_str = row["concept_label"]
            print(f"  {label_str:<30} judge={'YES' if verdict==1 else 'NO' if verdict==0 else 'UNSURE'}")

        judged += 1
        new_rows.append(new_row)

    if not dry_run:
        # Merge with already-judged rows
        if already_judged and out_path.exists():
            existing = {f"{r['concept_iri']}|{r['strategy']}": r
                        for r in csv.DictReader(open(out_path))}
            merged = []
            for r in new_rows:
                k = f"{r['concept_iri']}|{r['strategy']}"
                if k in existing and "llm_judge_contradiction" in existing[k]:
                    r = dict(r)
                    r["llm_judge_contradiction"] = existing[k]["llm_judge_contradiction"]
                    r["llm_judge_response"] = existing[k].get("llm_judge_response", "")
                merged.append(r)
            new_rows = merged

        fieldnames = list(new_rows[0].keys()) if new_rows else []
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)

    print(f"  → {path.parts[-4]}/{path.parts[-3]}/{path.parts[-2]}: {judged} judged, {skipped} skipped")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="mistral:7b")
    parser.add_argument("--api-base", default="http://localhost:11434/v1")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling API")
    args = parser.parse_args()

    results_root = Path("outputs/results")
    csvs = sorted(results_root.rglob("phase1_results.csv"))

    if not csvs:
        print("No phase1_results.csv files found.")
        return

    client = None
    if not args.dry_run:
        from openai import OpenAI
        api_key = "ollama" if "11434" in args.api_base or "localhost" in args.api_base else "key"
        client = OpenAI(api_key=api_key, base_url=args.api_base)
        print(f"Judge model: {args.judge_model} via {args.api_base}")
        # Pre-warm: force model load before the timed batch begins
        print("Warming up model (may take ~60s on cold start)...")
        from src.prompting.llm_client import call_llm
        try:
            call_llm(client, args.judge_model, [{"role": "user", "content": "Say OK"}],
                     max_tokens=5, timeout=120.0)
            print("Model ready.")
        except Exception as e:
            print(f"Warm-up failed: {e} — proceeding anyway")

    print(f"Processing {len(csvs)} phase1_results.csv files{'  [DRY RUN]' if args.dry_run else ''}")

    for path in csvs:
        judge_file(path, client, args.judge_model, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nDone. Judged CSVs: phase1_results_judged.csv alongside each input.")
        print("Run src/reanalyze_phase1.py to also apply the improved Z3 parser (if not done).")


if __name__ == "__main__":
    main()
