"""Mechanical stem-overlap pass over the manual audit (all 96 items).

Why this exists. The audit distinguishes two things that look alike:

  circular    the definiens restates the definiendum -- verbatim, by alias, by
              synonym, by acronym expansion, or by negation. A defect.
  stem_reuse  a word from the concept label recurs in the definiens, but the
              definiens genuinely decomposes. NOT a defect: supplying the genus
              of "Mediastinal_Lymph_Node" as "Lymph_Node" is what a definition
              is supposed to do.

`circular` is a judgement and stays with the author. `stem_reuse` is mechanical,
so it is computed here rather than annotated, which makes it comparable across
all 96 items. Items 1-64 were annotated before the tag existed, so without this
pass `stem_reuse` would read 0% for the older models and 18.8% for the newer
ones -- an artefact of when the tag was introduced, not a property of the models.

The two are reported as nested, not exclusive: a circular definition normally
reuses a stem as well. So `stem_reuse` counts all stem-sharing rows, and
`circular` is the subset where the reuse restates instead of decomposing.

Deliberately crude and auditable: lowercase alphabetic tokens, logical and
stop-words removed, compared on a fixed-length prefix. Every matched stem is
written back to the row so a reader can check it. A prefix match will over-fire
on unrelated words that happen to share an opening (an earlier pass flagged
"2-Acetolactate Mutase" against "Acetolactate"), so the matched stems are
reported and the rate is a ceiling.

No LLM calls. CPU-only, local, instant.

Run:
    python src/analysis/stem_overlap.py
    python src/analysis/stem_overlap.py stem.write=true
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

#: Logic keywords and prose stop-words. These recur in every definiens and would
#: otherwise match any label containing them, which measures nothing.
_IGNORE = {
    "forall", "exists", "and", "or", "not", "iff", "implies", "true", "false",
    "the", "a", "an", "of", "for", "with", "that", "this", "which", "such",
    "is", "are", "be", "has", "have", "its", "it", "in", "on", "at", "to",
    "from", "by", "as", "if", "then", "all", "any", "some", "each", "every",
    "x", "y", "z", "s", "t", "n", "set", "type", "kind", "thing", "entity",
    "step", "genus", "differentia", "necessary", "sufficient", "condition",
    "conditions", "biconditional", "definition", "define", "defined", "where",
}

def _stems(text: str, prefix_len: int, min_len: int) -> set[str]:
    words = re.split(r"[^A-Za-z]+", (text or "").lower())
    return {w[:prefix_len] for w in words
            if len(w) >= min_len and w not in _IGNORE}

@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    stem = cfg.stem
    path = root / stem.worksheet
    rows = list(csv.DictReader(open(path, newline="", errors="replace"), delimiter=";"))
    fieldnames = list(rows[0].keys())

    older = set(stem.older_models)
    counts = {"older": Counter(), "newer": Counter()}
    totals = {"older": 0, "newer": 0}
    examples: list[tuple[str, str, list[str]]] = []

    for row in rows:
        group = "older" if row.get("llm_model") in older else "newer"
        totals[group] += 1
        label_stems = _stems(row.get("concept_label", ""), stem.prefix_len, stem.min_word_len)
        body = (row.get("extracted_fol_block") or "")[: int(stem.body_chars)]
        shared = sorted(label_stems & _stems(body, stem.prefix_len, stem.min_word_len))

        tags = [t for t in (row.get("DEF_QUALITY") or "").split("+") if t]
        if shared:
            counts[group]["stem_reuse"] += 1
            if "stem_reuse" not in tags:
                tags.append("stem_reuse")
            if len(examples) < 12:
                examples.append((row["item"], row.get("concept_label", "")[:34], shared[:3]))
        else:
            tags = [t for t in tags if t != "stem_reuse"]
        if "circular" in tags:
            counts[group]["circular"] += 1
        row["DEF_QUALITY"] = "+".join(tags)
        row["GENUS_CHECK"] = ("stems:" + ",".join(shared)) if shared else "no_stem_overlap"

    if bool(stem.write):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"written: {path.relative_to(root)}")
    else:
        print("DRY RUN — nothing written. Re-run with stem.write=true to persist.\n")

    print(f"{'group':8s} {'n':>4s} {'stem_reuse':>12s} {'circular':>10s}  circular as % of stem_reuse")
    for group in ("older", "newer"):
        n = totals[group]
        sr = counts[group]["stem_reuse"]
        ci = counts[group]["circular"]
        share = f"{100 * ci / sr:.0f}%" if sr else "-"
        print(f"{group:8s} {n:4d} {sr:6d} {100*sr/n:5.1f}% {ci:5d} {100*ci/n:5.1f}%   {share:>6s}")
    total_n = sum(totals.values())
    total_sr = sum(counts[g]["stem_reuse"] for g in counts)
    print(f"{'all':8s} {total_n:4d} {total_sr:6d} {100*total_sr/total_n:5.1f}%")
    print("\nsample matches (verify these — a prefix match over-fires):")
    for item, label, shared in examples:
        print(f"  {item:>3s} {label:36s} {shared}")

if __name__ == "__main__":
    main()
