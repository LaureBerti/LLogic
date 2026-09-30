"""Resolve each sampled concept's parent IRIs to their human-readable labels.

Why. Phase 3 scores the reconstructed ontology against a ground-truth edge set built in
``_load_gt_graph``, which pairs a concept's *label* with the last path segment of its
parent's IRI:

    parent_label = parent_iri.split("/")[-1].split("#")[-1]

For the two ontologies that actually carry subclass edges in the sample that yields an
opaque code -- ``(Stratified_Epithelium, NCI_C12710)`` for Anatomy,
``(Metabolome, D008660)`` for MeSH -- while a reconstruction names both endpoints in words.
The measured overlap was therefore zero for every cell regardless of what the model
produced, and the reported ``edge_coverage`` of 0.000 across all 24 scorable cells is an
artifact of this, not a result.

The labels are present in the source ontologies, under a different predicate in each:

  * ``anatomy.owl``  -- RDF/XML, ``rdfs:label``      (9,403 label elements)
  * ``mesh.ttl``     -- Turtle, ``skos:prefLabel``   (803 MB; streamed, not parsed)
  * ``book.rdf``     -- every sampled parent is ``owl:Thing``, so Book contributes no
                        informative edge and is skipped rather than resolved.

AGROVOC and CSO have no ``parents`` entries in their samples at all (SKOS, no OWL subclass
edges in the sampled subgraph), so they have no ground-truth edges to resolve.

Output: ``data/samples/parent_labels_<ontology>_seed<seed>.json``, an IRI -> label map that
``_load_gt_graph`` consults when present. Writing it as a separate artifact keeps the fix
auditable and leaves the published results reproducible under the original code path.

Compute: CPU only, local, no network. Anatomy is instant; MeSH streams 803 MB once
(~1 min).

Run:
    python src/analysis/build_parent_labels.py
    python src/analysis/build_parent_labels.py --seed 123
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SAMPLES = Path("data/samples")
ONTOLOGIES = Path("data/ontologies")

# owl:Thing and other builtins are not informative parents.
BUILTIN = re.compile(r"(owl|rdfs|rdf)#(Thing|Resource|Class)$")


def _targets(ontology: str, seed: int) -> tuple[set[str], int]:
    """The distinct parent IRIs we need labels for, and how many links reference them."""
    path = SAMPLES / f"concepts_{ontology}_seed{seed}.json"
    concepts = json.loads(path.read_text())
    want, links = set(), 0
    for c in concepts:
        for p in c.get("parents", []):
            if BUILTIN.search(p):
                continue
            want.add(p)
            links += 1
    return want, links


def _from_rdfxml(path: Path, want: set[str]) -> dict[str, str]:
    """rdfs:label inside owl:Class blocks. Handles rdf:about="#Frag" shorthand."""
    text = path.read_text(encoding="utf8", errors="replace")
    base = "http://human.owl"
    out: dict[str, str] = {}
    for m in re.finditer(r'<owl:Class\b[^>]*rdf:about="([^"]+)"(.*?)</owl:Class>', text, re.S):
        iri = m.group(1)
        full = iri if iri.startswith("http") else base + iri
        if full not in want:
            continue
        lab = re.search(r"<rdfs:label[^>]*>([^<]+)</rdfs:label>", m.group(2))
        if lab:
            out[full] = lab.group(1).strip()
    return out


def _from_turtle(path: Path, want: set[str], prefix: str) -> dict[str, str]:
    """skos:prefLabel, streamed. The file is too large to parse into a graph."""
    subj_re = re.compile(rf"^<({re.escape(prefix)}[^>]+)>")
    lab_re = re.compile(r'skos:prefLabel\s+(?:"""(.*?)"""|"([^"]*)")')
    out: dict[str, str] = {}
    current: str | None = None
    with path.open(encoding="utf8", errors="replace") as f:
        for line in f:
            if line[:1] == "<":
                m = subj_re.match(line)
                current = m.group(1) if m else None
                continue
            if current in want and "skos:prefLabel" in line:
                m = lab_re.search(line)
                if m:
                    out[current] = (m.group(1) or m.group(2) or "").strip()
    return out


RESOLVERS = {
    "anatomy": lambda want: _from_rdfxml(ONTOLOGIES / "anatomy.owl", want),
    "mesh": lambda want: _from_turtle(
        ONTOLOGIES / "mesh.ttl", want, "http://purl.bioontology.org/ontology/MESH/"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for ontology, resolve in RESOLVERS.items():
        want, links = _targets(ontology, args.seed)
        if not want:
            print(f"  {ontology:9} no non-builtin parents in the sample; nothing to resolve")
            continue
        mapping = resolve(want)
        out = SAMPLES / f"parent_labels_{ontology}_seed{args.seed}.json"
        out.write_text(json.dumps(mapping, indent=1, sort_keys=True))
        print(
            f"  {ontology:9} {len(mapping)}/{len(want)} distinct parent IRIs resolved "
            f"({links} links) -> {out}"
        )

    for ontology in ("book", "agrovoc", "cso"):
        want, links = _targets(ontology, args.seed)
        note = "all parents are builtins" if links == 0 else f"{links} links, no resolver"
        print(f"  {ontology:9} skipped: {note}")


if __name__ == "__main__":
    main()
