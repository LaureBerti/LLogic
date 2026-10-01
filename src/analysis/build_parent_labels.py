"""Resolve each sampled concept's parent IRIs to their human-readable labels.

Why. Phase 3 scores the reconstructed ontology against a ground-truth edge set built in
``_load_gt_graph``, which originally paired a concept's *label* with the last path segment
of its parent's IRI:

    parent_label = parent_iri.split("/")[-1].split("#")[-1]

For the two ontologies that actually carry subclass edges in the sample that yields an
opaque code -- ``(Stratified_Epithelium, NCI_C12710)`` for Anatomy,
``(Metabolome, D008660)`` for MeSH -- while a reconstruction names both endpoints in words.
The measured overlap was therefore zero for every cell regardless of what the model
produced, and the reported ``edge_coverage`` of 0.000 across all 24 scorable cells was an
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
(~1 min). No LLM calls.

Configuration lives in ``conf/analysis.yaml`` under ``parent_labels`` -- the ontology
sources, the predicates, the builtin pattern, the paths and the seed are all config fields,
overridable with Hydra dot-notation. Print the resolved config without running via
``--cfg job``.

Run:
    python src/analysis/build_parent_labels.py
    python src/analysis/build_parent_labels.py parent_labels.seed=123
    python src/analysis/build_parent_labels.py --cfg job
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class MissingSample(FileNotFoundError):
    """The concept sample for this (ontology, seed) does not exist."""


def _targets(samples_dir: Path, ontology: str, seed: int, builtin: re.Pattern) -> tuple[set[str], int]:
    """The distinct parent IRIs we need labels for, and how many links reference them.

    Raises MissingSample rather than returning an empty set: an absent sample file and a
    sample whose parents are all builtins are different situations, and reporting the first
    as the second hides a wrong seed behind a plausible-looking message.
    """
    path = samples_dir / f"concepts_{ontology}_seed{seed}.json"
    if not path.exists():
        raise MissingSample(path)
    concepts = json.loads(path.read_text())
    want: set[str] = set()
    links = 0
    for c in concepts:
        for p in c.get("parents", []):
            if builtin.search(p):
                continue
            want.add(p)
            links += 1
    return want, links


def _from_rdfxml(path: Path, want: set[str], base: str) -> dict[str, str]:
    """rdfs:label inside owl:Class blocks. Handles rdf:about="#Frag" shorthand."""
    text = path.read_text(encoding="utf8", errors="replace")
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


def _resolve(spec: DictConfig, path: Path, want: set[str]) -> dict[str, str]:
    """Dispatch on the configured serialisation format."""
    fmt = str(spec.format)
    if fmt == "rdfxml":
        return _from_rdfxml(path, want, str(spec.base))
    if fmt == "turtle":
        return _from_turtle(path, want, str(spec.iri_prefix))
    raise ValueError(f"unknown format {fmt!r}; expected 'rdfxml' or 'turtle'")


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    pl = cfg.parent_labels
    seed = int(pl.seed)
    samples_dir = root / str(pl.samples_dir)
    ontologies_dir = root / str(pl.ontologies_dir)
    builtin = re.compile(str(pl.builtin_pattern))

    missing: list[str] = []
    for ontology, spec in pl.sources.items():
        try:
            want, links = _targets(samples_dir, ontology, seed, builtin)
        except MissingSample as exc:
            print(f"  {ontology:9} NO SAMPLE at {exc.args[0]} -- wrong seed?")
            missing.append(ontology)
            continue
        if not want:
            print(f"  {ontology:9} no non-builtin parents in the sample; nothing to resolve")
            continue
        src = ontologies_dir / str(spec.file)
        if not src.exists():
            print(f"  {ontology:9} SKIPPED: {src} not found")
            continue
        mapping = _resolve(spec, src, want)
        out = samples_dir / str(pl.out_pattern).format(ontology=ontology, seed=seed)
        out.write_text(json.dumps(mapping, indent=1, sort_keys=True))
        print(
            f"  {ontology:9} {len(mapping)}/{len(want)} distinct parent IRIs resolved "
            f"({links} links) -> {out.relative_to(root)}"
        )

    for ontology in pl.skip:
        try:
            want, links = _targets(samples_dir, ontology, seed, builtin)
        except MissingSample:
            print(f"  {ontology:9} skipped: no sample for seed {seed}")
            missing.append(ontology)
            continue
        note = "all parents are builtins" if links == 0 else f"{links} links, no resolver"
        print(f"  {ontology:9} skipped: {note}")

    if len(missing) == len(pl.sources) + len(pl.skip):
        raise SystemExit(
            f"no concept sample found for seed {seed} in {samples_dir} -- nothing was written"
        )


if __name__ == "__main__":
    main()
