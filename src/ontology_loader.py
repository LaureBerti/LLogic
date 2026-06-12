"""Load OWL/RDF ontologies and produce stratified concept/relation samples."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any

try:
    import owlready2 as owl
except ImportError:
    owl = None  # type: ignore

try:
    from rdflib import Graph, RDF, RDFS, OWL
    from rdflib.namespace import SKOS, Namespace
    SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
except ImportError:
    Graph = None  # type: ignore
    SKOSXL = None  # type: ignore


ONTOLOGY_FILES = {
    "book":    ("book.rdf",    "owl"),   # RDF/XML OWL — .rdf URL is live; .owl URL is dead
    "anatomy": ("anatomy.owl", "owl"),
    "agrovoc": ("agrovoc_core.rdf", "rdf"),
    "gemet":   ("gemet.rdf",   "rdf"),
    "cso":     ("cso.owl",     "rdf"),   # uses cso:Topic, not owl:Class — rdflib required
    "snomed":  ("snomed.owl",  "snomed"), # SNOMED CT OWL — custom loader (see _sample_snomed)
    "mesh":    ("mesh.ttl",   "mesh"),   # MeSH RDF Turtle — custom loader (see _sample_mesh)
}


def load_and_sample(cfg: Any) -> None:
    """Entry point for phase=0: load ontology, sample concepts & relations, write JSON."""
    name = cfg.ontology.name
    if name not in ONTOLOGY_FILES:
        raise ValueError(f"Unknown ontology: {name}. Choose from {list(ONTOLOGY_FILES)}")

    fname, fmt = ONTOLOGY_FILES[name]
    onto_path = Path(cfg.ontology.path) / fname

    if not onto_path.exists():
        raise FileNotFoundError(
            f"Ontology file not found: {onto_path}\n"
            f"Download instructions:\n"
            f"  Book:    http://oaei.ontologymatching.org/2007/benchmarks/232/onto.rdf  → save as book.rdf\n"
            f"  Anatomy: http://oaei.ontologymatching.org/2007/anatomy/nci_anatomy.owl\n"
            f"  AGROVOC: https://www.fao.org/agrovoc/ (SKOS RDF)\n"
            f"  GEMET:   https://www.eionet.europa.eu/gemet (RDF)\n"
            f"  CSO:     https://cso.kmi.open.ac.uk/home/\n"
        )

    rng = random.Random(cfg.sampling.seed)

    if fmt == "snomed" and Graph is not None:
        concepts, relations = _sample_snomed(onto_path, cfg, rng)
    elif fmt == "mesh" and Graph is not None:
        concepts, relations = _sample_mesh(onto_path, cfg, rng)
    elif fmt == "owl" and owl is not None:
        concepts, relations = _sample_owl(onto_path, cfg, rng)
    elif Graph is not None:
        concepts, relations = _sample_rdf(onto_path, cfg, rng)
    else:
        raise ImportError("Install owlready2 and rdflib: pip install owlready2 rdflib")

    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.sampling.seed

    concept_file = out_dir / f"concepts_{name}_seed{seed}.json"
    relation_file = out_dir / f"edges_{name}_seed{seed}.json"

    with open(concept_file, "w") as f:
        json.dump(concepts, f, indent=2)
    with open(relation_file, "w") as f:
        json.dump(relations, f, indent=2)

    print(f"  Sampled {len(concepts)} concepts → {concept_file}")
    print(f"  Sampled {len(relations)} relations → {relation_file}")


def _sample_owl(onto_path: Path, cfg: Any, rng: random.Random) -> tuple[list, list]:
    """Sample from OWL ontology using owlready2."""
    onto = owl.get_ontology(str(onto_path)).load()

    all_classes = list(onto.classes())

    # Stratify by depth (approximate via ancestor count)
    def depth(cls: Any) -> int:
        return len(list(cls.ancestors()))

    depths = [(depth(c), c) for c in all_classes]
    depths.sort(key=lambda x: x[0])
    n = len(depths)
    thirds = n // 3
    leaves = [c for _, c in depths[n - thirds:]]
    intermediates = [c for _, c in depths[thirds: n - thirds]]
    roots = [c for _, c in depths[:thirds]]

    n_per_stratum = cfg.sampling.n_concepts // 3
    sampled = (
        rng.sample(leaves, min(n_per_stratum, len(leaves)))
        + rng.sample(intermediates, min(n_per_stratum, len(intermediates)))
        + rng.sample(roots, min(n_per_stratum, len(roots)))
    )

    concepts = [
        {
            "iri": str(c.iri),
            "label": str(c.label.first() or c.name),
            "depth": depth(c),
            "stratum": "leaf" if depth(c) >= depths[n - thirds][0] else "root" if depth(c) < depths[thirds][0] else "intermediate",
            "parents": [str(p.iri) for p in c.is_a if hasattr(p, "iri")],
            "definition": str(c.comment.first() or ""),
        }
        for c in sampled
    ]

    # Sample object properties
    all_props = list(onto.object_properties())
    sampled_props = rng.sample(all_props, min(cfg.sampling.n_relations, len(all_props)))
    relations = [
        {
            "iri": str(p.iri),
            "label": str(p.label.first() or p.name),
            "is_transitive": owl.TransitiveProperty in p.is_a,
            "is_symmetric": owl.SymmetricProperty in p.is_a,
            "domain": [str(d.iri) for d in p.domain if hasattr(d, "iri")],
            "range": [str(r.iri) for r in p.range if hasattr(r, "iri")],
        }
        for p in sampled_props
    ]

    return concepts, relations


def _dominant_type_subjects(g: Any) -> list:
    """Return subjects of the most frequent rdf:type in the graph (fallback for custom schemas)."""
    from collections import Counter
    from rdflib import RDF as _RDF
    type_counts: Counter = Counter()
    for _, _, o in g.triples((None, _RDF.type, None)):
        type_counts[o] += 1
    if not type_counts:
        return []
    dominant = type_counts.most_common(1)[0][0]
    return list(g.subjects(_RDF.type, dominant))


def _sample_rdf(onto_path: Path, cfg: Any, rng: random.Random) -> tuple[list, list]:
    """Sample from RDF/SKOS ontology using rdflib."""
    g = Graph()
    g.parse(str(onto_path))

    # Collect SKOS concepts or OWL classes; fall back to dominant rdf:type in file
    concepts_iris = (
        list(g.subjects(RDF.type, SKOS.Concept))
        or list(g.subjects(RDF.type, OWL.Class))
        or _dominant_type_subjects(g)
    )
    concepts_iris = [str(c) for c in concepts_iris if str(c).startswith("http")]

    sampled_iris = rng.sample(concepts_iris, min(cfg.sampling.n_concepts, len(concepts_iris)))
    concepts = []
    for iri in sampled_iris:
        from rdflib import URIRef
        ref = URIRef(iri)
        # Try standard SKOS prefLabel first
        labels = [o for o in g.objects(ref, SKOS.prefLabel)
                  if not hasattr(o, 'language') or o.language in ('en', None)]
        if not labels:
            labels = list(g.objects(ref, SKOS.prefLabel))
        # Fallback: SKOS-XL prefLabel → literalForm (used by AGROVOC)
        if not labels and SKOSXL is not None:
            for xl_ref in g.objects(ref, SKOSXL.prefLabel):
                en_forms = [o for o in g.objects(xl_ref, SKOSXL.literalForm)
                            if hasattr(o, 'language') and o.language == 'en']
                if en_forms:
                    labels = en_forms
                    break
            if not labels:
                for xl_ref in g.objects(ref, SKOSXL.prefLabel):
                    forms = list(g.objects(xl_ref, SKOSXL.literalForm))
                    if forms:
                        labels = [forms[0]]
                        break
        # Fallback: rdfs:label
        if not labels:
            labels = list(g.objects(ref, RDFS.label))
        defs = list(g.objects(ref, SKOS.definition)) or list(g.objects(ref, RDFS.comment))
        concepts.append({
            "iri": iri,
            "label": str(labels[0]) if labels else iri.split("/")[-1],
            "depth": -1,  # depth not trivially available in SKOS
            "stratum": "unknown",
            "definition": str(defs[0]) if defs else "",
        })

    # Relations: use SKOS broader/narrower/related
    relations = [
        {"label": "broader",  "iri": str(SKOS.broader),  "is_transitive": True,  "is_symmetric": False},
        {"label": "narrower", "iri": str(SKOS.narrower), "is_transitive": True,  "is_symmetric": False},
        {"label": "related",  "iri": str(SKOS.related),  "is_transitive": False, "is_symmetric": True},
    ]

    return concepts, relations


def _sample_go(onto_path: Path, cfg: Any, rng: random.Random) -> tuple[list, list]:
    """Sample from Gene Ontology OWL using rdflib.

    GO uses owl:Class with rdfs:label (English) and oboInOwl:hasDefinition.
    Depth is approximated via BFS on rdfs:subClassOf from owl:Thing.
    Key object properties: part_of (transitive), regulates (non-transitive),
    positively/negatively_regulates.
    """
    from rdflib import URIRef, Literal
    from rdflib.namespace import OWL as _OWL, RDFS as _RDFS, RDF as _RDF
    from collections import deque

    OBO = "http://purl.obolibrary.org/obo/"
    OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"

    print(f"  Loading Gene Ontology from {onto_path} (may take 30–60s)…")
    g = Graph()
    g.parse(str(onto_path))
    print(f"  Loaded {len(g):,} triples.")

    # Collect named GO classes (exclude blank nodes and obsolete terms)
    deprecated = set(str(s) for s in g.subjects(_OWL.deprecated, Literal(True)))
    go_classes = [
        str(s) for s in g.subjects(_RDF.type, _OWL.Class)
        if str(s).startswith(OBO + "GO_") and str(s) not in deprecated
    ]
    print(f"  Found {len(go_classes):,} non-deprecated GO classes.")

    # Compute approximate depth via BFS on rdfs:subClassOf
    # child → set(parents)
    parent_map: dict[str, set[str]] = {}
    for child, parent in g.subject_objects(_RDFS.subClassOf):
        child_s, parent_s = str(child), str(parent)
        if child_s.startswith(OBO + "GO_") and parent_s.startswith(OBO + "GO_"):
            parent_map.setdefault(child_s, set()).add(parent_s)

    depth_map: dict[str, int] = {}
    queue: deque = deque()
    # Seed with roots (no parents in GO namespace)
    for iri in go_classes:
        if not parent_map.get(iri):
            depth_map[iri] = 0
            queue.append(iri)
    # BFS downward via child → parent inverted
    child_map: dict[str, set[str]] = {}
    for child, parents in parent_map.items():
        for p in parents:
            child_map.setdefault(p, set()).add(child)
    while queue:
        iri = queue.popleft()
        d = depth_map[iri]
        for child in child_map.get(iri, set()):
            if child not in depth_map:
                depth_map[child] = d + 1
                queue.append(child)
    # Assign depth=0 to any class not reached
    for iri in go_classes:
        depth_map.setdefault(iri, 0)

    # Stratified sample: ~1/3 leaf (high depth), 1/3 intermediate, 1/3 root (low depth)
    sorted_go = sorted(go_classes, key=lambda x: depth_map[x])
    n = len(sorted_go)
    third = n // 3
    roots_list  = sorted_go[:third]
    inter_list  = sorted_go[third: 2 * third]
    leaves_list = sorted_go[2 * third:]

    n_per = cfg.sampling.n_concepts // 3
    sampled = (
        rng.sample(roots_list,  min(n_per, len(roots_list)))
        + rng.sample(inter_list,  min(n_per, len(inter_list)))
        + rng.sample(leaves_list, min(n_per + cfg.sampling.n_concepts % 3, len(leaves_list)))
    )

    def _go_label(iri: str) -> str:
        ref = URIRef(iri)
        for label in g.objects(ref, _RDFS.label):
            if not hasattr(label, 'language') or label.language in ('en', None):
                return str(label)
        return iri.split("_")[-1]

    def _go_def(iri: str) -> str:
        ref = URIRef(iri)
        for prop in [URIRef(OBO_IN_OWL + "hasDefinition"),
                     URIRef(OBO + "IAO_0000115"),
                     _RDFS.comment]:
            defs = list(g.objects(ref, prop))
            if defs:
                return str(defs[0])
        return ""

    concepts = [
        {
            "iri": iri,
            "label": _go_label(iri),
            "depth": depth_map[iri],
            "stratum": ("root" if depth_map[iri] < depth_map.get(sorted_go[third], 1)
                        else "leaf" if depth_map[iri] >= depth_map.get(sorted_go[2 * third], 99)
                        else "intermediate"),
            "parents": [str(p) for p in parent_map.get(iri, set())],
            "definition": _go_def(iri),
        }
        for iri in sampled
    ]

    # Object properties: part_of (transitive), regulates, has_part
    go_props = [
        {"label": "part_of",              "iri": OBO + "BFO_0000050", "is_transitive": True,  "is_symmetric": False},
        {"label": "has_part",             "iri": OBO + "BFO_0000051", "is_transitive": True,  "is_symmetric": False},
        {"label": "regulates",            "iri": OBO + "RO_0002211",  "is_transitive": False, "is_symmetric": False},
        {"label": "positively_regulates", "iri": OBO + "RO_0002213",  "is_transitive": False, "is_symmetric": False},
        {"label": "negatively_regulates", "iri": OBO + "RO_0002212",  "is_transitive": False, "is_symmetric": False},
    ]
    relations = go_props[:cfg.sampling.n_relations]

    return concepts, relations


def _sample_snomed(onto_path: Path, cfg: Any, rng: random.Random) -> tuple[list, list]:
    """Sample from SNOMED CT OWL (RDF/XML format from international release).

    Expects: data/ontologies/snomed.owl
    Download: SNOMED CT International RF2 release → extract SnomedCT_*.owl
              OR use snomed-owl-toolkit to generate the OWL file.
    Concepts: owl:Class with rdfs:label (FSN) and sct:definition.
    Key relations (transitive): Is a (116680003), Part of (123005000).
    """
    from rdflib import URIRef, Literal
    from rdflib.namespace import OWL as _OWL, RDFS as _RDFS, RDF as _RDF
    from collections import deque

    SCT = "http://snomed.info/id/"
    print(f"  Loading SNOMED CT OWL from {onto_path} (may take 60–120s)…")
    g = Graph()
    g.parse(str(onto_path))
    print(f"  Loaded {len(g):,} triples.")

    # Collect SNOMED concept classes (filter to SCT namespace)
    deprecated = set(str(s) for s in g.subjects(_OWL.deprecated, Literal(True)))
    sct_classes = [
        str(s) for s in g.subjects(_RDF.type, _OWL.Class)
        if str(s).startswith(SCT) and str(s) not in deprecated
    ]
    print(f"  Found {len(sct_classes):,} SNOMED CT concepts.")

    # Depth via BFS on rdfs:subClassOf
    parent_map: dict[str, set[str]] = {}
    for child, parent in g.subject_objects(_RDFS.subClassOf):
        cs, ps = str(child), str(parent)
        if cs.startswith(SCT) and ps.startswith(SCT):
            parent_map.setdefault(cs, set()).add(ps)

    depth_map: dict[str, int] = {}
    child_map: dict[str, set[str]] = {}
    for child, parents in parent_map.items():
        for p in parents:
            child_map.setdefault(p, set()).add(child)
    roots = [iri for iri in sct_classes if not parent_map.get(iri)]
    queue: deque = deque()
    for r in roots:
        depth_map[r] = 0
        queue.append(r)
    while queue:
        iri = queue.popleft()
        for child in child_map.get(iri, set()):
            if child not in depth_map:
                depth_map[child] = depth_map[iri] + 1
                queue.append(child)
    for iri in sct_classes:
        depth_map.setdefault(iri, 0)

    sorted_sct = sorted(sct_classes, key=lambda x: depth_map[x])
    n = len(sorted_sct)
    third = n // 3
    n_per = cfg.sampling.n_concepts // 3

    sampled = (
        rng.sample(sorted_sct[:third], min(n_per, third))
        + rng.sample(sorted_sct[third: 2 * third], min(n_per, third))
        + rng.sample(sorted_sct[2 * third:], min(n_per + cfg.sampling.n_concepts % 3, n - 2 * third))
    )

    def _sct_label(iri: str) -> str:
        ref = URIRef(iri)
        labels_en = [str(o) for o in g.objects(ref, _RDFS.label)
                     if not hasattr(o, 'language') or o.language in ('en', None)]
        if labels_en:
            # Prefer FSN (ends with "(…)")
            fsn = [l for l in labels_en if l.endswith(')')]
            return fsn[0] if fsn else labels_en[0]
        return iri.split("/")[-1]

    concepts = [
        {
            "iri": iri,
            "label": _sct_label(iri),
            "depth": depth_map[iri],
            "stratum": ("root" if depth_map[iri] < depth_map.get(sorted_sct[third], 1)
                        else "leaf" if depth_map[iri] >= depth_map.get(sorted_sct[2 * third], 99)
                        else "intermediate"),
            "parents": [str(p) for p in parent_map.get(iri, set())],
            "definition": "",
        }
        for iri in sampled
    ]

    # SNOMED CT key object properties
    snomed_props = [
        {"label": "is_a",              "iri": SCT + "116680003", "is_transitive": True,  "is_symmetric": False},
        {"label": "part_of",           "iri": SCT + "123005000", "is_transitive": True,  "is_symmetric": False},
        {"label": "finding_site",      "iri": SCT + "363698007", "is_transitive": False, "is_symmetric": False},
        {"label": "causative_agent",   "iri": SCT + "246075003", "is_transitive": False, "is_symmetric": False},
        {"label": "associated_morphology", "iri": SCT + "116676008", "is_transitive": False, "is_symmetric": False},
    ]
    relations = snomed_props[:cfg.sampling.n_relations]

    return concepts, relations


def _sample_mesh(onto_path: Path, cfg: Any, rng: random.Random) -> tuple[list, list]:
    """Sample from MeSH BioPortal/UMLS Turtle (owl:Class + rdfs:subClassOf + skos:prefLabel).

    This handles the BioPortal UMLS2RDF version of MeSH where:
      - Main headings: IRI http://purl.bioontology.org/ontology/MESH/D{code}
      - Supplementary: IRI http://purl.bioontology.org/ontology/MESH/C{code}
      - Hierarchy: rdfs:subClassOf (transitive IS_A)
      - Labels: skos:prefLabel (English)
      - Definitions: skos:definition
    """
    from rdflib import URIRef
    from rdflib.namespace import OWL as _OWL, RDFS as _RDFS, RDF as _RDF
    from rdflib import Namespace
    from collections import deque

    MESHB = "http://purl.bioontology.org/ontology/MESH/"
    SKOS_NS = Namespace("http://www.w3.org/2004/02/skos/core#")

    print(f"  Loading MeSH RDF from {onto_path} (may take 5-15 min)…")
    g = Graph()
    g.parse(str(onto_path), format="turtle")
    print(f"  Loaded {len(g):,} triples.")

    # D-class only (main headings), not C-class (supplementary concepts)
    mesh_d = [
        str(s) for s in g.subjects(_RDF.type, _OWL.Class)
        if str(s).startswith(MESHB + "D")
    ]
    print(f"  Found {len(mesh_d):,} MeSH D-class descriptors.")

    # Hierarchy via rdfs:subClassOf (child rdfs:subClassOf parent)
    parent_map: dict[str, set[str]] = {}
    for child, parent in g.subject_objects(_RDFS.subClassOf):
        cs, ps = str(child), str(parent)
        if cs.startswith(MESHB + "D") and ps.startswith(MESHB + "D"):
            parent_map.setdefault(cs, set()).add(ps)

    # BFS depth
    depth_map: dict[str, int] = {}
    child_map: dict[str, set[str]] = {}
    for child, parents in parent_map.items():
        for p in parents:
            child_map.setdefault(p, set()).add(child)
    roots = [d for d in mesh_d if not parent_map.get(d)]
    queue: deque = deque()
    for r in roots:
        depth_map[r] = 0
        queue.append(r)
    while queue:
        iri = queue.popleft()
        for child in child_map.get(iri, set()):
            if child not in depth_map:
                depth_map[child] = depth_map[iri] + 1
                queue.append(child)
    for d in mesh_d:
        depth_map.setdefault(d, 0)

    sorted_mesh = sorted(mesh_d, key=lambda x: depth_map[x])
    n = len(sorted_mesh)
    third = max(n // 3, 1)
    n_per = cfg.sampling.n_concepts // 3

    sampled = (
        rng.sample(sorted_mesh[:third], min(n_per, third))
        + rng.sample(sorted_mesh[third: 2 * third], min(n_per, max(third, 1)))
        + rng.sample(sorted_mesh[2 * third:], min(n_per + cfg.sampling.n_concepts % 3, n - 2 * third))
    )

    def _label(iri: str) -> str:
        ref = URIRef(iri)
        for label in g.objects(ref, SKOS_NS.prefLabel):
            if not hasattr(label, "language") or label.language in ("en", None):
                return str(label)
        for label in g.objects(ref, _RDFS.label):
            return str(label)
        return iri.split("/")[-1]

    def _defn(iri: str) -> str:
        ref = URIRef(iri)
        defs = list(g.objects(ref, SKOS_NS.definition))
        return str(defs[0]) if defs else ""

    concepts = [
        {
            "iri": iri,
            "label": _label(iri),
            "depth": depth_map[iri],
            "stratum": ("root" if depth_map[iri] < depth_map.get(sorted_mesh[third], 1)
                        else "leaf" if depth_map[iri] >= depth_map.get(sorted_mesh[2 * third], 99)
                        else "intermediate"),
            "parents": [str(p) for p in parent_map.get(iri, set())],
            "definition": _defn(iri),
        }
        for iri in sampled
    ]

    # rdfs:subClassOf is transitive IS_A — ideal for Phase 2b
    # No symmetric relation in BioPortal MeSH; use mapped_from as a weak association
    MESH_NS = Namespace(MESHB)
    relations = [
        {"label": "is_a",        "iri": str(_RDFS.subClassOf),        "is_transitive": True,  "is_symmetric": False},
        {"label": "mapped_from", "iri": MESHB + "mapped_from",        "is_transitive": False, "is_symmetric": False},
        {"label": "mapped_to",   "iri": MESHB + "mapped_to",          "is_transitive": False, "is_symmetric": False},
    ]

    return concepts, relations
