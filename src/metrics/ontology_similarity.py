"""
Compute structural and semantic similarity between two ontology graphs.

onto_sim_structural: normalised graph edit distance (GED) ∈ [0,1], higher = more similar.
onto_sim_semantic:   mean cosine similarity of concept label embeddings (matched pairs).
concept_coverage:    |LLM ∩ GT| / |GT|  (label-normalised)
edge_coverage:       |LLM_edges ∩ GT_edges| / |GT_edges|
"""
from __future__ import annotations
import re
from typing import Any

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_ST = True
except ImportError:
    HAS_ST = False

_OWL_BUILTINS = frozenset({"owl", "rdfs", "rdf", "xsd", "thing", "class", "nothing"})

def _norm_label(s: str) -> str:
    """Lowercase + collapse underscores/hyphens to spaces for fuzzy label matching."""
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_manchester_to_graph(manchester_text: str) -> Any | None:
    """
    Multi-format parser for LLM-generated ontology text.

    Handles surface forms observed across LLMs:
      1. Manchester OWL SubClassOf  — Mistral, Qwen (with optional [.;] terminators)
      2. Turtle  "ClassName a owl:Class"  — Gemma
      3. LLaMA hybrid "Class:\\n  Name" + "rdfs:subClassOf [ X ]"
      4. RDF/XML <Class rdf:ID="Name"> + <rdfs:subClassOf rdf:resource="#Parent"/>
      5. URI Turtle  <URI> a rdfs:Class/owl:Class  — Gemma cso
      6. Prefixed  prefix:Name owl:Class .  — LLaMA cso
      7. rdf:type  X rdf:type owl:Class .  — Mistral anatomy
      8. Arrow  X |-> Y  — Qwen cso CoT
      9. rdfs:subClassOf prefix:Parent ;  — linked to nearest class

    Returns a DiGraph of (child, parent, rel=subClassOf) edges, or None on failure.
    """
    if not HAS_NX:
        return None

    g = nx.DiGraph()

    # ── 1: Manchester OWL SubClassOf — allows optional [.;] terminators ───────
    subclass_re = re.compile(
        r"(\w[\w\s]*?)\s+SubClassOf[:\s]+(\w[\w\s]*?)\s*[.;]?\s*(?:\n|$)",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in subclass_re.finditer(manchester_text):
        child = m.group(1).strip()
        parent = m.group(2).strip()
        g.add_edge(child, parent, rel="subClassOf")

    # ── 2: Turtle "ClassName a owl:Class" ─────────────────────────────────────
    turtle_class_re = re.compile(r"^\s*(\w+)\s+a\s+owl:Class", re.MULTILINE)
    for m in turtle_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

    # ── 3: Manchester "Class: ClassName" or LLaMA "Class:\n  ClassName" ───────
    class_decl_re = re.compile(
        r"Class:\s*(?:(\w[\w]*)[ \t]*\n|\n[ \t]+(\w[\w]*))",
        re.MULTILINE,
    )
    class_positions: dict[int, str] = {}
    for m in class_decl_re.finditer(manchester_text):
        cls = (m.group(1) or m.group(2) or "").strip()
        if cls and cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            class_positions[m.start()] = cls

    # ── 4: LLaMA "rdfs:subClassOf [ ClassName ]" → link to nearest class ─────
    rdf_subclass_re = re.compile(r"rdfs:subClassOf\s*\[\s*(\w[\w\s]*?)\s*\]", re.MULTILINE)
    for m in rdf_subclass_re.finditer(manchester_text):
        parent = m.group(1).strip()
        if parent.lower() in _OWL_BUILTINS:
            continue
        pos = m.start()
        nearest_cls = min(
            ((cpos, cname) for cpos, cname in class_positions.items() if cpos < pos),
            key=lambda x: pos - x[0],
            default=(None, None),
        )[1]
        if nearest_cls:
            g.add_edge(nearest_cls, parent, rel="subClassOf")
        else:
            g.add_node(parent)

    # ── 5: "X sub Y" or "X sub Y;" ───────────────────────────────────────────
    sub_keyword_re = re.compile(r"(\w[\w]*)\s+sub\s+(\w[\w]*)\s*[;,]?", re.MULTILINE)
    for m in sub_keyword_re.finditer(manchester_text):
        child = m.group(1).strip()
        parent = m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")

    # ── 6: "X ⊃ Y" description-logic notation ────────────────────────────────
    dl_subclass_re = re.compile(r"(\w[\w]*)\s+[⊃⊂]\s+(\w[\w]*)", re.MULTILINE)
    for m in dl_subclass_re.finditer(manchester_text):
        child, parent = m.group(1).strip(), m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")

    # ── 7: rdfs:subClassOf <URI/ClassName> (Gemma CoT URI style) ─────────────
    uri_subclass_re = re.compile(
        r"rdfs:subClassOf\s+<[^>]*/(\w[\w]*)>", re.MULTILINE
    )
    for m in uri_subclass_re.finditer(manchester_text):
        parent = m.group(1).strip()
        if parent.lower() in _OWL_BUILTINS:
            continue
        pos = m.start()
        md_cls_re = re.compile(r"\*\*(\w[\w]*):\*\*")
        chunk = manchester_text[max(0, pos - 300): pos]
        md_matches = list(md_cls_re.finditer(chunk))
        if md_matches:
            child = md_matches[-1].group(1).strip()
            if child.lower() not in _OWL_BUILTINS:
                g.add_edge(child, parent, rel="subClassOf")
        else:
            g.add_node(parent)

    # ── 8: Markdown "**ClassName:**" standalone declarations ──────────────────
    md_class_re = re.compile(r"\*\*(\w[\w]*):\*\*")
    for m in md_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

    # ── 9: RDF/XML <Class rdf:ID="Name"> + <rdfs:subClassOf rdf:resource="#Parent"/> ──
    rdfxml_block_re = re.compile(
        r'<Class\s+rdf:ID="(\w[\w]*)"\s*>.*?<rdfs:subClassOf\s+rdf:resource="#(\w[\w]*)"\s*/>',
        re.DOTALL,
    )
    for m in rdfxml_block_re.finditer(manchester_text):
        child, parent = m.group(1).strip(), m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")
    rdfxml_class_re = re.compile(r'<Class\s+rdf:ID="(\w[\w]*)"\s*>', re.MULTILINE)
    for m in rdfxml_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            class_positions[m.start()] = cls

    # ── 10: <URI> a rdfs:Class or owl:Class — extract local name from URI ─────
    uri_a_class_re = re.compile(
        r"<[^>]*?(\w[\w]*)>\s+a\s+(?:owl|rdfs):Class\s*[.;,]?", re.MULTILINE
    )
    uri_class_positions: dict[int, str] = {}
    for m in uri_a_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            uri_class_positions[m.start()] = cls

    # ── 11: prefix:Name owl:Class . (e.g. cso:BuildingModel owl:Class .) ──────
    prefix_class_re = re.compile(
        r"(?:\w+:)(\w[\w]*)\s+owl:Class\s*[.;]?", re.MULTILINE
    )
    for m in prefix_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            class_positions[m.start()] = cls

    # ── 12: X rdf:type owl:Class . (e.g. AnatomicalRegion rdf:type owl:Class .) ──
    rdf_type_class_re = re.compile(
        r"^(\w[\w]*)\s+rdf:type\s+owl:Class\s*[.;]?", re.MULTILINE
    )
    for m in rdf_type_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            class_positions[m.start()] = cls

    # Build merged position map for linking patterns below
    all_class_positions = {**class_positions, **uri_class_positions}

    # ── 13: rdfs:subClassOf <URI> (any form) — link to nearest preceding class ─
    uri_subclass_any_re = re.compile(
        r"rdfs:subClassOf\s+<[^>]*?(\w[\w]*)>", re.MULTILINE
    )
    for m in uri_subclass_any_re.finditer(manchester_text):
        parent = m.group(1).strip()
        if parent.lower() in _OWL_BUILTINS:
            continue
        pos = m.start()
        nearest_cls = min(
            ((cpos, cname) for cpos, cname in all_class_positions.items() if cpos < pos),
            key=lambda x: pos - x[0],
            default=(None, None),
        )[1]
        if nearest_cls and nearest_cls != parent:
            g.add_edge(nearest_cls, parent, rel="subClassOf")
        else:
            g.add_node(parent)

    # ── 14: rdfs:subClassOf prefix:Parent ; — link to nearest preceding class ──
    prefixed_parent_re = re.compile(
        r"rdfs:subClassOf\s+\w+:(\w[\w]*)\s*[.;]?", re.MULTILINE
    )
    for m in prefixed_parent_re.finditer(manchester_text):
        parent = m.group(1).strip()
        if parent.lower() in _OWL_BUILTINS:
            continue
        pos = m.start()
        nearest_cls = min(
            ((cpos, cname) for cpos, cname in all_class_positions.items() if cpos < pos),
            key=lambda x: pos - x[0],
            default=(None, None),
        )[1]
        if nearest_cls and nearest_cls != parent:
            g.add_edge(nearest_cls, parent, rel="subClassOf")
        else:
            g.add_node(parent)

    # ── 15: Arrow format prefix:X |-> prefix:Y (Qwen cso CoT) ───────────────
    arrow_subclass_re = re.compile(
        r"(?:\w+:)?(\w[\w]*)\s*\|->\s*(?:\w+:)?(\w[\w]*)", re.MULTILINE
    )
    for m in arrow_subclass_re.finditer(manchester_text):
        child, parent = m.group(1).strip(), m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")

    # ── 16: rdfs:label "name" — treat label values as class node candidates ───
    rdfs_label_re = re.compile(r'rdfs:label\s+"([^"]+)"', re.MULTILINE)
    for m in rdfs_label_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS and len(cls) > 1:
            g.add_node(cls)

    # ── 17: Concept/Class: <URI> — extract local name from angle-bracket URI ─
    # Handles Gemma ZS "Concept: <http://...#disease>" and "SubClassOf: <URI>"
    concept_uri_re = re.compile(
        r"(?:Concept|Class):\s*<[^>]*?(\w[\w]*)>", re.MULTILINE | re.IGNORECASE
    )
    for m in concept_uri_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

    # ── 18: Class: name(args) — FOL-style class declaration (Qwen ZS) ─────────
    # Handles "Class: sawfish(x)\n" → "sawfish"
    class_fol_re = re.compile(
        r"Class:\s+(\w[\w\s]*?)\s*\([^)]*\)\s*\n", re.MULTILINE
    )
    for m in class_fol_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls and cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

    if g.number_of_nodes() == 0:
        return None
    return g

def structural_similarity(gt_graph: Any, llm_graph: Any) -> float:
    """
    Approximate normalised GED via node/edge overlap (exact GED is NP-hard).
    Returns value ∈ [0, 1]. Higher = more similar.
    """
    if not HAS_NX or gt_graph is None or llm_graph is None:
        return float("nan")

    gt_nodes = set(_norm_label(n) for n in gt_graph.nodes())
    llm_nodes = set(_norm_label(n) for n in llm_graph.nodes())
    gt_edges = {(_norm_label(a), _norm_label(b)) for a, b in gt_graph.edges()}
    llm_edges = {(_norm_label(a), _norm_label(b)) for a, b in llm_graph.edges()}

    node_overlap = len(gt_nodes & llm_nodes) / max(len(gt_nodes | llm_nodes), 1)
    edge_overlap = len(gt_edges & llm_edges) / max(len(gt_edges | llm_edges), 1)

    if node_overlap + edge_overlap == 0:
        return 0.0
    return round(2 * node_overlap * edge_overlap / (node_overlap + edge_overlap), 4)

def semantic_similarity(gt_concepts: list[str], llm_concepts: list[str], model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Average cosine similarity between GT concept labels and their nearest LLM concept match.
    Requires sentence-transformers.
    """
    if not HAS_ST or not gt_concepts or not llm_concepts:
        return float("nan")

    model = SentenceTransformer(model_name)
    gt_embs = model.encode(gt_concepts, normalize_embeddings=True)
    llm_embs = model.encode(llm_concepts, normalize_embeddings=True)

    sims = (gt_embs @ llm_embs.T).max(axis=1)
    return round(float(np.mean(sims)), 4)

def compute_coverage(gt_concepts: list[str], llm_concepts: list[str],
                     gt_edges: list[tuple], llm_edges: list[tuple]) -> dict:
    gt_set = set(_norm_label(c) for c in gt_concepts)
    llm_set = set(_norm_label(c) for c in llm_concepts)
    gt_edge_set = {(_norm_label(a), _norm_label(b)) for a, b in gt_edges}
    llm_edge_set = {(_norm_label(a), _norm_label(b)) for a, b in llm_edges}

    concept_cov = round(len(gt_set & llm_set) / max(len(gt_set), 1), 4)
    edge_cov = round(len(gt_edge_set & llm_edge_set) / max(len(gt_edge_set), 1), 4)

    return {
        "concept_coverage": concept_cov,
        "edge_coverage": edge_cov,
        "n_gt_concepts": len(gt_set),
        "n_llm_concepts": len(llm_set),
        "n_gt_edges": len(gt_edge_set),
        "n_llm_edges": len(llm_edge_set),
    }
