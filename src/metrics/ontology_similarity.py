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

try:
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False


_OWL_BUILTINS = frozenset({"owl", "rdfs", "rdf", "xsd", "thing", "class", "nothing"})


def _norm_label(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _local_name(term: Any) -> str:
    if not isinstance(term, rdflib.URIRef):
        return ""
    s = str(term)
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s.strip()


def _rdf_graph(text: str) -> Any | None:
    if not (HAS_RDFLIB and HAS_NX):
        return None

    blocks = re.findall(r"```[A-Za-z0-9_-]*\n(.*?)```", text, re.DOTALL) or [text]
    g = nx.DiGraph()
    found = False

    for block in blocks:
        for fmt in ("turtle", "xml"):
            rg = rdflib.Graph()
            try:
                rg.parse(data=block, format=fmt)
            except Exception:
                continue
            for subject in rg.subjects(RDF.type, OWL.Class):
                name = _local_name(subject)
                if name and name.lower() not in _OWL_BUILTINS:
                    g.add_node(name)
                    found = True
            for child, parent in rg.subject_objects(RDFS.subClassOf):
                cn, pn = _local_name(child), _local_name(parent)
                if (cn and pn and cn.lower() not in _OWL_BUILTINS
                        and pn.lower() not in _OWL_BUILTINS):
                    g.add_edge(cn, pn, rel="subClassOf")
                    found = True
            break

    return g if found else None


_TTL_CLASS = re.compile(
    r"^\s*(?:[\w.-]*:)?([A-Za-z_][\w.-]*)\s+(?:a|rdf:type)\s+(?:owl|rdfs):Class",
    re.IGNORECASE,
)
_TTL_SUBJECT = re.compile(
    r"^\s*(?:[\w.-]*:)?([A-Za-z_][\w.-]*)\s+(?:rdfs:)?subClassOf", re.IGNORECASE
)
_TTL_SUBCLASS = re.compile(
    r"(?:rdfs:)?subClassOf\s+(?:<[^>]*[/#])?(?:[\w.-]*:)?([A-Za-z_][\w.-]*)",
    re.IGNORECASE,
)


def _add_prefixed_turtle(g: Any, text: str) -> None:
    subject = None
    for line in text.split("\n"):
        declared = _TTL_CLASS.match(line)
        if declared:
            subject = declared.group(1)
            if subject.lower() not in _OWL_BUILTINS:
                g.add_node(subject)
        else:
            implied = _TTL_SUBJECT.match(line)
            if implied:
                subject = implied.group(1)
        for found in _TTL_SUBCLASS.finditer(line):
            parent = found.group(1)
            if (subject and parent.lower() not in _OWL_BUILTINS
                    and subject.lower() not in _OWL_BUILTINS):
                g.add_edge(subject, parent, rel="subClassOf")
        if line.rstrip().endswith("."):
            subject = None


_BLOCK_HEADER = re.compile(r"^\s*(SubClassOf|EquivalentTo|Equivalent|Disjoint\w*)\s*:\s*$",
                           re.IGNORECASE)
_BLOCK_PAIR = re.compile(
    r"^\s+<?(?:[\w.-]*:)?([A-Za-z_][\w.-]*)>?\s+<?(?:[\w.-]*:)?([A-Za-z_][\w.-]*)>?\s*$"
)


def _add_block_subclass(g: Any, text: str) -> None:
    in_subclass = False
    for line in text.split("\n"):
        header = _BLOCK_HEADER.match(line)
        if header:
            in_subclass = header.group(1).lower() == "subclassof"
            continue
        if not line.strip():
            in_subclass = False
            continue
        if not in_subclass:
            continue
        pair = _BLOCK_PAIR.match(line)
        if not pair:
            continue
        child, parent = pair.group(1), pair.group(2)
        if (child.lower() not in _OWL_BUILTINS
                and parent.lower() not in _OWL_BUILTINS):
            g.add_edge(child, parent, rel="subClassOf")


def parse_manchester_to_graph(manchester_text: str) -> Any | None:
    if not HAS_NX:
        return None

    g = nx.DiGraph()
    _add_block_subclass(g, manchester_text)

    parsed = _rdf_graph(manchester_text)
    if parsed is not None:
        g.add_nodes_from(parsed.nodes())
        g.add_edges_from(parsed.edges(data=True))

    _add_prefixed_turtle(g, manchester_text)

    subclass_re = re.compile(
        r"(\w[\w\s]*?)\s+SubClassOf[:\s]+(\w[\w\s]*?)\s*[.;]?\s*(?:\n|$)",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in subclass_re.finditer(manchester_text):
        child = m.group(1).strip()
        parent = m.group(2).strip()
        g.add_edge(child, parent, rel="subClassOf")

    turtle_class_re = re.compile(r"^\s*(\w+)\s+a\s+owl:Class", re.MULTILINE)
    for m in turtle_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

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

    sub_keyword_re = re.compile(r"(\w[\w]*)\s+sub\s+(\w[\w]*)\s*[;,]?", re.MULTILINE)
    for m in sub_keyword_re.finditer(manchester_text):
        child = m.group(1).strip()
        parent = m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")

    dl_subclass_re = re.compile(r"(\w[\w]*)\s+[⊃⊂]\s+(\w[\w]*)", re.MULTILINE)
    for m in dl_subclass_re.finditer(manchester_text):
        child, parent = m.group(1).strip(), m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")

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

    md_class_re = re.compile(r"\*\*(\w[\w]*):\*\*")
    for m in md_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

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

    uri_a_class_re = re.compile(
        r"<[^>]*?(\w[\w]*)>\s+a\s+(?:owl|rdfs):Class\s*[.;,]?", re.MULTILINE
    )
    uri_class_positions: dict[int, str] = {}
    for m in uri_a_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            uri_class_positions[m.start()] = cls

    prefix_class_re = re.compile(
        r"(?:\w+:)(\w[\w]*)\s+owl:Class\s*[.;]?", re.MULTILINE
    )
    for m in prefix_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            class_positions[m.start()] = cls

    rdf_type_class_re = re.compile(
        r"^(\w[\w]*)\s+rdf:type\s+owl:Class\s*[.;]?", re.MULTILINE
    )
    for m in rdf_type_class_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)
            class_positions[m.start()] = cls

    all_class_positions = {**class_positions, **uri_class_positions}

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

    arrow_subclass_re = re.compile(
        r"(?:\w+:)?(\w[\w]*)\s*\|->\s*(?:\w+:)?(\w[\w]*)", re.MULTILINE
    )
    for m in arrow_subclass_re.finditer(manchester_text):
        child, parent = m.group(1).strip(), m.group(2).strip()
        if child.lower() not in _OWL_BUILTINS and parent.lower() not in _OWL_BUILTINS:
            g.add_edge(child, parent, rel="subClassOf")

    rdfs_label_re = re.compile(r'rdfs:label\s+"([^"]+)"', re.MULTILINE)
    for m in rdfs_label_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS and len(cls) > 1:
            g.add_node(cls)

    concept_uri_re = re.compile(
        r"(?:Concept|Class):\s*<[^>]*?(\w[\w]*)>", re.MULTILINE | re.IGNORECASE
    )
    for m in concept_uri_re.finditer(manchester_text):
        cls = m.group(1).strip()
        if cls.lower() not in _OWL_BUILTINS:
            g.add_node(cls)

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
    edge_cov = (round(len(gt_edge_set & llm_edge_set) / len(gt_edge_set), 4)
                if gt_edge_set else float("nan"))

    return {
        "concept_coverage": concept_cov,
        "edge_coverage": edge_cov,
        "n_gt_concepts": len(gt_set),
        "n_llm_concepts": len(llm_set),
        "n_gt_edges": len(gt_edge_set),
        "n_llm_edges": len(llm_edge_set),
    }
