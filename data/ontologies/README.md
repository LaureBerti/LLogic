# Ontology files

The five ontologies are **not** included in the repository (some exceed GitHub's file-size
limits — AGROVOC is 1.2 GB, MeSH 803 MB). Download them into this directory before running
Phase 0. The pre-computed stratified samples in `../samples/` already cover the experiments
in the paper, so you only need the raw ontologies if you want to **re-sample** (Phase 0).

| File (place here)       | Domain        | Source |
|-------------------------|---------------|--------|
| `book.rdf`              | Bibliographic | OAEI 2007 benchmark — <http://oaei.ontologymatching.org/2007/benchmarks/> |
| `anatomy.owl`           | Biomedical    | OAEI Anatomy / NCI Thesaurus — <http://oaei.ontologymatching.org/2007/anatomy/> |
| `agrovoc_core.rdf`      | Food/Agric.   | FAO AGROVOC (SKOS-XL release) — <https://www.fao.org/agrovoc/releases> |
| `cso.owl`               | Computer Sci. | Computer Science Ontology — <https://cso.kmi.open.ac.uk/downloads> (OWL format) |
| `mesh.ttl`              | Biomedical    | MeSH BioPortal/UMLS2RDF Turtle — <https://bioportal.bioontology.org/ontologies/MESH> |

Notes:
- **AGROVOC** must be the full SKOS-XL version (labels in `skosxl:prefLabel` / `literalForm`).
- **MeSH** is the BioPortal UMLS2RDF Turtle release; the loader reads `owl:Class` +
  `rdfs:subClassOf` over `D`-prefixed descriptors.
- **GEMET** is intentionally excluded — its backbone file has no concept labels.

After downloading, regenerate samples with, e.g.:
```bash
python src/main.py phase=0 ontology.name=mesh sampling.seed=42
```
(MeSH parsing of the 803 MB Turtle takes ~10 minutes.)
