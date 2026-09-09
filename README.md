# LLOGIC — LLM Logical Ontology-grounded Inconsistency Checker

Reproducibility code for the paper:

> **Measuring Logical Inconsistency of Large Language Models over OWL Ontologies**
> Laure Berti-Équille (IRD). *Knowledge-Based Systems* (Elsevier), 2026.

LLOGIC is a three-phase empirical framework that compiles LLM-generated first-order-logic
(FOL) statements into a Z3 SMT solver to measure whether large language models stay
logically consistent when defining and relating concepts in formal OWL/SKOS ontologies.

**Headline finding:** *LLM inconsistency is relational, not definitional.* Across 2,574
compiled FOL concept definitions the solver finds one contradiction (95% one-sided upper
bound 0.116%), yet symmetry-violation rates reach **100%** and `subClassOf`
transitivity-violation rates span **15–92%**.

**Two qualifications the numbers require, both measured rather than assumed:**

* The contradiction bound covers responses that *compile*. **31.2% do not**, and their
  content is characterised rather than treated as benign.
* Passing the satisfiability check does not mean a definition is usable. Asserting
  filter-accepted definitions into ontologies that declare disjointness produces **0
  unsatisfiable of 138**; asserting the same definitions into the same ontology enriched
  with 170,144 published disjointness axioms makes **43.1% unsatisfiable**. A pass is a
  property of how constrained the target ontology is. And of 64 accepted definitions
  audited by hand, **none was adequate**.

Prompting does not repair relational failure. Chain-of-thought lowers the `subClassOf`
transitivity violation rate by 14.4 pp against zero-shot (p = 0.047), leaving 21.4%.
Across five independent concept samples no prompting strategy reliably improves format
compliance; the one contrast that replicates is model-specific.

---

## The LLM calls

`data/llm_calls/` holds **every call made to a language model in this study** — 24,762 rows
across five stratified concept samples, each giving the concept, the model, the prompting
strategy, and the model's reply verbatim. That is the part of the record that cannot be
regenerated deterministically, so it is versioned here.

```
data/llm_calls/llm_calls_seed{42,123,456,789,1011}.csv
data/llm_calls/README.md        columns, provenance, and what is incomplete
data/llm_calls/MANIFEST.json    per-file counts
```

Derived results — solver verdicts, extracted formulae, timings, per-cell rates — are not
versioned: `outputs/` is created empty and populated by running the pipeline or the
analyses. The complete result tree is archived on Zenodo: **doi:10.5281/zenodo.22643175**.

`ONTOLOGY_VERSIONS.json` gives the SHA-256, size and source URL of each ontology, which
are excluded from git for size (AGROVOC alone is 1.2 GB).

## Reproducing

Analyses over existing results — no LLM calls, no network, seconds each:

```bash
python src/analysis/inventory.py        # what result cells exist, computed from the files
python src/analysis/c1_recompute.py     # concept-level figures over a stated grid
python src/analysis/stats_tests.py      # blocked tests, Holm-corrected
python src/analysis/reasoner_validation.py reasoner.arm=arm_a
```

Generating results needs a local Ollama serving the models named in `conf/jamer.yaml`;
the pipeline talks to it through an OpenAI-compatible endpoint, so any compatible server
works.

```bash
python src/main.py phase=0 ontology.name=book sampling.seed=42
python src/main.py phase=1 ontology.name=book llm.model=mistral:7b prompting.strategy=cot
```

Every phase is resume-safe: rerunning a cell fills only the rows that are missing, keyed on
`(concept_iri, strategy)`, and never rewrites a row that already exists.

## Known limits

* Phase 2b needs a testable `subClassOf` hierarchy. `book` has none — 36 classes, no parent
  relations — so it cannot be probed for transitivity.
* The FOL→OWL translator used for reasoner validation accepts only conjunctions of named
  classes, 6.4% of accepted definitions. Refusal causes are counted per category rather
  than folded into a denominator.
* AGROVOC publishes SKOS-XL labels rather than plain `skos:prefLabel`; the loader handles
  both, and a cache-key version guards against a stale parent map surviving a loader change.

---

## The three phases

| Phase | What it measures | Script entry |
|-------|------------------|--------------|
| **0** | Stratified sampling of 50 concepts + relations per ontology | `phase=0` |
| **1** | Concept-level: LLM defines concepts in FOL → Z3 checks satisfiability | `phase=1` |
| **2** | Relational: probes symmetry / transitivity / domain-range against OWL axioms | `phase=2` |
| **2b**| `subClassOf` IS-A transitivity over the ontology hierarchy | `phase=2 phase_2b=true` |
| **3** | Ontology reconstruction from conversation context; concept-recall metric | `phase=3` |

---

## Quick start

### 1. Environment (Python 3.13 tested)

```bash
git clone https://github.com/LaureBerti/LLogic.git
cd LLogic
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `sentence-transformers` (optional, for Phase-3 semantic similarity) needs `torch` and
> Python ≤ 3.12 on macOS. Without it, the semantic metric returns `NaN` and the rest of
> the pipeline runs normally.

### 2. Local LLM backend (Ollama)

All experiments run locally — **no cloud API keys required**.

```bash
# install Ollama: https://ollama.ai
ollama serve &
ollama pull llama3.2:latest   # Meta, 3B
ollama pull gemma2:2b         # Google, 2B
ollama pull mistral:7b        # Mistral AI, 7B
ollama pull qwen2.5:7b        # Alibaba, 7B
```

### 3. Reproduce one cell (smoke test, ~1 min)

The repo ships the **pre-computed concept samples** (`data/samples/`), so you can run
Phase 1 immediately without downloading the raw ontologies:

```bash
python src/main.py phase=1 ontology.name=book \
    llm.model=llama3.2:latest prompting.strategy=zero_shot
```

This writes `outputs/results/book/llama3.2_latest/zero_shot/phase1_results.csv`
(contradiction rate, parse rate, per-concept FOL + solver verdict).

---

## Reproducing the full paper matrix

The paper evaluates **4 models × 5 ontologies × 4 strategies × 3 phases = 240 conditions**.

### Phase 1 (concept consistency) — all cells

```bash
for ONTO in book anatomy agrovoc cso mesh; do
  for MODEL in llama3.2:latest gemma2:2b mistral:7b qwen2.5:7b; do
    for STRAT in zero_shot cot tot self_consistency; do
      python src/main.py phase=1 ontology.name=$ONTO \
          llm.model=$MODEL prompting.strategy=$STRAT
    done
  done
done
```

### Phase 2 + 2b (relational consistency) — ZS + CoT

```bash
for ONTO in book anatomy agrovoc cso mesh; do
  for MODEL in llama3.2:latest gemma2:2b mistral:7b qwen2.5:7b; do
    python src/main.py phase=2 ontology.name=$ONTO llm.model=$MODEL prompting.strategy=zero_shot
    python src/main.py phase=2 ontology.name=$ONTO llm.model=$MODEL prompting.strategy=cot
    python src/main.py phase=2 phase_2b=true ontology.name=$ONTO llm.model=$MODEL prompting.strategy=zero_shot
  done
done
```

### Phase 3 (reconstruction)

```bash
for ONTO in book anatomy agrovoc cso mesh; do
  for MODEL in llama3.2:latest gemma2:2b mistral:7b qwen2.5:7b; do
    python src/main.py phase=3 ontology.name=$ONTO llm.model=$MODEL prompting.strategy=zero_shot
    python src/main.py phase=3 ontology.name=$ONTO llm.model=$MODEL prompting.strategy=cot
  done
done
```

**All runs are resume-safe** — re-running skips conditions already recorded in the output
CSV (resume key: `(concept_iri, strategy)` for Phase 1; `relation_label` for Phase 2).

Ready-made batch scripts are in `scripts/` (e.g. `run_exp_mesh_all_vm.sh` runs the full
MeSH ontology end-to-end with pre-flight checks, resume logic, and a `caffeinate`-friendly
layout for long unattended runs).

### Timing

On an Apple-silicon Mac, one Phase-1 cell (50 concepts) is ~7 min for zero-shot/CoT/ToT
and ~3× longer for self-consistency. The full 240-condition matrix is ~1 day of compute.
CPU-only machines are ~8× slower (no GPU) — use a GPU host for the full run.

---

## Configuration (Hydra)

All parameters live in `conf/jamer.yaml` and are overridable on the command line
(dot-notation, no `--`):

```bash
python src/main.py phase=1 ontology.name=anatomy \
    llm.model=mistral:7b llm.temperature=0.0 \
    prompting.strategy=cot solver.timeout_s=10 sampling.seed=42

python src/main.py --cfg job            # print the resolved config without running
```

Key fields: `phase`, `phase_2b`, `llm.{model,temperature,timeout_s,n_samples}`,
`ontology.{name,path}`, `sampling.{n_concepts,n_relations,seed}`,
`prompting.strategy ∈ {zero_shot, cot, tot, self_consistency}`.

---

## Repository layout

```
src/
  main.py                  Hydra entry point (dispatches by phase)
  ontology_loader.py       OWL/RDF/SKOS/MeSH loading + stratified sampling (Phase 0)
  phase1_concepts.py       FOL elicitation + Z3 satisfiability check
  phase2_edges.py          Relational property probing (symmetry/transitivity/domain-range)
  phase2b_subclassof.py    subClassOf IS-A transitivity probing
  phase3_reconstruct.py    Ontology reconstruction + concept-recall metric
  parsing/fol_parser.py    FOL → Z3 compiler (unary/binary preds, connectives, quantifiers)
  prompting/               LLM client (OpenAI-compatible / Ollama) + prompt templates
  metrics/                 consistency + similarity metrics
conf/jamer.yaml            all parameters (Hydra)
  analysis/                post-hoc analyses over generated results
conf/analysis.yaml         parameters for the analyses
data/samples/              pre-computed stratified samples (5 seeds) — enables exact reproduction
data/llm_calls/            every LLM call: concept, model, strategy, reply verbatim
data/ontologies/           download instructions (raw files excluded — see its README)
ONTOLOGY_VERSIONS.json     SHA-256, size and source URL for each ontology
scripts/                   batch experiment runners (resume-safe)
outputs/                   created empty; per-condition CSVs are written here by a run
```

---

## Citation

```bibtex
@article{bertiequille2026llogic,
  title   = {Measuring Logical Inconsistency of Large Language Models over OWL Ontologies},
  author  = {Berti-{\'E}quille, Laure},
  journal = {Knowledge-Based Systems (under review)},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). All five evaluated ontologies are publicly available under
their own open licenses (see `data/ontologies/README.md`).
