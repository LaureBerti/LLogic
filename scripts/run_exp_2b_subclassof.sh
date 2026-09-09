#!/usr/bin/env bash
# run_exp_2b_subclassof.sh — Phase 2b: subClassOf transitivity probing
# Contributions: C4 (named object-property coverage is thin; subClassOf gives 50 triples per cell)
#
# Strategy:
#   For each sampled concept C, we enumerate (C⊑B, B⊑A) triples from the FULL
#   ontology and ask the LLM whether C⊑A holds. A "NO" is a transitivity violation.
#
# Ontology notes:
#   - book    : flat hierarchy (all classes → owl:Thing). Expect 0 triples → skipped.
#   - anatomy : rich OWL hierarchy. 50 triples sampled from ~64 candidates.
#   - cso     : uses cso:superTopicOf as parent relation (rdflib).
#   - agrovoc : SKOS skos:broader used as parent. Slow to load (~120s).
#   - gemet   : SKOS skos:broader used as parent.
#
# Usage:
#   bash outputs/paper_ready/scripts/run_exp_2b_subclassof.sh
#
# Background launch (resume-safe):
#   nohup bash outputs/paper_ready/scripts/run_exp_2b_subclassof.sh \
#       > logs/p2b_run.log 2>&1 &
#   tail -f logs/p2b_run.log

set -euo pipefail
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

mkdir -p logs

LOCAL_API="http://localhost:11434/v1"
MODEL="llama3.2:latest"
MODEL_SAFE="${MODEL//:/_}"

# All 4 ontologies (book will self-skip if no triples are found).
ONTOLOGIES="book anatomy agrovoc cso"

# Full strategy matrix — subClassOf probe is strategy-independent in principle,
# but we run all 4 to match Phase 2 and capture any prompting-strategy effects.
STRATEGIES="zero_shot cot tot self_consistency"

echo "[exp2b] $(date '+%Y-%m-%d %H:%M %Z') Starting Phase 2b (subClassOf transitivity)"
echo "[exp2b] Model: $MODEL | Ontologies: $ONTOLOGIES"

for ONTO in $ONTOLOGIES; do
    for STRATEGY in $STRATEGIES; do
        LOG="logs/p2b_${MODEL_SAFE}_${ONTO}_${STRATEGY}.log"
        echo "  $(date '+%H:%M') $MODEL | $ONTO | $STRATEGY → $LOG"
        python src/phase2b_subclassof.py \
            --ontology "$ONTO" \
            --model "$MODEL" \
            --strategy "$STRATEGY" \
            --seed 42 \
            --api_base "$LOCAL_API" \
            2>&1 | tee "$LOG" || {
            echo "  ERROR: $MODEL | $ONTO | $STRATEGY failed — continuing" >&2
        }
    done
    echo "  ↳ $ONTO done"
done

echo "[exp2b] $(date '+%Y-%m-%d %H:%M %Z') Phase 2b complete for $MODEL."
echo "[exp2b] To add more LLMs, repeat with MODEL=gemma2:2b / mistral:7b / qwen2.5:7b"
