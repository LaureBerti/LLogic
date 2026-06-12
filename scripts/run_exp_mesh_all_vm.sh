#!/usr/bin/env bash
# =============================================================================
# run_exp_mesh_all_vm.sh — Full MeSH experiment on a single VM
# =============================================================================
# Runs Phase 0 → Phase 1 (all 4 models × 4 strategies) →
#         Phase 2 (all models, ZS+CoT) →
#         Phase 3 (all models, ZS+CoT)
#
# Resume-safe: skips already-completed cells at every phase.
# Expects a single Ollama instance (default port 11434).
# Models are run sequentially to avoid memory conflicts.
#
# Usage:
#   nohup bash outputs/paper_ready/scripts/run_exp_mesh_all_vm.sh \
#       > logs/mesh_all.log 2>&1 & disown
#
# Prerequisites on VM:
#   1. data/ontologies/mesh.ttl must be present (~803 MB)
#   2. Ollama running: ollama serve &
#   3. Models pulled:
#        ollama pull llama3.2:latest
#        ollama pull gemma2:2b
#        ollama pull mistral:7b
#        ollama pull qwen2.5:7b
#   4. Project synced (including data/samples/ if Phase 0 was run locally)
# =============================================================================

set -uo pipefail
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true
mkdir -p logs

MODELS="llama3.2:latest gemma2:2b mistral:7b qwen2.5:7b"
STRATEGIES_P1="zero_shot cot tot self_consistency"
STRATEGIES_P23="zero_shot cot"
OLLAMA_URL="http://localhost:11434"
ONTO="mesh"

# ── Helper: timestamp ─────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "============================================================"
echo " JAMER MeSH full experiment — $(ts)"
echo "============================================================"

# ── Pre-flight: Ollama ─────────────────────────────────────────────────────────
echo ""
echo "[preflight] Checking Ollama at $OLLAMA_URL …"
if ! curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running."
    echo "  Start it: ollama serve &"
    exit 1
fi
echo "[preflight] Ollama ✅"

# ── Pre-flight: models ─────────────────────────────────────────────────────────
echo "[preflight] Checking models …"
MISSING=""
for MODEL in $MODELS; do
    if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$(echo $MODEL | cut -d: -f1)\""; then
        MISSING="$MISSING $MODEL"
    fi
done
if [ -n "$MISSING" ]; then
    echo "WARNING: These models may not be pulled:$MISSING"
    echo "  Pull them with: ollama pull <model>"
    echo "  Continuing anyway (Ollama will pull on first use)."
fi
echo "[preflight] Models ✅"

# ── Pre-flight: mesh.ttl ───────────────────────────────────────────────────────
TTL="data/ontologies/mesh.ttl"
if [ ! -f "$TTL" ]; then
    echo "ERROR: $TTL not found."
    echo "  Copy your MeSH Turtle file to: $ROOT/data/ontologies/mesh.ttl"
    exit 1
fi
echo "[preflight] mesh.ttl found ($(du -sh $TTL | cut -f1)) ✅"

# ── Phase 0: sample concepts ───────────────────────────────────────────────────
SAMPLE="data/samples/concepts_${ONTO}_seed42.json"
if [ -f "$SAMPLE" ]; then
    N=$(python3 -c "import json; print(len(json.load(open('$SAMPLE'))))" 2>/dev/null || echo "?")
    echo ""
    echo "[phase0] $SAMPLE already exists ($N concepts) — skipping Phase 0."
else
    echo ""
    echo "[phase0] $(ts) Sampling MeSH concepts (takes ~10 min) …"
    python src/main.py phase=0 ontology.name=$ONTO sampling.seed=42 \
        2>&1 | tee logs/mesh_phase0.log
    echo "[phase0] $(ts) Done."
fi

# ── Phase 1: FOL + Z3 per model × strategy ────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 1 — FOL concept definitions ($(ts))"
echo "============================================================"

for MODEL in $MODELS; do
    MODEL_SLUG=$(echo "$MODEL" | tr ':.' '_')
    for STRATEGY in $STRATEGIES_P1; do
        CSV="outputs/results/${ONTO}/${MODEL_SLUG}/${STRATEGY}/phase1_results.csv"
        if [ -f "$CSV" ]; then
            # Count UNIQUE concept_iri (not physical lines — FOL blocks are multiline)
            UNIQ=$(python3 -c "import csv;import sys;
rows=list(csv.DictReader(open('$CSV')));
print(len(set(r.get('concept_iri','') for r in rows)))" 2>/dev/null || echo 0)
            if [ "$UNIQ" -ge 50 ]; then
                echo "[p1] SKIP  $MODEL | $ONTO | $STRATEGY ($UNIQ concepts already done)"
                continue
            fi
        fi
        LOG="logs/p1_${ONTO}_${MODEL_SLUG}_${STRATEGY}.log"
        echo "[p1] START $MODEL | $ONTO | $STRATEGY — $(ts)"
        python src/main.py \
            phase=1 \
            ontology.name="$ONTO" \
            llm.model="$MODEL" \
            llm.temperature=0.0 \
            prompting.strategy="$STRATEGY" \
            2>&1 | tee "$LOG"
        ROWS=$(tail -n +2 "$CSV" 2>/dev/null | wc -l | tr -d ' ')
        echo "[p1] DONE  $MODEL | $ONTO | $STRATEGY — $ROWS rows — $(ts)"
    done
done

echo ""
echo "[phase1] All models complete — $(ts)"

# ── Phase 2: relational probing ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2 — Relational probing ($(ts))"
echo "============================================================"

for MODEL in $MODELS; do
    MODEL_SLUG=$(echo "$MODEL" | tr ':.' '_')
    for STRATEGY in $STRATEGIES_P23; do
        CSV="outputs/results/${ONTO}/${MODEL_SLUG}/${STRATEGY}/phase2_results.csv"
        if [ -f "$CSV" ]; then
            ROWS=$(tail -n +2 "$CSV" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$ROWS" -ge 1 ]; then
                echo "[p2] SKIP  $MODEL | $ONTO | $STRATEGY"
                continue
            fi
        fi
        LOG="logs/p2_${ONTO}_${MODEL_SLUG}_${STRATEGY}.log"
        echo "[p2] START $MODEL | $ONTO | $STRATEGY — $(ts)"
        python src/main.py \
            phase=2 \
            ontology.name="$ONTO" \
            llm.model="$MODEL" \
            llm.temperature=0.0 \
            prompting.strategy="$STRATEGY" \
            2>&1 | tee "$LOG"
        echo "[p2] DONE  $MODEL | $ONTO | $STRATEGY — $(ts)"
    done
done

# Phase 2b: subClassOf IS_A transitivity (zero_shot only — strategy-invariant)
echo ""
echo "[phase2b] Running IS_A transitivity probing ($(ts))"
for MODEL in $MODELS; do
    MODEL_SLUG=$(echo "$MODEL" | tr ':.' '_')
    CSV2B="outputs/results/${ONTO}/${MODEL_SLUG}/zero_shot/phase2b_results.csv"
    if [ -f "$CSV2B" ]; then
        echo "[p2b] SKIP  $MODEL | $ONTO"
        continue
    fi
    LOG="logs/p2b_${ONTO}_${MODEL_SLUG}.log"
    echo "[p2b] START $MODEL | $ONTO — $(ts)"
    python src/main.py \
        phase=2 \
        phase_2b=true \
        ontology.name="$ONTO" \
        llm.model="$MODEL" \
        llm.temperature=0.0 \
        prompting.strategy=zero_shot \
        2>&1 | tee "$LOG"
    echo "[p2b] DONE  $MODEL | $ONTO — $(ts)"
done

echo ""
echo "[phase2] All models complete — $(ts)"

# ── Phase 3: ontology reconstruction ──────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 3 — Ontology reconstruction ($(ts))"
echo "============================================================"

for MODEL in $MODELS; do
    MODEL_SLUG=$(echo "$MODEL" | tr ':.' '_')
    for STRATEGY in $STRATEGIES_P23; do
        CSV="outputs/results/${ONTO}/${MODEL_SLUG}/${STRATEGY}/phase3_results.csv"
        if [ -f "$CSV" ]; then
            ROWS=$(tail -n +2 "$CSV" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$ROWS" -ge 1 ]; then
                echo "[p3] SKIP  $MODEL | $ONTO | $STRATEGY"
                continue
            fi
        fi
        LOG="logs/p3_${ONTO}_${MODEL_SLUG}_${STRATEGY}.log"
        echo "[p3] START $MODEL | $ONTO | $STRATEGY — $(ts)"
        python src/main.py \
            phase=3 \
            ontology.name="$ONTO" \
            llm.model="$MODEL" \
            llm.temperature=0.0 \
            prompting.strategy="$STRATEGY" \
            phase3.use_phase1_context=true \
            2>&1 | tee "$LOG"
        echo "[p3] DONE  $MODEL | $ONTO | $STRATEGY — $(ts)"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " SUMMARY — $(ts)"
echo "============================================================"
echo ""
python3 - <<'EOF'
import csv
from pathlib import Path

BASE = Path("outputs/results/mesh")
MODELS = ["llama3.2_latest","gemma2_2b","mistral_7b","qwen2.5_7b"]
STRATS = ["zero_shot","cot","tot","self_consistency"]

print(f"{'Model':20s} {'Strategy':17s} {'P1 rows':8s} {'P2':4s} {'P3':4s}")
print("-"*60)
for m in MODELS:
    for s in STRATS:
        p1 = BASE/m/s/"phase1_results.csv"
        p2 = BASE/m/s/"phase2_results.csv"
        p3 = BASE/m/s/"phase3_results.csv"
        r1 = (len(open(p1).readlines())-1) if p1.exists() else 0
        r2 = "✓" if p2.exists() else "-"
        r3 = "✓" if p3.exists() else "-"
        print(f"  {m:18s} {s:17s} {r1:6d}   {r2:3s}  {r3:3s}")

print("")
print("All done. Sync results back:")
print("  rsync -av vm:/path/to/JAMER/outputs/results/mesh/ \\")
print("            /Users/laureberti/Projects/JAMER/outputs/results/mesh/")
EOF

echo ""
echo "============================================================"
echo " MeSH experiment complete — $(ts)"
echo "============================================================"
