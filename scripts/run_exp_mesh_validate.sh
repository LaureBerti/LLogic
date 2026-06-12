#!/usr/bin/env bash
# run_exp_mesh_validate.sh — 1-cell validation for MeSH before full matrix
#
# Runs ONE cell: llama3.2:latest × mesh × zero_shot × Phase 1
# Expected output: 1 row in outputs/results/mesh/llama3.2_latest/zero_shot/phase1_results.csv
# Validates: Phase 0 samples → LLM call → Z3 → CSV write, end-to-end.
# Delete this cell's CSV before running the full matrix if it succeeds.
#
# Usage: bash outputs/paper_ready/scripts/run_exp_mesh_validate.sh

set -euo pipefail

# Ollama pre-flight
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Start it first: ollama serve &"
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

SAMPLE="data/samples/concepts_mesh_seed42.json"
if [ ! -f "$SAMPLE" ]; then
    echo "ERROR: $SAMPLE not found. Run run_exp_mesh_phase0.sh first."
    exit 1
fi

echo "[mesh validate] $(date '+%Y-%m-%d %H:%M %Z') Running 1-cell validation…"

python src/main.py \
    phase=1 \
    ontology.name=mesh \
    llm.model=llama3.2:latest \
    llm.temperature=0.0 \
    prompting.strategy=zero_shot \
    sampling.n_concepts=3 \
    2>&1

CSV="outputs/results/mesh/llama3.2_latest/zero_shot/phase1_results.csv"
if [ -f "$CSV" ]; then
    ROWS=$(tail -n +2 "$CSV" | wc -l | tr -d ' ')
    echo "[mesh validate] ✅ Validation passed — $ROWS row(s) in $CSV"
    echo "  Run the full matrix next: run_exp_mesh_phase1_*.sh"
else
    echo "[mesh validate] ❌ No CSV produced — check the error above."
    exit 1
fi
