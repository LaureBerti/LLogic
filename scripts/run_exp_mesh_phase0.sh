#!/usr/bin/env bash
# run_exp_mesh_phase0.sh — Phase 0: sample 50 concepts + relations from MeSH
#
# Prerequisites:
#   1. Place mesh.ttl in data/ontologies/mesh.ttl  (NLM Turtle release)
#   2. Activate venv: source .venv/bin/activate
#
# WARNING: rdflib parsing of mesh.ttl takes 5–15 min depending on file size.
# Run once; output cached in data/samples/concepts_mesh_seed42.json
#
# Usage: bash outputs/paper_ready/scripts/run_exp_mesh_phase0.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

TTL="data/ontologies/mesh.ttl"
if [ ! -f "$TTL" ]; then
    echo "ERROR: $TTL not found."
    echo "Copy your MeSH Turtle file: cp /path/to/mesh.ttl $ROOT/data/ontologies/mesh.ttl"
    exit 1
fi

echo "[mesh phase0] $(date '+%Y-%m-%d %H:%M %Z') Sampling MeSH concepts and relations…"
python src/main.py \
    phase=0 \
    ontology.name=mesh \
    sampling.seed=42

echo "[mesh phase0] $(date '+%Y-%m-%d %H:%M %Z') Done."
echo "  Samples written to data/samples/concepts_mesh_seed42.json"
echo "  Next: run run_exp_mesh_validate.sh to confirm the pipeline works."
