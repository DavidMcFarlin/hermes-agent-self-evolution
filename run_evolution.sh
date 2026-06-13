#!/bin/bash
# ============================================================
# Hermes Agent Self-Evolution v1.0 — Master Runner
# Coordinates triage and optimization across all 5 phases.
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Check for virtual environment
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# --help never starts a run; everything else requires the agent repo
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec $PYTHON -m evolution.monitor.loop --help
fi
: "${HERMES_AGENT_REPO:?HERMES_AGENT_REPO must be set (e.g. export HERMES_AGENT_REPO=~/.hermes/hermes-agent)}"

echo "=== 🧬 Hermes Self-Evolution v1.0 Master Loop ==="
echo "Started: $(date)"
echo ""

# Default to 1 iteration for automated runs unless overridden
export EVOLUTION_ITERATIONS="${EVOLUTION_ITERATIONS:-1}"

# Run the master orchestrator
$PYTHON -m evolution.monitor.loop "$@"

echo ""
echo "=== 🧬 Evolution Run Complete: $(date) ==="
