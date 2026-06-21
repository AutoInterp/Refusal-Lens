#!/bin/bash
# Poll for the DONE marker; on completion commit the small artifacts to the
# branch (the HF push already happened inside the orchestrator).
set -u
cd "$(git rev-parse --show-toplevel)"
MARKER=data/results/pipeline_runs/.GEMMA_VARIANTS_DONE
echo "watching for $MARKER (5 min poll, 8h timeout)…"
for i in $(seq 1 96); do
  if [ -f "$MARKER" ]; then
    echo "DONE marker seen $(date); committing artifacts."
    git add -A data/results/pipeline_runs/gemma_var_*/02_attribution/attribution_results.json \
                data/results/pipeline_runs/gemma_var_*/05_frontend/data/graph-metadata.json 2>/dev/null || true
    git commit -m "gemma variants: attribution summaries + packed metadata" || echo "(nothing to commit)"
    git push origin HEAD || echo "(push failed; push manually)"
    exit 0
  fi
  sleep 300
done
echo "timeout waiting for DONE marker"; exit 1
