#!/usr/bin/env bash
# dataset_v5 RunPod runbook: Phase A = end-to-end smoke (3 bases, ALL classes) with a
# human inspect gate; Phase B = full comprehensive run (all 50, no limit).
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN first}"
# Reduce CUDA fragmentation on the long (~35k-token) many_shot_icl generations.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$(dirname "$0")"
OUT=../../new_dataset_results/refusal_results
PY=${PY:-python}                       # RunPod: plain python (CUDA torch); dev box: unused

echo "== install =="
# nanoGCG 0.3.0 pins transformers<=4.47.1, but Gemma-3 needs >=4.50. ANY pip command that
# resolves nanoGCG's deps will downgrade transformers to 4.47.1 -> Gemma-3 KeyError. So we
# install nanoGCG's real runtime deps explicitly, install nanoGCG itself with --no-deps
# (its stale transformers pin is ignored), then pin transformers LAST. This is safe because
# gcg_optimize.py sets use_prefix_cache=False (the only nanoGCG path that breaks on new
# transformers). torch omitted: keep the pod's CUDA build.
pip install -q accelerate litellm protobuf scipy sentencepiece
pip install -q --no-deps nanogcg
pip install -q "transformers==4.57.3"

echo "== Phase A: smoke gate =="
$PY gcg_optimize.py --smoke
$PY gcg_optimize.py --mode per_prompt --limit 3 --out "$OUT/gcg_suffixes_smoke.json"
$PY build_dataset_v5.py --gcg-suffixes "$OUT/gcg_suffixes_smoke.json" --limit 3 \
    --out ../../dataset_v5_smoke.json
$PY generate.py --dataset ../../dataset_v5_smoke.json --out "$OUT/v5_smoke_generations.json"
$PY report_v5.py --generations "$OUT/v5_smoke_generations.json" --inspect
echo ">>> INSPECT the blocks above: both classes must show a well-formed attack AND a"
echo ">>> plausible response. Re-run with RUN_FULL=1 to launch the full run."
[ "${RUN_FULL:-0}" = "1" ] || { echo "stopping after smoke (set RUN_FULL=1 to continue)"; exit 0; }

echo "== Phase B: full run (all 50, no limit) =="
$PY gcg_optimize.py --mode per_prompt --out "$OUT/gcg_suffixes.json"
$PY build_dataset_v5.py --gcg-suffixes "$OUT/gcg_suffixes.json" --sweep --out ../../dataset_v5.json
$PY generate.py --dataset ../../dataset_v5.json --out "$OUT/v5_generations.json"
$PY generate.py --dataset "$OUT/many_shot_sweep.json" --out "$OUT/v5_sweep_generations.json"
echo ">>> now judge $OUT/v5_generations.json per README (Ollama/litellm) -> v5_judged.json"
echo ">>> then: $PY report_v5.py --judged $OUT/v5_judged.json"
echo ">>> ALSO judge $OUT/v5_sweep_generations.json the same way -> v5_sweep_judged.json"
echo ">>> then: $PY report_v5.py --sweep-judged $OUT/v5_sweep_judged.json"
