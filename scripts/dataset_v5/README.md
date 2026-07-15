# dataset_v5 — GCG + Many-Shot jailbreak classes

Adds `gcg_per_prompt`, `many_shot_icl` on the 50 shared base prompts,
evaluated with Tejas's harness. Spec:
`docs/superpowers/specs/2026-07-13-gcg-manyshot-jailbreak-classes-design.md`.

## Provenance
- `generate.py` is a clean copy of `new_dataset_results/refusal_results/generate_v4 (1).py`
  (Tejas), logic unchanged. Judge = in-repo `refusal_llm_judge`.
- Many-shot demos reuse Gemma's own COMPLY outputs from `v3_judged.json`+`v4_judged.json`.

## Run (RunPod GPU)
```bash
export HF_TOKEN=...
bash run_v5_runpod.sh            # Phase A smoke + inspect gate; stops
RUN_FULL=1 bash run_v5_runpod.sh # after inspecting, full run
```
Phase B also produces `v5_sweep_generations.json` (many-shot K-sweep); judge it the
same way as the main file to get `v5_sweep_judged.json`, then run
`report_v5.py --sweep-judged v5_sweep_judged.json` for the K -> comply-rate curve.

## Judge (Ollama local or any litellm API), from the repo root
```bash
ollama serve & ; ollama pull qwen3.6:35b-a3b
.venv/bin/python -c "
import json; from refusal_llm_judge import judge_refusal
d=json.load(open('new_dataset_results/refusal_results/v5_generations.json'))
atk=[g for g in d['generations'] if g['kind']=='attack']
labels=judge_refusal([{'prompt':g['prompt_text'],'response':g['response']} for g in atk])
for g,v in zip(atk,labels): g['judge']='REFUSE' if v else 'COMPLY'
json.dump(d,open('new_dataset_results/refusal_results/v5_judged.json','w'),indent=2)
"
.venv/bin/python scripts/dataset_v5/report_v5.py --judged new_dataset_results/refusal_results/v5_judged.json
```

## Faithful many-shot re-run (valid demos)

The original `many_shot_icl` (0%) sourced demos from Gemma's v3/v4 COMPLY outputs, which
are mostly soft fiction-frame deflections — demonstrating deflection, not compliance, so
0% is not a faithful MSJ test (a strawman like GCG's token-mismatch 4%). Re-source demos
from the genuinely-harmful `refusal_suppression` COMPLY answers (98% run) instead, keeping
the RAW question so each demo is a clean (harmful Q -> harmful A) pair:

```bash
# build MSJ-only dataset (+K-sweep) from the refusal_suppression demo pool; distinct names
.venv/bin/python scripts/dataset_v5/build_dataset_v5.py --ms-only \
  --ms-pool-v5 new_dataset_results/refusal_results/v5_judged.json \
  --sweep --sweep-out new_dataset_results/refusal_results/many_shot_sweep_msfaithful.json \
  --out new_dataset_results/refusal_results/dataset_v5_msfaithful.json
# generate (GPU, bf16; ~35k-tok prompts -> expandable_segments)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/dataset_v5/generate.py \
  --dataset new_dataset_results/refusal_results/dataset_v5_msfaithful.json \
  --out new_dataset_results/refusal_results/v5_msfaithful_generations.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/dataset_v5/generate.py \
  --dataset new_dataset_results/refusal_results/many_shot_sweep_msfaithful.json \
  --out new_dataset_results/refusal_results/v5_msfaithful_sweep_generations.json
# judge each with the same Ollama snippet below (swap the in/out filenames), then:
.venv/bin/python scripts/dataset_v5/report_v5.py \
  --judged new_dataset_results/refusal_results/v5_msfaithful_judged.json \
  --sweep-judged new_dataset_results/refusal_results/v5_msfaithful_sweep_judged.json
# -> writes v5_msfaithful_report.md (report name derives from --judged; won't touch v5_report.md)
```
Interpretation: still ~0% with valid demos = robust true negative (Gemma resists MSJ); a
rising K-curve / non-trivial rate = MSJ lands and becomes a 3rd traceable class.

## Local CPU checks (no GPU)
```bash
.venv/bin/python -m pytest scripts/dataset_v5/tests/ -v
.venv/bin/python scripts/dataset_v5/build_dataset_v5.py --limit 3 --out /tmp/ds5_smoke.json  # placeholder many-shot
```
