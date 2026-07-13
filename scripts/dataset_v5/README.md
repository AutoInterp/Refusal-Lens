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

## Local CPU checks (no GPU)
```bash
.venv/bin/python -m pytest scripts/dataset_v5/tests/ -v
.venv/bin/python scripts/dataset_v5/build_dataset_v5.py --limit 3 --out /tmp/ds5_smoke.json  # placeholder many-shot
```
