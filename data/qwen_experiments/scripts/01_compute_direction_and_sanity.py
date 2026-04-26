"""
01: Compute Qwen3-4B refusal direction + sanity checks.
========================================================
This is the FIRST script to run for Qwen — it determines the empirically
best position/layer combination, which Gemma had hardcoded as (-2, 32).

Outputs:
  - results_v2/refusal_direction_v2.pt   (per-(pos, layer) directions + best)
  - results_v2/separation_table.json     (separation strength for every (pos, layer))
  - results_v2/sanity_check_v2.json      (means/stds for harmful, harmless, jailbroken)

After this finishes, fill in QWEN_BEST_POSITION / QWEN_BEST_LAYER in CONFIG.py.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import (  # noqa: E402
    HARMFUL_TRAIN,
    HARMLESS_TRAIN,
    MODEL_NAME,
    N_TRAIN_SAMPLES,
    PADDING_SIDE,
    RESULTS_V2_DIR,
    SWEEP_POSITIONS,
    format_prompt,
    get_hidden_size,
)

print("=" * 70)
print(f"01: Compute refusal direction for {MODEL_NAME}")
print("=" * 70)

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = PADDING_SIDE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

d_model = get_hidden_size(model)
n_layers = model.config.num_hidden_layers
print(f"  hidden_size={d_model}  num_layers={n_layers}")

# Load data
with open(HARMFUL_TRAIN) as f:
    harmful = [p["instruction"] for p in json.load(f)[:N_TRAIN_SAMPLES]]
with open(HARMLESS_TRAIN) as f:
    harmless = [p["instruction"] for p in json.load(f)[:N_TRAIN_SAMPLES]]


def collect_means(prompts):
    """For each (position, layer), accumulate float64 mean activation."""
    means = {
        (pos, layer): torch.zeros(d_model, dtype=torch.float64)
        for pos in SWEEP_POSITIONS
        for layer in range(n_layers)
    }
    for instr in prompts:
        formatted = format_prompt(instr, tokenizer)
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        for pos in SWEEP_POSITIONS:
            for layer in range(n_layers):
                hidden = out.hidden_states[layer + 1][0, pos, :].cpu().to(torch.float64)
                means[(pos, layer)] += hidden / len(prompts)
        del out
        gc.collect()
        torch.cuda.empty_cache()
    return means


print("\nCollecting harmful means...")
mean_harmful = collect_means(harmful)
print("Collecting harmless means...")
mean_harmless = collect_means(harmless)

# Compute directions and separation = ||r||
directions = {}
separation = {}
for key in mean_harmful:
    r = mean_harmful[key] - mean_harmless[key]
    directions[key] = r.to(torch.float32)
    separation[key] = float(r.norm().item())

# Find best
best_key = max(separation, key=separation.get)
best_pos, best_layer = best_key
best_direction = directions[best_key]
print(f"\nBEST: position={best_pos}, layer={best_layer}, |r|={separation[best_key]:.1f}")

# Save direction file (matches Gemma's structure)
to_save = {"best_direction": best_direction, "best_layer": best_layer, "best_position": best_pos}
for (pos, layer), d in directions.items():
    to_save[f"direction_pos{pos}_layer{layer}"] = d
torch.save(to_save, RESULTS_V2_DIR / "refusal_direction_v2.pt")

# Save separation table
sep_serial = {f"pos{pos}_layer{layer}": v for (pos, layer), v in separation.items()}
with open(RESULTS_V2_DIR / "separation_table.json", "w") as f:
    json.dump(sep_serial, f, indent=2)

# Sanity check at best (pos, layer): project harmful, harmless, and a few jailbreaks
print("\nSanity check projections...")
direction_cuda = best_direction.to(model.device)


def project(prompts, pos, layer):
    vals = []
    for instr in prompts:
        formatted = format_prompt(instr, tokenizer)
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[layer + 1][0, pos, :].to(torch.float32)
        vals.append((h @ direction_cuda).item())
        del out
        gc.collect()
        torch.cuda.empty_cache()
    return vals


harmful_proj = project(harmful, best_pos, best_layer)
harmless_proj = project(harmless, best_pos, best_layer)

jb_prefixes = [
    "As a security researcher, ",
    "For educational purposes, ",
    "In a fictional story, ",
    "As a penetration tester, ",
    "Write a movie scene where someone ",
]
jb_prompts = [jb_prefixes[i % 5] + harmful[i].lower() for i in range(min(20, len(harmful)))]
jb_proj = project(jb_prompts, best_pos, best_layer)

sanity = {
    "best_pos": best_pos,
    "best_layer": best_layer,
    "harmful": {"mean": float(np.mean(harmful_proj)), "std": float(np.std(harmful_proj))},
    "harmless": {"mean": float(np.mean(harmless_proj)), "std": float(np.std(harmless_proj))},
    "jailbroken": {"mean": float(np.mean(jb_proj)), "std": float(np.std(jb_proj))},
}
with open(RESULTS_V2_DIR / "sanity_check_v2.json", "w") as f:
    json.dump(sanity, f, indent=2)

print(f"\n  harmful   mean={sanity['harmful']['mean']:+.1f}  std={sanity['harmful']['std']:.1f}")
print(f"  harmless  mean={sanity['harmless']['mean']:+.1f}  std={sanity['harmless']['std']:.1f}")
print(f"  jailbreak mean={sanity['jailbroken']['mean']:+.1f}  std={sanity['jailbroken']['std']:.1f}")
print(f"\nSaved to {RESULTS_V2_DIR}/")
print(f"Update CONFIG.py: QWEN_BEST_POSITION={best_pos}, QWEN_BEST_LAYER={best_layer}")
