import os, torch, json
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.attribution.targets import CustomTarget

d = torch.load("/workspace/experiment_results/refusal_direction_layer13.pt", map_location="cpu", weights_only=True)
direction = d["direction"].to(torch.float32).cuda()

with open("/workspace/Refusal-Lens/dataset/refusal_direction_dataset/splits/harmful_train.json") as f:
    harmful = [p["instruction"] for p in json.load(f)]

# CONTROL: benign prefixes of similar length to jailbreak prefixes
control_prefixes = [
    "I have a question for you, ",
    "Can you help me with this, ",
    "I was wondering about this, ",
    "Please tell me about this, ",
    "I would like to know about, ",
]

print("Loading model...")
model = ReplacementModel.from_pretrained(
    "google/gemma-3-4b-pt",
    "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
    dtype=torch.float32, backend="nnsight", lazy_encoder=True,
)
print("Ready.")
target = CustomTarget(token_str="refusal_direction", prob=1.0, vec=direction)

results = []
for i in range(20):
    bare = harmful[i]
    ctrl = control_prefixes[i%5] + bare.lower()
    print(f"Pair {i+1}/20: {bare[:50]}...")
    torch.cuda.empty_cache()
    pair = {"bare": bare, "control": ctrl, "prefix": control_prefixes[i%5]}
    for label, prompt in [("bare", bare), ("ctrl", ctrl)]:
        try:
            g = attribute(prompt=prompt, model=model, attribution_targets=[target], batch_size=64, max_feature_nodes=3000, verbose=False)
            adj = g.adjacency_matrix
            last = adj[-1, :]
            pair[label+"_net"] = last.sum().item()
            print(f"  {label}: net={pair[label+'_net']:.3f}")
        except Exception as e:
            print(f"  {label} ERROR: {e}")
            pair[label+"_error"] = str(e)[:200]
    results.append(pair)
    with open("/workspace/experiment_results/control_20pairs.json","w") as f:
        json.dump(results, f, indent=2, default=str)

import numpy as np
print("\nCONTROL SUMMARY:")
bare_nets = [p["bare_net"] for p in results if "bare_net" in p]
ctrl_nets = [p["ctrl_net"] for p in results if "ctrl_net" in p]
print(f"Bare mean:    {np.mean(bare_nets):.3f} std={np.std(bare_nets):.3f}")
print(f"Control mean: {np.mean(ctrl_nets):.3f} std={np.std(ctrl_nets):.3f}")
print(f"Mean diff:    {np.mean(ctrl_nets) - np.mean(bare_nets):+.3f}")
ctrl_lower = sum(1 for p in results if "bare_net" in p and "ctrl_net" in p and p["ctrl_net"] < p["bare_net"])
print(f"Control lower: {ctrl_lower}/{len([p for p in results if 'bare_net' in p and 'ctrl_net' in p])}")
print("Done!")
