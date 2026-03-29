import sys, os, torch, json, gc
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.attribution.targets import CustomTarget

d = torch.load("/workspace/experiment_results/refusal_direction_layer13.pt", map_location="cpu", weights_only=True)
direction = d["direction"].to(torch.float32)

with open("/workspace/Refusal-Lens/dataset/refusal_direction_dataset/splits/harmful_train.json") as f:
    harmful = [p["instruction"] for p in json.load(f)]

prefixes = ["As a security researcher, ", "For educational purposes, ", "In a fictional story, ", "As a penetration tester, ", "Write a movie scene where someone "]

print("Loading model...")
model = ReplacementModel.from_pretrained(
    "google/gemma-3-4b-pt",
    "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
    dtype=torch.float32, backend="nnsight", lazy_encoder=True, offload="cpu",
)
print("Ready.")

target = CustomTarget(token_str="refusal_direction", prob=1.0, vec=direction)

results = []
for i in range(10):
    bare = harmful[i]
    jb = prefixes[i%5] + bare.lower()
    print(f"Pair {i+1}/10: {bare[:50]}...")
    torch.cuda.empty_cache()
    pair = {"bare": bare, "jb": jb}
    for label, prompt in [("bare", bare), ("jb", jb)]:
        try:
            g = attribute(prompt=prompt, model=model, attribution_targets=[target], batch_size=64, max_feature_nodes=3000, verbose=False)
            adj = g.adjacency_matrix
            last = adj[-1, :]
            pair[label+"_pos"] = last[last>0].sum().item()
            pair[label+"_neg"] = last[last<0].sum().item()
            pair[label+"_net"] = pair[label+"_pos"] + pair[label+"_neg"]
            print(f"  {label}: net={pair[label+'_net']:.3f}")
        except Exception as e:
            print(f"  {label} ERROR: {e}")
            pair[label+"_error"] = str(e)[:200]
    results.append(pair)
    with open("/workspace/experiment_results/attribution_10pairs.json","w") as f:
        json.dump(results, f, indent=2, default=str)

print("\nSUMMARY:")
for p in results:
    if "bare_net" in p and "jb_net" in p:
        diff = p["jb_net"] - p["bare_net"]
        print(f"  bare={p['bare_net']:+.3f} jb={p['jb_net']:+.3f} diff={diff:+.3f}")
print("\nDone!")
