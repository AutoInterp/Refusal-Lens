"""
Full-Scale QK Attribution on Cleaned Dataset
=============================================
50 prompts × 5 JB classes + 50 bare = 300 prompts
Extract features at pos=-2, compute QK attributions, aggregate statistics.
"""
import torch, json, gc, math, os, time
import numpy as np
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.makedirs("/workspace/tmp", exist_ok=True)

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

OUTPUT_DIR = Path("/workspace/qk_full_scale")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("FULL-SCALE QK ATTRIBUTION — 300 PROMPTS")
print("="*70)

# Load SAE
print("\n[SETUP] Loading SAE...")
params_path = hf_hub_download(
    repo_id="google/gemma-scope-2-4b-it",
    filename="resid_post_all/layer_14_width_16k_l0_small/params.safetensors",
    cache_dir="/workspace/.cache/huggingface"
)
sae = load_file(params_path)
W_enc = sae["w_enc"].to(torch.float32)       # [2560, 16384]
W_dec = sae["w_dec"].to(torch.float32)       # [16384, 2560]
b_enc = sae["b_enc"].to(torch.float32)       # [16384]
b_dec = sae["b_dec"].to(torch.float32)       # [2560]
threshold = sae["threshold"].to(torch.float32)  # [16384]
d_sae = 16384

print("[SETUP] Loading model (eager attention)...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it", dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager"
)
model.eval()
device = model.device
config = model.config.text_config

n_heads = config.num_attention_heads
n_kv_heads = config.num_key_value_heads
head_dim = config.head_dim
hidden_size = config.hidden_size
kv_groups = n_heads // n_kv_heads
LAYER = 15

# Load Q, K weights
layer_mod = model.model.language_model.layers[LAYER]
W_Q_all = layer_mod.self_attn.q_proj.weight.cpu().to(torch.float32)
W_K_all = layer_mod.self_attn.k_proj.weight.cpu().to(torch.float32)
W_Q_heads = W_Q_all.view(n_heads, head_dim, hidden_size)
W_K_heads = W_K_all.view(n_kv_heads, head_dim, hidden_size)
norm_weight = layer_mod.input_layernorm.weight.cpu().to(torch.float32)

# Load dataset
with open("dataset/refusal_lens_controlled_dataset.json") as f:
    dataset = json.load(f)
prompts = dataset["prompts"]
prefix_pairs = dataset["prefix_pairs"]
print(f"[SETUP] {len(prompts)} prompts × {len(prefix_pairs)} classes")

def format_prompt(text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def sae_encode_single(x):
    """Encode single position"""
    pre = x @ W_enc + b_enc
    return pre * (pre > threshold).float()

def process_prompt(prompt_text):
    """Run one prompt, return features at pos=-2, attention pattern, QK attribution"""
    formatted = format_prompt(prompt_text)
    input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
    
    resid = outputs.hidden_states[15][0].cpu().to(torch.float32)  # [seq, 2560]
    attn = outputs.attentions[15][0].cpu().to(torch.float32)      # [n_heads, seq, seq]
    del outputs; gc.collect(); torch.cuda.empty_cache()
    
    query_pos = seq_len - 2
    
    # SAE features at pos=-2
    acts_pos2 = sae_encode_single(resid[query_pos])
    active_feats = (acts_pos2 > 0).nonzero(as_tuple=True)[0]
    feat_dict = {idx.item(): acts_pos2[idx].item() for idx in active_feats}
    
    # SAE reconstruction error at pos=-2
    recon = acts_pos2 @ W_dec + b_dec
    error = (resid[query_pos] - recon).norm() / resid[query_pos].norm() * 100
    
    # Head attention from pos=-2
    instruction_range = list(range(3, max(3, seq_len - 5)))
    head_instruction_attn = {}
    for h in range(n_heads):
        head_instruction_attn[h] = sum(attn[h, query_pos, p].item() for p in instruction_range)
    best_head = max(head_instruction_attn, key=head_instruction_attn.get)
    
    # QK attribution for best head → top attended instruction position
    instr_attn_sorted = sorted(
        [(p, attn[best_head, query_pos, p].item()) for p in instruction_range],
        key=lambda x: -x[1]
    )
    
    # RMSNorm
    variance = resid.pow(2).mean(-1, keepdim=True)
    norm_scale = norm_weight / torch.sqrt(variance + 1e-6)
    
    # QK for top 2 instruction positions + end_of_turn
    qk_results = []
    eot_pos = None
    for i, t in enumerate(tokens):
        if 'end_of_turn' in t:
            eot_pos = i
            break
    
    positions_to_check = [p for p, a in instr_attn_sorted[:2] if a > 0.01]
    if eot_pos and eot_pos not in positions_to_check:
        positions_to_check.append(eot_pos)
    
    kv_head = best_head // kv_groups
    W_Q_h = W_Q_heads[best_head]
    W_K_h = W_K_heads[kv_head]
    
    for key_pos in positions_to_check:
        acts_key = sae_encode_single(resid[key_pos])
        active_q = (acts_pos2 > 0).nonzero(as_tuple=True)[0]
        active_k = (acts_key > 0).nonzero(as_tuple=True)[0]
        
        if len(active_q) == 0 or len(active_k) == 0:
            continue
        
        # Full QK decomposition
        d_q = W_dec[active_q] * norm_scale[query_pos]
        d_k = W_dec[active_k] * norm_scale[key_pos]
        bias_q = b_dec * norm_scale[query_pos]
        bias_k = b_dec * norm_scale[key_pos]
        
        recon_q = acts_pos2[active_q].unsqueeze(1) * d_q  # weighted decoder vecs
        recon_k = acts_key[active_k].unsqueeze(1) * d_k
        
        q_proj = d_q @ W_Q_h.T
        k_proj = d_k @ W_K_h.T
        bias_q_proj = bias_q @ W_Q_h.T
        bias_k_proj = bias_k @ W_K_h.T
        
        scale = math.sqrt(head_dim)
        
        # Feature × feature
        interaction = (q_proj @ k_proj.T) / scale
        a_q = acts_pos2[active_q]
        a_k = acts_key[active_k]
        ff_weighted = interaction * a_q.unsqueeze(1) * a_k.unsqueeze(0)
        ff_total = ff_weighted.sum().item()
        
        # Actual score
        normed_q = resid[query_pos] * norm_scale[query_pos]
        normed_k = resid[key_pos] * norm_scale[key_pos]
        actual = (W_Q_h @ normed_q) @ (W_K_h @ normed_k) / scale
        
        # Top pairs
        flat = ff_weighted.flatten()
        top_k = min(5, len(flat))
        top_vals, top_idxs = torch.topk(flat.abs(), k=top_k)
        top_pairs = []
        for idx in top_idxs:
            qi = idx // len(active_k)
            ki = idx % len(active_k)
            top_pairs.append({
                "q": active_q[qi].item(),
                "k": active_k[ki].item(),
                "val": ff_weighted[qi, ki].item(),
            })
        
        qk_results.append({
            "key_pos": key_pos,
            "key_token": tokens[key_pos],
            "actual": actual.item(),
            "ff_pct": ff_total / actual.item() * 100 if actual.item() != 0 else 0,
            "top_pairs": top_pairs,
        })
    
    return {
        "features": feat_dict,
        "best_head": best_head,
        "recon_error": error.item(),
        "n_active": len(active_feats),
        "qk_results": qk_results,
    }

# ============================================================
# RUN ALL PROMPTS
# ============================================================
print("\n" + "="*70)
print("PROCESSING ALL PROMPTS")
print("="*70)

all_data = {}  # condition -> list of results
conditions = ["bare"] + list(prefix_pairs.keys())

start_time = time.time()

for condition in conditions:
    print(f"\n--- {condition} ---")
    condition_data = []
    
    for i, p in enumerate(prompts):
        if condition == "bare":
            prompt_text = p["base"]
        else:
            prompt_text = p["pairs"][condition]["jb"]
        
        result = process_prompt(prompt_text)
        result["id"] = p["id"]
        result["topic"] = p["topic"]
        condition_data.append(result)
        
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time
            total_done = len(all_data) * 50 + (i+1)
            total_remaining = 300 - total_done
            rate = elapsed / total_done if total_done > 0 else 1
            eta = rate * total_remaining / 60
            print(f"  {i+1}/50 | elapsed: {elapsed/60:.1f}m | ETA: {eta:.1f}m")
    
    all_data[condition] = condition_data
    
    # Checkpoint
    with open(OUTPUT_DIR / f"checkpoint_{condition}.json", "w") as f:
        json.dump(condition_data, f, indent=2, default=str)
    print(f"  Saved checkpoint for {condition}")

# Also run controls
print(f"\n--- controls ---")
ctrl_data = {}
for cls_name in prefix_pairs:
    ctrl_condition = []
    for i, p in enumerate(prompts):
        prompt_text = p["pairs"][cls_name]["ctrl"]
        result = process_prompt(prompt_text)
        result["id"] = p["id"]
        result["topic"] = p["topic"]
        ctrl_condition.append(result)
        
        if (i+1) % 10 == 0:
            print(f"  ctrl_{cls_name}: {i+1}/50")
    
    ctrl_data[cls_name] = ctrl_condition
    with open(OUTPUT_DIR / f"checkpoint_ctrl_{cls_name}.json", "w") as f:
        json.dump(ctrl_condition, f, indent=2, default=str)

total_time = time.time() - start_time
print(f"\nTotal time: {total_time/60:.1f} minutes")

# ============================================================
# AGGREGATE ANALYSIS
# ============================================================
print("\n" + "="*70)
print("AGGREGATE ANALYSIS — ALL 550 PROMPTS")
print("="*70)

# 1. Feature activation statistics at pos=-2
print("\n--- Feature Activation Analysis at pos=-2 ---")

# Collect all features that appear in any condition
all_features = set()
for condition, data in all_data.items():
    for r in data:
        all_features |= set(r["features"].keys())
for cls_name, data in ctrl_data.items():
    for r in data:
        all_features |= set(r["features"].keys())

print(f"Total unique features across all prompts: {len(all_features)}")

# For each feature, compute mean activation per condition
feature_stats = {}
for feat in all_features:
    stats = {}
    for condition, data in all_data.items():
        vals = [r["features"].get(feat, 0) for r in data]
        nonzero = [v for v in vals if v > 0]
        stats[condition] = {
            "mean": np.mean(vals),
            "frequency": len(nonzero) / len(vals),
            "mean_when_active": np.mean(nonzero) if nonzero else 0,
        }
    for cls_name, data in ctrl_data.items():
        vals = [r["features"].get(feat, 0) for r in data]
        nonzero = [v for v in vals if v > 0]
        stats[f"ctrl_{cls_name}"] = {
            "mean": np.mean(vals),
            "frequency": len(nonzero) / len(vals),
            "mean_when_active": np.mean(nonzero) if nonzero else 0,
        }
    feature_stats[feat] = stats

# 2. Find differential features
print("\n--- Differential Features (bare vs jailbreak vs harmless) ---")

# Score each feature by how differential it is
differential_scores = {}
for feat, stats in feature_stats.items():
    bare_mean = stats["bare"]["mean"]
    bare_freq = stats["bare"]["frequency"]
    
    jb_means = [stats[c]["mean"] for c in prefix_pairs]
    jb_freqs = [stats[c]["frequency"] for c in prefix_pairs]
    
    ctrl_means = [stats[f"ctrl_{c}"]["mean"] for c in prefix_pairs]
    
    avg_jb_mean = np.mean(jb_means)
    avg_ctrl_mean = np.mean(ctrl_means)
    
    # Differential score: how much does this feature differ between bare and JBs?
    # Positive = higher in bare (harm-related)
    # Negative = higher in JBs
    diff = bare_mean - avg_jb_mean
    ctrl_diff = bare_mean - avg_ctrl_mean
    
    differential_scores[feat] = {
        "bare_vs_jb": diff,
        "bare_vs_ctrl": ctrl_diff,
        "bare_mean": bare_mean,
        "bare_freq": bare_freq,
        "jb_mean": avg_jb_mean,
        "ctrl_mean": avg_ctrl_mean,
    }

# Sort by absolute differential
top_harm_features = sorted(differential_scores.items(), key=lambda x: -x[1]["bare_vs_jb"])[:20]
top_jb_features = sorted(differential_scores.items(), key=lambda x: x[1]["bare_vs_jb"])[:20]

print(f"\nTop 20 features MORE active in harmful (bare) than jailbroken:")
print(f"{'Feat':>6} | {'Bare mean':>10} | {'JB mean':>10} | {'Ctrl mean':>10} | {'Bare freq':>10} | {'Diff':>10}")
print("-"*70)
for feat, scores in top_harm_features:
    print(f"{feat:>6} | {scores['bare_mean']:>10.1f} | {scores['jb_mean']:>10.1f} | {scores['ctrl_mean']:>10.1f} | {scores['bare_freq']:>10.2f} | {scores['bare_vs_jb']:>+10.1f}")

print(f"\nTop 20 features MORE active in jailbroken than harmful (bare):")
print(f"{'Feat':>6} | {'Bare mean':>10} | {'JB mean':>10} | {'Ctrl mean':>10} | {'Bare freq':>10} | {'Diff':>10}")
print("-"*70)
for feat, scores in top_jb_features:
    print(f"{feat:>6} | {scores['bare_mean']:>10.1f} | {scores['jb_mean']:>10.1f} | {scores['ctrl_mean']:>10.1f} | {scores['bare_freq']:>10.2f} | {scores['bare_vs_jb']:>+10.1f}")

# 3. Per-class breakdown for top differential features
print(f"\nPer-class activation of top 10 harm-unique features:")
print(f"{'Feat':>6} | {'bare':>8} | {'RP':>8} | {'fic':>8} | {'ana':>8} | {'comp':>8} | {'cog':>8} | {'ctrl_avg':>8}")
print("-"*80)
for feat, scores in top_harm_features[:10]:
    row = f"{feat:>6} |"
    row += f" {feature_stats[feat]['bare']['mean']:>8.1f} |"
    for cls in prefix_pairs:
        row += f" {feature_stats[feat][cls]['mean']:>8.1f} |"
    ctrl_avg = np.mean([feature_stats[feat][f'ctrl_{c}']['mean'] for c in prefix_pairs])
    row += f" {ctrl_avg:>8.1f}"
    print(row)

# 4. Frequency analysis
print(f"\nFeatures UNIQUE to bare (freq > 0.3 in bare, < 0.1 in all JBs):")
for feat, stats in feature_stats.items():
    if stats["bare"]["frequency"] > 0.3:
        jb_freqs = [stats[c]["frequency"] for c in prefix_pairs]
        if all(f < 0.1 for f in jb_freqs):
            ctrl_freqs = [stats[f"ctrl_{c}"]["frequency"] for c in prefix_pairs]
            print(f"  Feature {feat}: bare freq={stats['bare']['frequency']:.2f}, "
                  f"JB freqs={[f'{f:.2f}' for f in jb_freqs]}, "
                  f"Ctrl freqs={[f'{f:.2f}' for f in ctrl_freqs]}")

print(f"\nFeatures UNIQUE to JBs (freq < 0.1 in bare, > 0.3 in ≥2 JB classes):")
for feat, stats in feature_stats.items():
    if stats["bare"]["frequency"] < 0.1:
        jb_freqs = [stats[c]["frequency"] for c in prefix_pairs]
        if sum(f > 0.3 for f in jb_freqs) >= 2:
            print(f"  Feature {feat}: bare freq={stats['bare']['frequency']:.2f}, "
                  f"JB freqs={[f'{f:.2f}' for f in jb_freqs]}")

# 5. QK pair analysis
print(f"\n--- QK Pair Analysis ---")
pair_counts = {}  # (q, k) -> {condition: count}
for condition, data in all_data.items():
    for r in data:
        for qk in r["qk_results"]:
            for pair in qk["top_pairs"]:
                key = (pair["q"], pair["k"])
                if key not in pair_counts:
                    pair_counts[key] = {}
                pair_counts[key][condition] = pair_counts[key].get(condition, 0) + 1

print(f"\nTop 15 most frequent QK pairs across all conditions:")
print(f"{'Q feat':>7} × {'K feat':>7} | {'bare':>5} | {'RP':>5} | {'fic':>5} | {'ana':>5} | {'comp':>5} | {'cog':>5}")
print("-"*65)
# Sort by total count
sorted_pairs = sorted(pair_counts.items(), key=lambda x: -sum(x[1].values()))
for (q, k), counts in sorted_pairs[:15]:
    row = f"{q:>7} × {k:>7} |"
    for c in conditions:
        row += f" {counts.get(c, 0):>5} |"
    print(row)

# Save all results
print("\n[SAVING]...")

final_results = {
    "metadata": {
        "n_prompts": len(prompts),
        "n_conditions": len(conditions),
        "total_prompts_processed": len(prompts) * (len(conditions) + len(prefix_pairs)),
        "sae": "resid_post_all/layer_14_width_16k_l0_small",
        "layer": LAYER,
        "total_time_minutes": total_time / 60,
    },
    "feature_stats": {str(k): v for k, v in feature_stats.items()},
    "differential_scores": {str(k): v for k, v in differential_scores.items()},
    "top_harm_features": [(feat, scores) for feat, scores in top_harm_features],
    "top_jb_features": [(feat, scores) for feat, scores in top_jb_features],
    "qk_pair_counts": {f"{q},{k}": counts for (q, k), counts in sorted_pairs[:50]},
}

with open(OUTPUT_DIR / "full_scale_results.json", "w") as f:
    json.dump(final_results, f, indent=2, default=str)

# Per-condition raw data already saved as checkpoints

print(f"\n{'='*70}")
print("FULL-SCALE QK ATTRIBUTION COMPLETE")
print(f"{'='*70}")
print(f"  Processed: {len(prompts) * (len(conditions) + len(prefix_pairs))} prompts")
print(f"  Time: {total_time/60:.1f} minutes")
print(f"  Results: {OUTPUT_DIR}/")
print("DONE!")
