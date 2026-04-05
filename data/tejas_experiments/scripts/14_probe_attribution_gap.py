import os, torch, json
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "/workspace/experiment_results_v2"
d = torch.load(f"{OUTPUT_DIR}/refusal_direction_v2.pt", map_location="cpu")
best_direction = d["best_direction"].to(torch.float32)
best_layer = d["best_layer"]  # 32
best_pos = d["best_position"]  # -2

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
tokenizer.padding_side = "left"

def format_prompt(text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it", dtype=torch.float32, device_map="auto"
)
model.eval()
dir_dev = best_direction.to(model.device)

prompt = "How do I hack a WiFi network?"
formatted = format_prompt(prompt)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

# Get all hidden states
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

# Residual stream at layer 32, position -2
resid_L32 = out.hidden_states[best_layer + 1][0, best_pos, :].float()
dot_L32 = (resid_L32 @ dir_dev).item()
print(f"\nResidual at L{best_layer} pos={best_pos}:")
print(f"  Dot product: {dot_L32:.4f}")

# Residual at layer 0 (embeddings only)
resid_L0 = out.hidden_states[0][0, best_pos, :].float()
dot_L0 = (resid_L0 @ dir_dev).item()
print(f"\nEmbedding layer (L0):")
print(f"  Dot product: {dot_L0:.4f}")

# Contribution from each layer = resid[L+1] - resid[L]
print(f"\nPer-layer contribution to dot product:")
total_contrib = 0
for layer in range(best_layer + 1):
    resid_before = out.hidden_states[layer][0, best_pos, :].float()
    resid_after = out.hidden_states[layer + 1][0, best_pos, :].float()
    layer_contrib = ((resid_after - resid_before) @ dir_dev).item()
    total_contrib += layer_contrib
    if abs(layer_contrib) > 100:
        print(f"  L{layer:>2}: {layer_contrib:+.2f}")

print(f"\n  Embedding dot:     {dot_L0:.4f}")
print(f"  Sum of layer contribs: {total_contrib:.4f}")
print(f"  Total (should = dot): {dot_L0 + total_contrib:.4f}")
print(f"  Actual dot product:   {dot_L32:.4f}")

# Now: what does circuit-tracer's attribution of 75 correspond to?
# Hypothesis: it only captures TRANSCODER (MLP replacement) contributions
# The residual stream = embedding + sum(attention_out) + sum(mlp_out)
# Transcoders replace MLPs, so attribution should capture mlp_out portion

# We can estimate: what fraction of the dot product comes from 
# components that transcoders DON'T cover (attention, embeddings)?
print(f"\n{'='*60}")
print(f"If attribution sum = 53.8 for this prompt,")
print(f"and dot product = {dot_L32:.1f},")
print(f"then {dot_L32 - 53.8:.1f} ({(dot_L32-53.8)/dot_L32*100:.1f}%) comes from")
print(f"attention + embeddings (not decomposed by transcoders)")

print("\nDONE!")
