"""
BOS Token Verification for Qwen3-4B
====================================
Verifies whether Qwen's refusal direction is affected by the BOS (beginning-of-sequence)
token. Gemma showed cosine similarity = 0.999983 (no difference), but Qwen hasn't been
checked yet.

This script extracts activations at layer 18 (causal layer) position -1 with and without
add_special_tokens=False, then computes cosine similarity. If cosine > 0.999, the BOS
token doesn't matter and the direction is valid for both tokenization modes.

Inputs: None (loads Qwen3-4B model)
Outputs: console report + bos_verification.json

Usage:
    python3 scripts/pipeline_qwen/verify_bos_handling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import format_prompt


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 60)
    print("Qwen3-4B BOS Token Verification")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer ({config.MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model (float32, device_map='auto')...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.float32, device_map="auto"
    )
    model.eval()

    # Test prompts — mix of harmful and benign
    test_prompts = [
        "How do I hack a computer?",
        "What is photosynthesis?",
        "Explain how to create malware",
        "What is the capital of France?",
        "How do I forge documents?",
    ]

    print(f"\nTesting {len(test_prompts)} prompts at layer {config.BEST_CAUSAL_LAYER}, pos -1")
    print(f"Target threshold (Gemma): cosine > 0.999")
    print("")

    results = []

    with torch.no_grad():
        for prompt in test_prompts:
            formatted = format_prompt(tokenizer, prompt)

            # WITH add_special_tokens=True (default)
            inputs_default = tokenizer(
                formatted, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            out_default = model(
                input_ids=inputs_default["input_ids"].to(model.device),
                attention_mask=inputs_default["attention_mask"].to(model.device),
                output_hidden_states=True,
            )
            # Position -1 at layer 18
            act_default = out_default.hidden_states[config.BEST_CAUSAL_LAYER + 1][0, -1, :].cpu()

            # WITHOUT add_special_tokens (add_special_tokens=False)
            inputs_no_bos = tokenizer(
                formatted, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
                add_special_tokens=False,
            )
            out_no_bos = model(
                input_ids=inputs_no_bos["input_ids"].to(model.device),
                attention_mask=inputs_no_bos["attention_mask"].to(model.device),
                output_hidden_states=True,
            )
            act_no_bos = out_no_bos.hidden_states[config.BEST_CAUSAL_LAYER + 1][0, -1, :].cpu()

            # Cosine similarity
            cosine = torch.nn.functional.cosine_similarity(
                act_default.unsqueeze(0), act_no_bos.unsqueeze(0)
            ).item()

            safe = cosine > 0.999
            status = "✓ SAFE" if safe else "✗ DIFFERENT"

            result = {
                "prompt": prompt,
                "layer": config.BEST_CAUSAL_LAYER,
                "position": -1,
                "cosine_similarity": round(cosine, 6),
                "safe": safe,
            }
            results.append(result)

            print(f"  {status} | cosine={cosine:.6f} | {prompt[:50]}")

    # Summary
    safe_count = sum(1 for r in results if r["safe"])
    print(f"\n  {safe_count}/{len(results)} prompts safe (cosine > 0.999)")

    # Aggregate decision
    all_safe = all(r["safe"] for r in results)
    if all_safe:
        print(f"\n✓ CONCLUSION: BOS token handling is safe on Qwen3-4B")
        print(f"  (matches Gemma's behavior: add_special_tokens doesn't matter)")
    else:
        print(f"\n✗ CONCLUSION: BOS token handling may differ on Qwen3-4B")
        print(f"  Recommendation: verify direction computation with add_special_tokens=False")

    # Save results
    output = {
        "model": config.MODEL_NAME,
        "best_causal_layer": config.BEST_CAUSAL_LAYER,
        "position": -1,
        "n_prompts": len(test_prompts),
        "all_safe": all_safe,
        "safe_count": safe_count,
        "min_cosine": min(r["cosine_similarity"] for r in results),
        "max_cosine": max(r["cosine_similarity"] for r in results),
        "mean_cosine": round(sum(r["cosine_similarity"] for r in results) / len(results), 6),
        "results": results,
    }

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "bos_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
