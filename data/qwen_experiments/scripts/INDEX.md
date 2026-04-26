# Qwen3-4B Scripts Index

Mirror of `data/tejas_experiments/scripts/INDEX.md` for `Qwen/Qwen3-4B` with
`mwhanna/qwen3-4b-transcoders`.

## Layout

| Script | Purpose | Port status |
|---|---|---|
| `CONFIG.py` | Single source of truth (model, transcoder, paths, layers) | Ready |
| `01_compute_direction_and_sanity.py` | Sweep positions × layers, find best, sanity check | Ported (full) |
| `11_fix_all_qwen.py` | 10-pair attribution + mechanism comparison (RP vs fiction) | Ported (full) |
| `12_cosine_with_ruqiya.py` | Cosine-similarity check vs Ruqiya's direction | Stub — port from Gemma |
| `13_dot_product_check.py` | Verify attribution sum ≈ dot product | Stub |
| `14_probe_attribution_gap.py` | Per-layer probe of MLP-vs-attention split | Stub |
| `15_causal_intervention.py` | Direct ablation (Gemma version: failed at L15) | Stub |
| `16_causal_arditi.py` | Arditi method: add r at causal layer | Stub |
| `17_causal_georg_arditi.py` | Comparison of Arditi vs direct steering | Stub |
| `19_disentangle.py` | 2×2: every-step vs prefill-only × all-pos vs single-pos | Stub |
| `20_bulletproof_pipeline.py` | End-to-end on cleaned dataset | Ported (core phases) |
| `21_qk_full_scale.py` | Q/K attention attribution at scale | Stub |
| `22_qk_deep_rigorous.py` | Deep Q/K analysis + ablation | Stub |

## Run order

1. `01_compute_direction_and_sanity.py` — produces `refusal_direction_v2.pt`,
   `separation_table.json`, `sanity_check_v2.json`. Read the printed best
   `(position, layer)` and update `CONFIG.py`.
2. Inspect `https://huggingface.co/mwhanna/qwen3-4b-transcoders` and set
   `TRANSCODER_SUBPATH` in `CONFIG.py`.
3. `11_fix_all_qwen.py` — attribution + RP-vs-fiction mechanism comparison.
4. Identify Qwen's causal layer (port `15`/`16`/`17`); update
   `QWEN_CAUSAL_LAYER` in `CONFIG.py`.
5. `20_bulletproof_pipeline.py` — verified-dataset end-to-end with intervention.
6. Port `19`, `21`, `22` for the deeper analyses.

## Porting checklist (Gemma → Qwen)

When adapting any remaining script, replace:

| Gemma line | Qwen line | Why |
|---|---|---|
| `"google/gemma-3-4b-it"` | `MODEL_NAME` from CONFIG | central source |
| `"mwhanna/gemma-scope-2-4b-it/..."` | `f"{TRANSCODER_REPO}/{TRANSCODER_SUBPATH}"` | central source |
| `model.config.text_config.hidden_size` | `get_hidden_size(model)` | Qwen has flat config |
| `model.model.language_model.layers[L]` | `get_decoder_layers(model)[L]` | Qwen layout differs |
| `LAYER = 15` (causal) | `QWEN_CAUSAL_LAYER` | re-tune empirically |
| `pos=-2` (the `model` token) | `QWEN_BEST_POSITION` | Qwen template differs |
| `range(34)` for layers | `range(model.config.num_hidden_layers)` | Qwen has 36 |
| `/workspace/...` | `RESULTS_V2_DIR / ...` | use repo-relative paths |

See `MIGRATION_NOTES.md` in the parent directory for the full rationale.
