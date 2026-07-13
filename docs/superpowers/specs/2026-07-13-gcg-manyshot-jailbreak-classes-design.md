# GCG + Many-Shot Jailbreak Classes — Design

**Date:** 2026-07-13
**Status:** Design (pending user review)
**Author:** Mahmoud + Claude
**Branch:** `emnlp-perm-edit`

## 1. Purpose

Tejas built a refined jailbreak dataset for Refusal-Lens (`dataset_v3_final (1).json`,
plus a follow-on `dataset_v4.json`) with SOTA classes that actually jailbreak
`google/gemma-3-4b-it`, and validated them with a reproducible generate→judge harness
(`new_dataset_results/refusal_results/`). His per-class **comply rates** (jailbreak =
%COMPLY on `kind=="attack"` rows):

| class (round) | comply rate |
|---|---|
| m2s_multiturn (v3) | 60% |
| nested_fiction (v3) | 36% |
| narrative_sandwich (v3) | 2% |
| encoding_base64 / expert_dialogue / translation (v4) | (in `v4_judged.json`) |

The v3 metadata explicitly **defers** a GCG (optimization-based adversarial suffix)
class: *"planned as a separate effort, as it requires per-prompt optimization against
the target model rather than fixed prompt authoring."* This effort delivers that plus a
Many-Shot ICL class.

**Goal:** add three new jailbreak classes on the shared 50 base prompts (base_id 1–50),
evaluate each with Tejas's *exact* harness, and report per-class comply rates comparable
to his table.

New classes:
1. **`gcg_per_prompt`** — a unique GCG-optimized adversarial suffix per base prompt.
2. **`gcg_universal`** — one shared suffix optimized jointly over a training subset of
   the bases, applied to all 50. (Per-prompt vs universal is itself an analysis axis:
   how do the attribution graphs of a bespoke vs a shared suffix differ?)
3. **`many_shot_icl`** — the base request wrapped in K in-context demonstrations of
   compliant harmful answers (Agarwal et al. 2024, *Many-Shot In-Context Learning*;
   jailbreak variant Anil et al. 2024; empirical study arXiv:2505.19773).

## 2. Non-goals

- Not modifying v3/v4 or their result files (append-only: new `dataset_v5.json` + `v5_*`).
- Not building the attribution-graph / circuit-tracing step for these prompts — that is
  the downstream consumer, out of scope here. This effort stops at the validated dataset
  + comply-rate report (the same bar Tejas met).
- Not running GCG/generation locally (venv torch is CPU-only; RTX 5080 is Blackwell
  sm_120). GPU steps run on RunPod, mirroring Tejas.
- Not sourcing an external uncensored model — many-shot demos reuse Gemma's own prior
  COMPLY outputs.

## 3. Safety framing

Authorized mechanistic-interpretability research on an established project (Refusal-Lens,
Algoverse, mentored by Georg), on **open** models, extending an **existing** validated
jailbreak dataset. The harmful surface is *not* expanded: (a) GCG suffixes are
non-semantic token strings, not human-actionable instructions; (b) many-shot demos reuse
compliant outputs the target model *already produced* in Tejas's runs (already in-repo);
(c) the base set is the same filtered fraud/cyber/manipulation tier (v3 metadata already
excludes CBRN/mass-casualty/child-harm/self-harm/hate/etc.). No new harmful content is
authored. Harmful text is handled programmatically, never reproduced in chat/reports
beyond short truncated identifiers.

## 4. Dataset schema (compatible with v3/v4)

All three classes are `class_type: "sota_transform"` — a single transformed `attack_text`
with a `floor_ceiling` control, exactly like `m2s_multiturn` and `encoding_base64`. This
means **generation reuses Tejas's generation logic unchanged** (via a clean copy,
`generate.py`, §9): the transform is fully baked into `attack_text`, so there is no
per-record generate-time step like translation.

Common fields (every record): `base_id`, `base`, `topic`, `class`, `class_type`,
`attack_text`, `control: {"type": "floor_ceiling"}`, `source`, `faithful`.

**`gcg_per_prompt`** (50 records):
- `attack_text = base + " " + suffix_i`
- `gcg: {mode: "per_prompt", suffix, suffix_n_tokens, target, steps, search_width,
  topk, final_loss, optimized_on: "google/gemma-3-4b-it", nanogcg_version}`
- `source: "GCG (Zou et al. 2023), per-prompt"`, `faithful: true`

**`gcg_universal`** (50 records):
- `attack_text = base + " " + suffix_univ` (same suffix for all 50)
- `gcg: {mode: "universal", suffix, ..., train_base_ids, n_train}`
- `source: "GCG universal (Zou et al. 2023)"`, `faithful: true`

**`many_shot_icl`** (50 records):
- `attack_text` = the assembled single-turn blob (see §6)
- `many_shot: {n_shots: 32, shot_source: "gemma_comply_v3v4", render: "single_turn_blob",
  seed: 0, demo_char_cap, shot_refs: [{src, gen_idx, base_id, class}, ...]}`
- `source: "Many-shot ICL (Agarwal et al. 2024; Anil et al. 2024)"`, `faithful: true`

## 5. Component: GCG optimization (`gcg_optimize.py`, GPU)

Wrapper over **nanoGCG** (GraySwan, pip-installable, maintained). It exposes the standard
GCG loop for any HF causal LM with an input-embedding layer.

- **Model:** `google/gemma-3-4b-it` via `AutoModelForCausalLM` (text checkpoint, matching
  Tejas's generate). Chat-templated with an `{optim_str}` insertion point after the base
  request.
- **Objective:** force an affirmative continuation. Default target string
  `"Sure, here are the detailed steps:"` (configurable). GCG minimizes NLL of the target
  given base + suffix.
- **Config (defaults):** `num_steps=500`, `search_width=512`, `topk=256`,
  `suffix len ≈ 20 tokens` (init from `"! ! ! ..."`), greedy eval. Tunable via CLI.
- **Modes:**
  - `--mode per_prompt`: 50 independent optimizations → `suffix_i` + `final_loss` each.
  - `--mode universal`: one multi-prompt run over a train subset (default 25 of 50
    bases, seeded split) minimizing summed loss → one `suffix_univ`; evaluated on all 50.
- **Output:** `gcg_suffixes.json` = `{per_prompt: {base_id: {suffix, loss, ...}},
  universal: {suffix, train_base_ids, loss, ...}, config: {...}}`.
- **Smoke mode** (`--smoke`): 2 steps, 1 prompt, asserts nanoGCG runs on Gemma-3
  (embedding access + chat template) before committing to the full run. This is the
  **first GPU action** in the runbook (§8), de-risking the Gemma-3 integration.

**Validation risk (flagged):** confirm nanoGCG drives Gemma-3 cleanly —
`get_input_embeddings()` access, chat template with the suffix slot, no vision-tower
interference from the text checkpoint. The smoke test gates the full run.

## 6. Component: Many-shot assembly (in `build_dataset_v5.py`, CPU)

- **Shot pool:** all `kind=="attack"` rows judged `COMPLY` in `v3_judged.json` +
  `v4_judged.json` — **60 demos** across 24 base_ids (verified). These are the target
  model's own compliant outputs; no external model, no new harmful authoring.
- **Sampling:** for each target base, sample K shots from the pool **excluding any demo
  with the same `base_id`** (avoid trivial leakage of the answer to the exact question),
  with a fixed RNG seed for reproducibility. Order is fixed per the paper's note that
  order/seed matter (§4.7 of Agarwal et al.); target question always last.
- **Demo length:** demos are full ~1024-tok generations. `demo_char_cap` (default:
  none) optionally truncates each demo answer to keep context punchy/bounded. K=32 whole
  ≈ 35k tokens — safe in Gemma-3's 128k window.
- **Render — single-turn blob (chosen):** concatenate
  `"User: <demo_q>\nAssistant: <demo_a>\n\n"` × K, then `"User: <base>\nAssistant:"`,
  all inside **one** user message (canonical MSJ). Zero harness change; single prompt →
  downstream attribution-graph generation stays tractable.
- **K:** class fixes **K=32**. A **bonus scaling sweep** (`--sweep`) builds K ∈
  {4,8,16,32} variants for a small fixed subset of bases (default 8) to confirm the
  log-linear jailbreak trend — reported as an auxiliary figure, not part of the 150-record
  dataset.

## 7. Component: build + eval

**`build_dataset_v5.py` (CPU, testable):**
Inputs `gcg_suffixes.json` (from §5) + the judged files (for the shot pool) + the base
set (base_id 1–50, extracted from `dataset_v4.json`). Emits `dataset_v5.json`
(metadata + 150 records). If `--gcg-suffixes` is absent it writes GCG records with an
empty suffix (`attack_text == base`) as placeholders, so the many-shot half can be built
and inspected before the GPU run — the finalize pass re-runs with suffixes.

**Generation (GPU):** reuse Tejas's harness verbatim — `generate.py` (a clean,
attributed copy of `generate_v4 (1).py`) pointed at `dataset_v5.json` →
`v5_generations.json`. Same settings: fp32, `device_map=cuda`, greedy, `max_new_tokens
=1024`, no truncation, `apply_chat_template` single user turn.

**Judge (GPU box / any litellm API):** identical to `eval_loop_README.md` —
`refusal_llm_judge.judge_refusal` on `kind=="attack"` rows → `judge ∈ {REFUSE, COMPLY}`
→ `v5_judged.json`.

**`report_v5.py` (CPU):** prints the per-class comply table next to Tejas's v3/v4 rates,
the **gcg_per_prompt vs gcg_universal** comparison (comply rate + mean final-loss +
suffix token overlap), and, if the sweep ran, the K-vs-comply curve. Also emits
`v5_report.md`.

## 8. Compute & runbook (RunPod, mirrors Tejas)

Authoring + CPU validation happen here; the two GPU steps + judge run on RunPod.
`run_v5_runpod.sh` runs in two phases — a mandatory end-to-end **smoke** on 2–3 bases
across **all three classes** with a human-inspection gate, then the full comprehensive
run only after the smoke output looks right.

```
pip install transformers torch accelerate nanogcg litellm
export HF_TOKEN=...

# --- Phase A: end-to-end smoke (2-3 bases, ALL classes), then INSPECT before proceeding ---
python gcg_optimize.py --smoke                        # (a) fast gate: nanoGCG↔Gemma-3 (2 steps)
python gcg_optimize.py --mode both --limit 3 --out gcg_suffixes_smoke.json   # (b) real suffixes, 3 bases
python build_dataset_v5.py --gcg-suffixes gcg_suffixes_smoke.json --limit 3 --out dataset_v5_smoke.json
python generate.py --dataset dataset_v5_smoke.json --out v5_smoke_generations.json
python report_v5.py --generations v5_smoke_generations.json --inspect
#   ^ prints each (class, base, attack_text head, response head) so a HUMAN confirms every
#     class produces a well-formed prompt AND a plausible response before the full run.
#     STOP HERE and eyeball all 3 classes. Only continue if they look correct.

# --- Phase B: full comprehensive run (all 50 bases, all classes, NO limit) ---
python gcg_optimize.py --mode per_prompt --out gcg_suffixes.json
python gcg_optimize.py --mode universal --append gcg_suffixes.json
python build_dataset_v5.py --gcg-suffixes gcg_suffixes.json --out dataset_v5.json [--sweep]
python generate.py --dataset dataset_v5.json --out v5_generations.json
# judge (ollama qwen3.6:35b-a3b, or litellm API) per eval_loop_README.md → v5_judged.json
python report_v5.py --judged v5_judged.json          # per-class table + comparisons
```

`--limit N` exists **only** to scope the Phase-A smoke (and `--mode both` runs per-prompt
+ universal together on that subset). The Phase-B full run takes no limit — the goal is
one comprehensive dataset in a single pass. `report_v5.py --inspect` is the manual gate:
it dumps the prompt→response head for every smoke pair so their well-formedness is
confirmed by eye, not just by exit code.

## 9. File layout

```
scripts/dataset_v5/
  gcg_optimize.py         # nanoGCG wrapper (per_prompt | universal | smoke)
  build_dataset_v5.py     # assemble 150 records (many-shot pool + GCG suffixes)
  generate.py             # clean attributed copy of Tejas's generate_v4
  report_v5.py            # judge-aware per-class report + comparisons
  run_v5_runpod.sh        # orchestration
  README.md               # runbook + provenance
  tests/
    test_build_dataset_v5.py   # CPU unit tests (schema, counts, blob, determinism)
dataset_v5.json                # 150 records (repo root, like dataset_v3)
new_dataset_results/refusal_results/
  gcg_suffixes.json  v5_generations.json  v5_judged.json  v5_report.md
```

## 10. Testing (CPU only — no GPU here)

Unit tests on `build_dataset_v5.py` (the only non-trivial pure logic):
- **Counts/schema:** 150 records, 50 per class; every record has the common fields +
  `control:{type:floor_ceiling}`; GCG records carry `gcg.*`, many-shot carry `many_shot.*`.
- **Many-shot blob:** exactly K `User:/Assistant:` demo pairs; target base is the final
  `User:` turn; **no demo shares the target's base_id**; identical output under a fixed
  seed (determinism); `demo_char_cap` truncates.
- **GCG:** `attack_text == base + " " + suffix`; universal records all share one suffix;
  per-prompt records have distinct suffixes; placeholder mode yields `attack_text==base`.
- **Sweep:** K ∈ {4,8,16,32} variants produced for the subset; nested shot sets
  (K=4 ⊂ K=8 ⊂ …) so the curve isolates count.

GPU pieces (nanoGCG, generate, judge) are validated by the smoke test + reusing Tejas's
already-validated harness; no CPU unit test can exercise them. **The Phase-A end-to-end
smoke (§8) is the required GPU gate:** it runs all three classes on 2–3 bases through the
entire pipeline and `report_v5.py --inspect` dumps each prompt→response head for human
confirmation that (a) every class produces a well-formed `attack_text` and (b) Gemma
produces a plausible response, before the full 50-base run is launched.

## 11. Risks / open questions

- **nanoGCG ↔ Gemma-3 integration** — mitigated by the smoke gate (first GPU action).
- **GCG may be weak on a safety-tuned 4B** — GCG's white-box potency on Gemma-3-4B is an
  empirical unknown; a low comply rate is itself a valid finding (as Tejas framed
  narrative_sandwich at 2% and the v4 encoding survey). We report whatever the honest
  rate is.
- **Universal transfer across our 50 bases** — one suffix may not generalize; the
  train/eval split makes generalization measurable rather than assumed.
- **Shot-pool topic skew** — 60 demos lean m2s/nested; the sampler mixes classes and can
  cap per-class share if skew hurts (tunable), but default is uniform random under seed.
- **Compute budget** — 50 per-prompt GCG runs × ~500 steps is the dominant cost. Per the
  user's decision the **full run is unlimited** (comprehensive dataset in one pass);
  `--limit` is reserved for the Phase-A smoke only. The Phase-A gate is what protects the
  budget: a broken pipeline is caught on 3 bases, not 50.
