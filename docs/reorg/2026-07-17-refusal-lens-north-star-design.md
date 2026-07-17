# Refusal-Lens Reorganization — North-Star Design

**Status:** Draft for review · **Date:** 2026-07-17 · **Branch:** jb-dataset-refined
**Authors:** Mahmoud Shabana (with Tejas Dahiya) · **Mentor:** Georg Lange

---

## §1 · Purpose & guardrails

### Why this document exists
Georg has laid out a clear paper thesis as a chain of hypotheses (H1→H6, §2). Our
code proves large parts of this **for Gemma**, some of it for Qwen, and scatters the
rest across ~11 branches and two incompatible code organizations. This document is the
**umbrella plan** to reorganize the codebase so that:

1. Every hypothesis H1→H6 has **one obvious, model-agnostic home**.
2. Adding a model is **a registry entry, not a new folder**.
3. "Which experiment, on which model, is done?" is answerable by **looking at the tree**.
4. The move to **transcoders** (per-layer / skip / CLT) is behind a single abstraction.

It is a **north star**, not an implementation plan. Each sub-project in §7 gets its own
`brainstorm → spec → plan` cycle later. Nothing here should be built until its
sub-project spec is written and approved.

### Guardrails (non-negotiable)
- **Humans own the science; tooling owns the scaffolding.** Per Georg's guidance,
  experiment *design*, choice of *what to measure*, and *interpretation of results* are
  done by Mahmoud and Tejas — never delegated to AI. This reorg only makes those
  experiments easy to express, parameterize, run, and reproduce.
- **Every experiment records its reasoning.** Each `experiments/hN/` carries a
  `README.md` answering the 5 methodology questions (Tejas's convention):
  1. What is the methodology/experiment?
  2. **Why** are we doing this?
  3. What do we expect the results to be?
  4. What are the results? What do they tell us? Does it match expectation?
  5. What can we follow up on?
  This is where human reasoning lives; the code is downstream of it.
- **A prompt is not a jailbreak because we say so** (Georg). It is a jailbreak only when
  the model emits harmful content on the attack **and** refuses the same content without
  it. H2 enforces this; unqualified prompts are excluded from the dataset.

---

## §2 · The research spine (H1→H6 as a dependency DAG)

Georg's hypotheses form a dependency chain. The reorg's directory layout mirrors it.

| # | Hypothesis | Needs | Transcoder? | Paper role |
|---|---|---|---|---|
| **H1** | Refusal is a linear direction (r̂ = mean(harmful) − mean(harmless) at the last prompt token; ablate→comply, add→refuse). | model + residual hooks | ❌ | Reproduced in newer models → **appendix**, 1–2 sentences in body |
| **H2** | Our prompts genuinely jailbreak: harmful-on-attack **and** not-on-matched-control. Exclude the rest. | model + judge + **attack generators** | ❌ | Defines the dataset every later step consumes |
| **H3** | Jailbreaks are mediated via the refusal direction. **(a) Correlation:** JB prompts have lower r̂ magnitude, near harmless. **(b) Causation:** add magnitude back to harmful level → patches the jailbreak. **(c) Sufficiency:** subtract magnitude from a harmful prompt → induces jailbreak-like compliance. | H1 + H2 | ❌ | **The non-negotiable result** that justifies circuit tracing (was in the title but missing from the last submission) |
| **H4** | Circuit-tracing the refusal direction (attribution graphs A_{s→R}). | H1–H3 + **transcoders** | ✅ | The main experiment |
| **H5** | Method: math statement for tracing an arbitrary residual-stream direction. | — | ✅ (context) | Brief methods section |
| **H6** | Exploration: what the attribution graphs reveal (not yet fleshed out). | H4 | ✅ | The meat; a multi-day exploration phase |

**The load-bearing seam:** H1, H2, H3 need only the **model + residual-stream hooks** — no
transcoders. Only **H4+** needs transcoders. Therefore the foundation experiments run on
**all** models immediately, and the transcoder/model-tiering question (§5) gates **only**
H4. The reorg is **not blocked** on transcoder integration.

---

## §3 · Current-state assessment

*(Findings from a full read-only audit of `main`/`jb-dataset-refined` plus a branch survey.)*

### What's in place (and good)
- A real library `src/refusal_lens/` (15 modules): `model_loader`, `refusal_directions`
  (H1), `attribution` (H4 primitive, already direction/layer-parameterized),
  `attribution_pipeline`, `clt` (transcoder wrapper), `sae`, `supernode_analyzer`,
  `data_loader`, `refusal_classifier/detector`, `prompt_template`, `experiment_runner`.
- A model-agnostic judge package `refusal_llm_judge/` (litellm/vLLM, model passed as a
  string).
- Mature, working experiment code for Gemma across H1/H3/H4 and a `dataset_v5` attack
  pipeline that produced a strong Gemma jailbreak sample.

### What needs refactoring (the duplication disease)
- **Per-model script forks, not parameterization.** `scripts/pipeline/` (Gemma) and
  `scripts/pipeline_qwen/` (Qwen) are near-duplicates: 7 files byte-identical, the rest
  structurally identical with the model/layer/template lines swapped. Adding Llama today
  = a **third** fork.
- **~9 copy-pasted two-model dicts** in `scripts/emnlp_perm_edit/*` (`{"gemma": {...},
  "qwen": {...}}`), each the closest thing to a registry but re-declared per file.
- **Three competing config surfaces**, none shared: `src/refusal_lens/config.py`,
  `scripts/pipeline/config.py`, `scripts/pipeline_qwen/config.py` (imported as bare
  `import config`, unlinked from the package).
- **H1 direction extraction exists in ≥3 unlinked implementations** (the package,
  `pipeline/01_compute_direction.py`, `emnlp_perm_edit/00_compute_directions.py`) plus
  inline recomputes in three top-level scripts. The pipeline scripts **do not import the
  package**.
- **Transcoder naming drift.** `src/refusal_lens/clt.py` and `TRANSCODER_PATH` load
  **single-layer** transcoders (`mwhanna/gemma-scope-2-4b-it/…affine`) but the module is
  named "CLT" and its docstrings say "CLT features." The name lies about the substance.
- **Two incompatible code organizations** across branches: the staged
  `scripts/pipeline{,_qwen}/` (jb lineage) vs. the numbered `data/{tejas,qwen}_experiments/scripts/`
  (tejas lineage). The reorg must unify them, not just de-dup one.
- **Two attack lineages, unlinked:** Tejas's v3/v4 **literature techniques** —
  `nested_fiction, narrative_sandwich, m2s_multiturn, encoding_base64,
  translation_lowresource, expert_dialogue` (in
  `new_dataset_results/refusal_results/v3_judged.json` / `v4_judged.json`, which already
  carry length-matched `kind: control` records — i.e. the H2 criterion is baked into
  Tejas's harness) — and Mahmoud's v5 classes (`gcg_per_prompt, many_shot_icl,
  refusal_suppression[_prefill]` in `dataset_v5.json`). `dataset_v5` is Gemma-wired
  (`gcg_optimize.py` hardcodes the model; GCG excludes *Gemma* special tokens; many-shot
  pool = *Gemma's* comply outputs; prefill = *Gemma* chat template). Gemma-measured ASR
  already spans **0–60%** (m2s_multiturn 60%, nested_fiction 36%, expert_dialogue 22%,
  narrative_sandwich 2%, encoding_base64 / translation_lowresource 0%) — concrete evidence
  that "which classes work" is per-model empirical, not fixed.

### What's stranded on branches (must be consolidated — §6)
- `temp/gemma-vs-qwen-pipeline` (23 commits vs jb): the **canonical Qwen3-4B full-pipeline
  run** (Stage 01–08) + Gemma-vs-Qwen comparison. Unmerged code **and** results.
- `tejas-circuit-experiments` → `qwen_experiments` (own lineage off `main`): the
  **SAE/QK/linear-probe** investigation + the original controlled 50-prompt dataset, under
  `data/*_experiments/scripts/`.
- `dataset-v2-verification` (4 commits): the **H2 behavioral-verification toolchain** +
  the "v2 jailbreaks ≠ compliance" negative result.
- `tejas/dataset-10-classes` (1 commit: dataset_v2 + 5 literature classes),
  `add-refusal-direction` (1 commit: a precomputed direction `.pt`).
- `architectural_improvements`, `dev`, `visualization-figures`: **fully merged into main**;
  carry nothing unique → safe to delete.

*Everything unique on `foundation`, `l15-refactor`, `emnlp-perm-edit`,
`mshabana-rp-causal-exp` is already absorbed into `jb-dataset-refined`.*

---

## §4 · Target architecture (Approach B: library + hypothesis-organized experiments)

### Target tree
```
src/refusal_lens/
  registry.py         # MODELS: dict[name -> ModelSpec]   <-- the ONE source of truth (§4.1)
  models.py           # load + hook-manager + chat-template abstraction (absorbs pipeline/utils.format_prompt)
  directions.py       # H1 primitive: diff-in-means extraction (consolidates the 3 impls)
  steering.py         # H3 primitives: r-hat projection magnitude, add/subtract interventions
  attacks/            # attack GENERATORS + ATTACKS registry (unifies Tejas v3/v4 + v5) (§4.3)
    __init__.py       #   ATTACKS: dict[name -> AttackSpec]
    framing.py        #   nested_fiction, narrative_sandwich, expert_dialogue, refusal_suppression[_prefill]
    multiturn.py      #   m2s_multiturn
    obfuscation.py    #   encoding_base64, translation_lowresource
    many_shot.py      #   in-context; per-model comply-pool bootstrap
    gcg.py            #   per-tokenizer special-token exclusion + on-target optimization
  transcoders.py      # was clt.py — type-agnostic loader: PLT / CLT / skip, native(ct)+adapter (§4.5)
  transcoder_adapters/ #  KokosDev raw-torch, EleutherAI sae  -> circuit-tracer ReplacementModel
  attribution.py      # H4 primitive: attribute_to_direction (kept; already agnostic)
  judge/              # was refusal_llm_judge/ (already agnostic)
  probes.py           # linear probes (absorbs tejas lineage)

experiments/
  h1_direction/           run.py  README.md   # reproduce Arditi across models -> appendix
  h2_jailbreak_validation/ run.py README.md   # attack gen + judge + control-filter -> data/jailbreaks/<model>.json
  h3_mediation/           run.py  README.md   # (a) correlation (b) causation/patch (c) sufficiency/induce
  h4_attribution/         run.py  README.md   # circuit tracing (transcoder-gated; regenerate graphs)
  h5_method/              run.py  README.md   # direction-attribution math + validation
  h6_exploration/         (notebooks + scripts; later)

data/
  base_prompts/           # shared harmful/harmless bases
  jailbreaks/<model>.json # H2 output: validated "jailbreaks that work" per model
  jailbreaks/reports/<model>_asr.md
  directions/<model>/     # H1 output: per-layer r-hat

results/
  <model>/<hypothesis>/   # coverage == directory listing; a status matrix falls out for free

docs/reorg/               # this doc + one spec per sub-project
```

### §4.1 · Model registry — the one source of truth
Replaces the 3 config surfaces + 9 copy-pasted dicts. One `ModelSpec` per model:

```
ModelSpec:
  name            # "gemma-3-4b", "qwen3-4b", "llama-3.2-1b", ...
  hf_id           # "google/gemma-3-4b-it"
  family          # gemma | qwen | llama | deepseek
  n_layers, d_model
  measurement_layer(s)      # Gemma L15; Qwen L18; ... (causal layer)
  measurement_position      # Gemma -2 ("model" token); Qwen -1 (after assistant\n)
  direction_position        # position for H1 diff-in-means
  chat_template_kwargs      # e.g. Qwen enable_thinking=False ("load-bearing")
  transcoder:               # nullable — H1-H3 don't need it
    repo, type (plt|clt|skip), backend (nnsight|transformerlens), source (native|adapter)
  tier            # 1 (ct-native) | 2 (adapter-gated)
```
The Gemma-specific physics currently frozen as constants in `pipeline/config.py` (the
`hook_resid_post` basis-match, `pos=-2`, the L15-causal / L32-separation split) become
**per-model registry fields** — every one differs on Qwen and will differ again on Llama.

### §4.2 · Library consolidation
- Fold the ≥3 H1 implementations into `directions.py`; the pipeline/emnlp stages **import
  the package** instead of `import config` / re-implementing.
- Rename `clt.py` → `transcoders.py` and drop "CLT" from docstrings where the substance is
  single-layer (kill the misnomer).
- Lift `pipeline/utils.py:format_prompt` (the Gemma/Qwen chat-template divergence) into
  `models.py` as the shared chat-template abstraction, driven by `ModelSpec`.

### §4.3 · Attack-generation subsystem (the dataset engine)
The machinery that produces **per-model jailbreaks that work**, unifying both lineages
under one `ATTACKS` registry (categories: *framing / competing-objectives*, *multi-turn*,
*obfuscation*, *in-context*, *optimization*). The classes:

- *Framing / competing-objectives* (Tejas v3/v4 + v5): `nested_fiction, narrative_sandwich,
  expert_dialogue` and `refusal_suppression[_prefill]` — CPU prompt/chat-template work.
- *Multi-turn* (Tejas v3): `m2s_multiturn` (many-to-single multi-turn).
- *Obfuscation* (Tejas v4): `encoding_base64`, `translation_lowresource`.
- *In-context* (v5): `many_shot_icl` — needs the model's **own** comply pool (bootstrap
  dependency, below).
- *Optimization* (v5): `gcg_per_prompt` — per-tokenizer special-token exclusion +
  optimization **on the target model** (GPU).

Source data: Tejas's classes in `new_dataset_results/refusal_results/v3_judged.json` /
`v4_judged.json` (each with `kind: attack|control` — the H2 control-filter is already in
the harness); v5 in `dataset_v5.json`. Model-parameterized end-to-end:

- **Generators** (`attacks/`) take `(base_prompts, ModelSpec)` and emit candidate attacks.
  Framing/competing-objectives/obfuscation are CPU prompt/chat-template work; `gcg`
  optimizes on the target model with per-tokenizer special-token exclusion; `many_shot`
  consumes the model's own comply pool.
- **Orchestration is H2** (§4.4). Per model: render/optimize attacks → generate → judge →
  **harvest COMPLY outputs → build many-shot pool → gen/judge** (bootstrap ordering) →
  apply the H2 control-filter → write `data/jailbreaks/<model>.json` + an ASR report.
- **ASR is per-model, per-class.** Which classes survive on a given model is empirical
  output (e.g. many-shot was ~0% on Gemma, refusal_suppression ~98%); the human decides
  which to keep, the code just measures.

### §4.4 · Hypothesis experiments
Each `experiments/hN/run.py` is a thin `--model <name>` CLI over the library + a
methodology `README.md`. H2's `run.py` is the dataset orchestrator above; its output is the
input to H3 and H4. No experiment hardcodes a model.

### §4.5 · Transcoder abstraction & the "regenerate graphs" migration
`transcoders.py` loads by `ModelSpec.transcoder` regardless of type:
- **Native** (circuit-tracer): Gemma-3-4B, Qwen3-4B/14B, Llama-3.2-1B — repo string only.
- **Adapter** (`transcoder_adapters/`): `KokosDev/qwen25-32b-clt` (raw `torch.load`
  state-dicts) and `EleutherAI/skip-transcoder-…` (EleutherAI `sae` format, first-15-layers
  only) → converted into a circuit-tracer `ReplacementModel`.

Because the existing attribution graphs were generated with a **true CLT** (per Georg) and
the config now points at single-layer transcoders, **H4 must be re-run** under the chosen
transcoder per model. This is real work (P2), not a rename. (See open question §8.1 on
reconciling "graphs were CLT" with the config already pointing at a PLT repo.)

---

## §5 · Model fleet & coverage matrix

**H1–H3 run on all six** (transcoder-free). **H4 splits by transcoder availability.**

| Model | Tier | Transcoder | Type | ct-native? | H1–H3 | H4 |
|---|---|---|---|---|---|---|
| Gemma-3-4B | 1 | `mwhanna/gemma-scope-2-4b-it` | PLT | ✅ | ✅ | ✅ now |
| Qwen3-4B | 1 | `mwhanna/qwen3-4b-transcoders` | PLT | ✅ | ✅ | ✅ now |
| Llama-3.2-1B | 1 | circuit-tracer default | PLT/CLT | ✅ | ✅ | ✅ now |
| Qwen3-14B | 1 | circuit-tracer Qwen3 PLT | PLT | ✅ (≤14B) | ✅ | ✅ now |
| Qwen2.5-32B | 2 | `KokosDev/qwen25-32b-clt` | per-layer, raw torch | ❌ adapter | ✅ | after adapter |
| DeepSeek-R1-Distill-Qwen-1.5B | 2 | `EleutherAI/skip-transcoder-…-65k` | skip, sae fmt, **L0–14 only** | ❌ adapter | ✅ | after adapter (partial layers) |

**Tier-2 caveats (verified 2026-07-17):** `KokosDev` is raw PyTorch state-dicts with no
stated circuit-tracer integration and a name ("clt") that contradicts its own card
("one transcoder per layer") — confirm before relying on it. The DeepSeek skip-transcoder
loads via EleutherAI `sae` and covers only the first 15 MLPs, so its H4 graphs are
layer-limited. Both are worth including, but as **adapter-gated** targets.

Per-model status columns to track: **direction (H1) · jailbreak dataset (H2) · mediation
(H3) · attribution (H4)** — populated by the `results/<model>/<hypothesis>/` tree.

---

## §6 · Consolidation plan

| Stranded work | Lands in |
|---|---|
| `temp/gemma-vs-qwen-pipeline` — Qwen3-4B full run + comparison | `results/qwen3-4b/{h1,h3,h4}/`; comparison → `docs/reorg/` |
| `tejas-circuit-experiments` / `qwen_experiments` — SAE/QK/**linear probe** + causal variants | `src/refusal_lens/probes.py` + `experiments/h6_exploration/`; results → `results/<model>/` |
| `dataset-v2-verification` — behavioral-verification toolchain + negative result | `experiments/h2_jailbreak_validation/` (the control-filter is H2's core) |
| `tejas/dataset-10-classes` — dataset_v2 + 5 literature classes | `src/refusal_lens/attacks/` (framing/obfuscation/…) + `data/base_prompts/` |
| `add-refusal-direction` — precomputed r̂ `.pt` | `data/directions/gemma-3-4b/` |
| `dataset_v5` (current branch) | `src/refusal_lens/attacks/` + `experiments/h2_.../` |

**Branch cleanup:** delete `architectural_improvements`, `dev`, `visualization-figures`
(fully merged). Keep `temp/gemma-vs-qwen-pipeline`, `tejas-circuit-experiments`,
`qwen_experiments`, `dataset-v2-verification`, `tejas/dataset-10-classes`,
`add-refusal-direction` **until their unique work is migrated**, then delete. The absorbed
ancestry (`foundation`, `l15-refactor`, `emnlp-perm-edit`, `mshabana-rp-causal-exp`) can be
deleted once `jb-dataset-refined` merges to `main`.

---

## §7 · Sub-project roadmap (sequenced)

Each is a separate `brainstorm → spec → plan` cycle. Ordered by dependency.

```
P0  Model-agnostic core
      registry.py + ModelSpec (all 6 models) · models.py (hooks + chat template)
      consolidate directions.py · rename clt.py -> transcoders.py
      DoD: `--model X` works for a trivial forward pass; one config surface; 0 per-model forks
        |
        v
P1  Foundation experiments (transcoder-free; all 6 models)
      P1a  Attack subsystem + H2: attacks/ registry + per-model dataset engine
             DoD: data/jailbreaks/<model>.json + ASR report for each model
      P1b  H1 direction across models  -> appendix figure
      P1c  H3 mediation (correlation + causation/patch + sufficiency/induce)  <-- Georg's non-negotiable
             DoD: the H3 result reproduced on the new dataset, ≥3 models
        |
        v
P2  Transcoder migration + H4 attribution (regenerate graphs)
      P2a  Tier-1 models: re-run attribution under chosen transcoder
      P2b  Tier-2 adapters (KokosDev raw-torch, EleutherAI sae) -> ct ReplacementModel
        |
        v
P3  H5 method write-up  +  H6 exploration (multi-day; graph mining)

(supporting tracks, parallel: dataset consolidation §6 · branch cleanup §6 · CI/tests)
```

**Rationale for the order (defensible to Georg):** P1 *is* Georg's H1→H3 chain, the
prerequisite he says makes circuit tracing non-nonsensical. Attack generation (P1a) leads
P1 because H3 can't measure a jailbreak's r̂ magnitude without per-model jailbreaks. P2
(H4) waits on P1 and on the transcoder abstraction from P0. P3 is the open-ended meat.

---

## §8 · Open questions for Georg / to resolve before P2

1. **CLT-vs-PLT history.** Existing graphs were "true CLT," yet `config.py` /
   `clt.py` already point at a single-layer `mwhanna` repo. Was the migration *started*
   (config updated) but the **results not regenerated**? Confirm so P2 re-runs the right set.
2. **Tier-2 cost/benefit.** Are Qwen2.5-32B (32B GPU + a raw-torch adapter) and DeepSeek-1.5B
   (sae adapter, first-15-layers only) worth H4, or H1–H3 only for the paper's scope?
3. **H5 method statement.** Who authors the math for tracing an arbitrary residual
   direction, and to what rigor for the methods section?
4. **Dataset standardization.** Standardize on v5 attack classes + Tejas's v3/v4 classes as
   the unified `ATTACKS` registry? Any classes to drop as reviewer-fragile?
5. **H6 scope.** Is exploration in-scope for this reorg, or a follow-on once P0–P2 land?
6. **Larger-Qwen3 = 14B** confirmed (ct-native ceiling). Flag if a 32B Qwen3 is wanted (→ Tier 2).

---

## §9 · Success criteria (definition of done for the north star)

- Adding a model is **one `ModelSpec` entry** — no new folder, no forked script.
- Every hypothesis runs as `experiments/hN/run.py --model X`; no experiment hardcodes a model.
- **One** config surface (`registry.py`); the 3 old configs and 9 dicts are gone.
- H1 direction extraction exists in **one** place; `clt.py` misnomer is gone.
- `data/jailbreaks/<model>.json` exists for each in-scope model, gated by the H2 control-filter.
- `results/<model>/<hypothesis>/` populates a coverage matrix at a glance.
- Stranded branch work is migrated (§6); merged/stale branches deleted.
- Each `experiments/hN/README.md` answers the 5 methodology questions.

---

*This is the umbrella. Next step: the P0 sub-project brainstorm → spec, then implementation.
No code changes until P0's spec is approved.*
