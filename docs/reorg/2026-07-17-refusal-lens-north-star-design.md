# Refusal-Lens Reorganization — North-Star Design

**Status:** Draft for review · **Date:** 2026-07-17 (rev. 2026-07-19) · **Branch:** jb-dataset-refined
**Authors:** Mahmoud Shabana (with Tejas Dahiya) · **Mentor:** Georg Lange
**Companion:** `docs/reorg/2026-07-18-refusal-lens-review-addendum.md` (recommendations `R1…R34`). **This document is authoritative**; the addendum is the review record. R-numbers below cite the addendum item each decision absorbs.

---

## §1 · Purpose & guardrails

### Why this document exists
Georg has framed the paper as a chain of hypotheses (H1→H6, §2). Our code proves large
parts of this **for Gemma**, some for Qwen, and scatters the rest across ~11 branches and
two incompatible code organizations. This is the **umbrella plan** to reorganize so that:

1. Every hypothesis H1→H6 has **one obvious, model-agnostic home**.
2. Adding a model is **a registry entry, not a new folder**.
3. "Which experiment, on which model, is done — and can I *trust and re-run* it?" is
   answerable from the tree.
4. The move to **transcoders** is behind a single abstraction.

It is a **north star**, not an implementation plan. Each sub-project (§8) gets its own
`brainstorm → spec → plan` cycle. No code is written until a sub-project's spec is approved.

### What this revision adds (from the review addendum)
The first draft optimized for **organization** (where code lives) and under-weighted two
layers that a re-runnable paper needs:
- **Correctness-provenance of the attribution physics** — load-bearing, per-architecture,
  and in places *unverified* (placeholder configs, silent measurement bugs). §4.1, §4.6.
- **Reproducibility & packaging** — pinning, judge-label freezing, run manifests, repo
  hygiene, a viz module, a datasheet. §7.
Both are now first-class. The connective tissue is one bug-class: **metadata that lies
about what the code actually measured** (the placeholder configs, the "params-not-passed"
Bug 2, the viz layer recomputing metrics at render time). The registry *increases* that
risk by adding a config→call indirection, so §4.6 guards it.

### Guardrails (non-negotiable)
- **Humans own the science; tooling scaffolds (R31).** Experiment *design*, choice of
  *what to measure*, and *interpretation* are Mahmoud/Tejas/Georg's — never delegated to
  AI. This reorg only makes experiments easy to express, parameterize, run, and reproduce.
  Where a recommendation touches statistics or judging (R18, R22), it makes the *machinery*
  uniform; the metric/test choice stays with the researchers.
- **Every experiment records its reasoning (R32).** Each `experiments/hN/README.md` answers
  the 5 methodology questions (what / why / expected / results-vs-expectation / follow-up).
- **A prompt is a jailbreak only when proven (H2).** Harmful-on-attack **and** refuses the
  same content without it. Unqualified prompts are excluded.

---

## §2 · The research spine (H1→H6 as a dependency DAG)

| # | Hypothesis | Needs | Transcoder? | Paper role |
|---|---|---|---|---|
| **H1** | Refusal is a linear direction (r̂ = mean(harmful) − mean(harmless); ablate→comply, add→refuse). | model + residual hooks | ❌ | Reproduced in newer models → **appendix** |
| **H2** | Prompts genuinely jailbreak: harmful-on-attack **and** not-on-matched-control. Exclude the rest. | model + judge + **attack generators** | ❌ | Defines the dataset every later step consumes |
| **H3** | Jailbreaks mediated via the refusal direction. **(a)** correlation of r̂ magnitude; **(b)** add magnitude → patch; **(c)** subtract → induce. | H1 + H2 | ❌ | **The non-negotiable result** justifying circuit tracing |
| **H4** | Circuit-tracing the refusal direction (attribution graphs A_{s→R}). | H1–H3 + **transcoders** | ✅ | The main experiment |
| **H5** | Method: math for tracing an arbitrary residual direction. | — | ✅ (context) | Brief methods section |
| **H6** | Exploration: what the graphs reveal. | H4 | ✅ | The meat |

**The load-bearing seam:** H1–H3 need only **model + residual hooks** — no transcoders.
Only **H4+** needs them. The foundation experiments run on **all** models immediately; the
transcoder/model-tiering question (§5) gates **only** H4. The reorg is **not blocked** on
transcoder integration.

---

## §3 · Current-state assessment

*(Full read-only audit of `jb-dataset-refined` + a branch survey + the addendum's verified findings.)*

### In place (and good)
- A real library `src/refusal_lens/` (14 modules + `__init__`): `model_loader`,
  `refusal_directions` (H1), `attribution` (H4 primitive, already direction/layer-param),
  `clt` (transcoder wrapper), `sae`, `supernode_analyzer`, `data_loader`, etc.
- A model-agnostic judge package `refusal_llm_judge/` (litellm/vLLM, model as a string).
- Mature Gemma experiment code across H1/H3/H4 + a `dataset_v5` attack pipeline.

### Needs refactoring — the duplication disease
- **Per-model script forks, not parameterization.** `scripts/pipeline/` (Gemma) vs
  `scripts/pipeline_qwen/` (Qwen): 7 files byte-identical, the rest structurally identical.
  Adding Llama today = a **third** fork.
- **~9 copy-pasted two-model dicts** in `scripts/emnlp_perm_edit/*`; **three competing
  `config.py` surfaces** (unlinked from the package); **H1 in ≥3 unlinked implementations**.
- **Transcoder naming drift.** `clt.py` / `TRANSCODER_PATH` load **single-layer** affine
  transcoders (`mwhanna/…width_16k_l0_small_affine`) but are named "CLT."
- **Two incompatible code organizations** across branches: staged `scripts/pipeline{,_qwen}/`
  vs numbered `data/{tejas,qwen}_experiments/scripts/`.
- **Two attack lineages, unlinked.** Tejas's v3/v4 **literature techniques** —
  `nested_fiction, narrative_sandwich, m2s_multiturn, encoding_base64,
  translation_lowresource, expert_dialogue` (in `new_dataset_results/refusal_results/v3_judged.json`
  / `v4_judged.json`, which already carry `kind: control` records = the H2 criterion) — and
  Mahmoud's v5 (`gcg_per_prompt, many_shot_icl, refusal_suppression[_prefill]` in
  `dataset_v5.json`, Gemma-wired). Gemma-measured ASR already spans **0–60%** (m2s_multiturn
  60%, nested_fiction 36%, expert_dialogue 22%, narrative_sandwich 2%, encoding_base64 /
  translation_lowresource 0%) — proof that "which classes work" is per-model empirical.

### Needs verification — the provenance holes (addendum-verified 2026-07-19)
- **Placeholder physics (R1).** `scripts/pipeline_qwen/config.py:8` literally reads
  `UNVERIFIED PLACEHOLDERS: MEASUREMENT_LAYER, MEASUREMENT_POSITION, CAUSAL_LAYER` with
  `# TODO: verify via Stage 01 position sweep`. A coverage matrix must not mark Qwen's
  H1–H3 "done" while its measurement layer/position are guesses.
- **Silent measurement bugs (R6).** `MENTEE_NOTE_three_bugs.md` Bug 2: attribution was
  measured at post-stack/pos=-1 while metadata claimed L32/pos=-2. The registry's
  config→call indirection can reintroduce exactly this.
- **The vendored fork is referenced three inconsistent ways (R9).** `pyproject.toml` pins
  `circuit-tracer@refusal-lens-measurement-patch`; `.gitmodules` sets
  `branch = refusal-lens-multi-position-fix`; the gitlink pins commit `76c7af4b…`. Three
  references, two branch names — a `submodule update --remote` would drift.
- **No revision pins (R19).** `model_loader.py`, `clt.py`, configs load models/transcoders
  by name only — a silent upstream re-upload changes the numbers.
- **Judge unpinned/inconsistent (R16).** docstring uses `openrouter/anthropic/claude-haiku-4.5`;
  `judge.py:25 DEFAULT_MODEL="ollama_chat/qwen3.6:35b-a3b"`; neither version-pinned.
- **Repo hygiene (R23/R24/R25).** `.git` is **~4.0 GB** (11.8 MB `attribution_checkpoint.json`
  committed across 9 run-variant dirs; a 16.9 MB residuals `.pt`; 15.7 MB `v5_generations.json`).
  `README.md` is a 42-line stub with an empty Usage section and a garbled char in the pip
  line. There are 14 modules (not 16); `jailbreak_tracer.py` is unmapped and
  `test/test_neuronpedia_bridge.py` is an **orphaned test** (no `neuronpedia_bridge.py`).

### Stranded on branches (consolidate — §6)
- `temp/gemma-vs-qwen-pipeline` — canonical **Qwen3-4B full-pipeline run** + comparison.
- `tejas-circuit-experiments` → `qwen_experiments` — SAE/QK/**linear-probe** lineage + the
  original controlled 50-prompt dataset (under `data/*_experiments/scripts/`).
- `dataset-v2-verification` — the H2 behavioral-verification toolchain + a negative result.
- `tejas/dataset-10-classes`, `add-refusal-direction` — small data artifacts.
- `architectural_improvements`, `dev`, `visualization-figures` — fully merged → deletable.

---

## §4 · Target architecture (Approach B: library + hypothesis-organized experiments)

### Target tree
```
src/refusal_lens/
  registry.py         # MODELS: dict[name -> ModelSpec]   <-- the ONE source of truth (§4.1)
  models.py           # load + hook-manager + chat-template abstraction (absorbs pipeline/utils.format_prompt)
  directions.py       # H1 primitive: diff-in-means (consolidates the 3 impls; R2)
  steering.py         # H3 primitives: r-hat projection magnitude, add/subtract interventions
  attacks/            # attack GENERATORS + ATTACKS registry (unifies Tejas v3/v4 + v5) (§4.3)
    __init__.py       #   ATTACKS: dict[name -> AttackSpec]
    framing.py        #   nested_fiction, narrative_sandwich, expert_dialogue, refusal_suppression[_prefill]
    multiturn.py      #   m2s_multiturn
    obfuscation.py    #   encoding_base64, translation_lowresource
    many_shot.py      #   in-context; per-model comply-pool bootstrap
    gcg.py            #   per-tokenizer special-token exclusion + on-target optimization
  transcoders.py      # was clt.py — type-agnostic loader: PLT / CLT / skip, native(ct)+adapter (§4.5, R3)
  transcoder_adapters/ #  KokosDev raw-torch, EleutherAI sae  -> circuit-tracer ReplacementModel
  attribution.py      # H4 primitive: attribute_to_direction (kept)
  verification.py     # R6a param-reached + R2 direction-repro checks; R6b Σedges==direct_dot (§4.6)
  probes.py           # linear probes (absorbs tejas lineage)
  judge/              # was refusal_llm_judge/ (already agnostic); pinned + label-freezing (§7)
  viz/                # assembler + FastAPI ablation server (absorbs pipeline/05_frontend_patches) (§7, R26)
    viewer/           #   static circuit-tracer viewer assets + injected patches

experiments/
  h1_direction/  h2_jailbreak_validation/  h3_mediation/
  h4_attribution/  h5_method/  h6_exploration/     # each: run.py (--model) + README (5-Q methodology)

data/
  base_prompts/           # shared harmful/harmless bases
  jailbreaks/<model>.json # H2 output: validated "jailbreaks that work" per model + reports/<model>_asr.md
  directions/<model>/     # H1 output: per-layer r-hat

results/<model>/<hypothesis>/   # coverage matrix falls out of the tree; each run carries a manifest (R21)
docs/reorg/                     # this doc + the addendum + one spec per sub-project
```

### §4.1 · Model registry — the one source of truth (R1)
Replaces the 3 config surfaces + 9 dicts. `ModelSpec` carries the **full physics surface,
each field tagged with provenance** so a placeholder can never be mistaken for a verified value:

```
ModelSpec:
  name, hf_id, hf_revision            # R19: pin the exact upstream revision
  family, n_layers, d_model
  causal_layer   {value, provenance}  # Gemma L15 (verified); Qwen L18 (PLACEHOLDER)   — intervention point
  separation_layer {value, provenance}# Gemma L32 (verified)                            — distinct from causal
  measurement_positions [ {pos, separation} ]   # anchored SET, not a scalar: Gemma [-5,-3,-2], Qwen [-5,-3,-1]
  direction_position   {value, provenance}      # -2 (Gemma) vs -1 (Qwen); cos(pos=-2,pos=-5) ≈ -0.80 → empirical
  hook_resolution                     # arch-specific: Gemma-3's 4-RMSNorm block resolves hook_resid_post via
                                      #   the NEXT layer's input_layernorm — needs a per-arch hooks sub-spec
  chat_template_kwargs                # e.g. Qwen enable_thinking=False ("load-bearing"; R4)
  transcoder {repo, revision, type(plt|clt|skip), backend, source(native|adapter)}  # nullable — H1-H3 don't need it
  tier                                # 1 (ct-native) | 2 (adapter-gated)
```
**`provenance ∈ {verified-by-sweep, seeded-placeholder}` gates the coverage matrix (§5):**
H1–H3 for a model are not "done" while any physics field is a placeholder.

### §4.2 · Library consolidation
- **R2** — fold the ≥3 H1 implementations into `directions.py`; stages **import the
  package**. Preserve the exact extraction point (`hidden_states[L+1]`) and float64
  accumulation (Bug 3 shows the extraction point is load-bearing); add a test asserting the
  consolidated function reproduces the checked-in reference direction **bit-for-bit**.
- **R3** — rename `clt.py` → `transcoders.py`; drop "CLT" from docstrings where the artifact
  is per-layer affine.
- **R4** — lift `pipeline/utils.py:format_prompt` into `models.py`; Qwen's
  `enable_thinking=False` becomes a `ModelSpec` field.

### §4.3 · Attack-generation subsystem (the dataset engine)
Unifies both lineages under one `ATTACKS` registry (categories: *framing / competing-
objectives*, *multi-turn*, *obfuscation*, *in-context*, *optimization*). Classes: Tejas
`nested_fiction, narrative_sandwich, expert_dialogue, m2s_multiturn, encoding_base64,
translation_lowresource` (v3/v4) + v5 `refusal_suppression[_prefill], many_shot_icl,
gcg_per_prompt`. Source data: `v3_judged.json` / `v4_judged.json` (with `kind:
attack|control`) + `dataset_v5.json`.

**Orchestration is H2 (R12–R15).** Per model: render/optimize attacks → generate → judge →
**harvest COMPLY outputs → build many-shot pool → gen/judge** (bootstrap ordering) → apply
one *identical* control-filter across both lineages → write `data/jailbreaks/<model>.json` +
ASR report. Generators take `(base_prompts, ModelSpec)`; `gcg` optimizes on the target model
with per-tokenizer special-token exclusion; `many_shot` consumes the model's own comply
pool. **ASR is per-model, per-class empirical output — the human decides which classes to
keep; the code only measures (R15).**

### §4.4 · Hypothesis experiments
Each `experiments/hN/run.py` is a thin `--model` CLI over the library + a methodology
README. H2's output feeds H3 and H4. No experiment hardcodes a model.

### §4.5 · Transcoders & the vendored fork
- **R8 — resolve CLT-vs-PLT provenance *before* P0 freezes the interface** (§9.1). If the
  paper figures used a true CLT while `clt.py` loads per-layer affine transcoders, `type:
  plt|clt|skip` is either a live inconsistency or a dropped axis — the answer determines
  whether the abstraction must genuinely support CLT.
- **R9 — pin the fork to ONE immutable reference.** Reconcile the three references (§3) to a
  single pinned commit (`76c7af4b`), remove the `.gitmodules branch =` line, and tag/mirror
  the commit (it holds the load-bearing `measurement_hook="hook_resid_post"` patch in a
  personal fork). Add a coverage column stricter than "ct-native?": **"residual-stream
  measurement patch verified for this arch?"**
- **R10 — H4 must be *re-run* under the chosen transcoder per model** (the draft's graphs
  were "true CLT"; §9.1). Real P2 work; report baseline subtraction correctly per model
  (`Σ edges = direct_dot − baseline`, non-zero at intermediate layers).
- **R11 — Tier-2 adapters are unverified** (`KokosDev` raw torch; EleutherAI sae, first-15-
  layers). Convert into a `ReplacementModel`, verify before relying, likely H1–H3-only
  unless H4 is explicitly scoped for them.

### §4.6 · Verification harness (R6/R7)
Because all three mentor bugs were *silent*, correctness must be asserted, not assumed:
- **R6a (P0-blocking):** assert the ModelSpec's layer/position/hook **actually reach the
  `attribute()` call** (not just the recorded metadata) + **R2's bit-for-bit direction
  reproduction** (H1, checkable now on Gemma, transcoder-free). This is what makes the
  registry safe.
- **R6b (P2-gated):** the full `Σ(edges) + baseline == direct_dot` invariant, with a golden
  reference-values file per model — lands with H4, since it needs a graph to check.
- **R7:** land R6a in `test/` (or extend `testpaths` beyond the package) so CI runs it on
  every PR; R6b joins once H4 exists.

---

## §5 · Model fleet & coverage matrix

**H1–H3 on all six** (transcoder-free); **H4 splits by transcoder availability.**

| Model | Tier | Transcoder | Type | ct-native? | H1–H3 | H4 |
|---|---|---|---|---|---|---|
| Gemma-3-4B | 1 | `mwhanna/gemma-scope-2-4b-it` | PLT | ✅ | ✅ | ✅ now |
| Qwen3-4B | 1 | `mwhanna/qwen3-4b-transcoders` | PLT | ✅ | ⚠️ physics placeholder | ✅ now |
| Llama-3.2-1B | 1 | circuit-tracer default | PLT/CLT | ✅ | ✅ | ✅ now |
| Qwen3-14B | 1 | circuit-tracer Qwen3 PLT | PLT | ✅ (≤14B) | ✅ | ✅ now |
| Qwen2.5-32B | 2 | `KokosDev/qwen25-32b-clt` | per-layer, raw torch | ❌ adapter | ✅ | after adapter |
| DeepSeek-R1-Distill-Qwen-1.5B | 2 | `EleutherAI/skip-transcoder-…-65k` | skip, sae fmt, **L0–14 only** | ❌ adapter | ✅ | after adapter (partial) |

Status columns per model — **each gated on provenance, not just existence**: *direction (H1)
· jailbreak dataset (H2) · mediation (H3) · attribution (H4) · physics verified-not-placeholder
(R1) · measurement-patch verified (R9)*. Qwen3-4B's H1–H3 stay ⚠️ until its
layer/position/causal fields are swept off "UNVERIFIED PLACEHOLDERS."

**Tier-2 caveats:** `KokosDev` is raw state-dicts whose "clt" name contradicts its card
("one per layer"); the DeepSeek skip-transcoder covers only the first 15 MLPs. Both are
adapter-gated.

---

## §6 · Consolidation plan

| Stranded work | Lands in |
|---|---|
| `temp/gemma-vs-qwen-pipeline` — Qwen3-4B run + comparison | `results/qwen3-4b/{h1,h3,h4}/`; comparison → `docs/reorg/` |
| `tejas-circuit-experiments` / `qwen_experiments` — SAE/QK/**probe** | `src/refusal_lens/probes.py` + `experiments/h6_exploration/` |
| `dataset-v2-verification` — behavioral toolchain + negative result | `experiments/h2_jailbreak_validation/` |
| `tejas/dataset-10-classes` — dataset_v2 + 5 literature classes | `src/refusal_lens/attacks/` (framing/obfuscation/…) + `data/base_prompts/` |
| `add-refusal-direction` — precomputed r̂ `.pt` | `data/directions/gemma-3-4b/` |
| `dataset_v5` (current branch) | `src/refusal_lens/attacks/` + `experiments/h2_.../` |
| `pipeline/05_frontend_patches`, `ablation_server.py` | `src/refusal_lens/viz/` (§7, R26) |

**R30 — migrate before deleting; preserve the hard-won fixes** (Bugs 1–3; the L32→L15
attribution-layer decision). Delete merged/stale branches
(`architectural_improvements`, `dev`, `visualization-figures`) only after confirming nothing
unique remains. Absorbed ancestry (`foundation`, `l15-refactor`, `emnlp-perm-edit`,
`mshabana-rp-causal-exp`) deletes once `jb-dataset-refined` merges to `main`.

---

## §7 · Provenance, reproducibility & packaging

The layer the first draft under-covered. Grouped by concern; each item keyed to the addendum.

### Judge — the labels *define* the H2 dataset H3/H4 consume
- **R16 — pin ONE canonical judge to a specific version.** Resolve the docstring/`DEFAULT_MODEL`
  split; version-pin whichever is chosen.
- **R17 — freeze the judge's outputs as the dataset of record.** Cache verdicts into a
  versioned dataset; downstream reads frozen labels so they don't move when the judge does.
- **R18 — calibrate against human labels on a sample.** *Researcher-owned:* the metric,
  sample, and sufficiency threshold are yours; the code runs the comparison and reports it.

### Reproducibility infrastructure
- **R19 — pin every external HF artifact revision** (`hf_revision`/`transcoder_revision` in
  the registry; the pushed-graphs dataset too).
- **R20 — seed policy + determinism.** Global seed recorded per run; pin decoding params;
  document the canonical backend (`transformerlens`, per config) + device — the mentor note
  flags MPS-vs-CUDA and nnsight-vs-transformerlens as behavior-affecting.
- **R21 — per-run manifest** in `results/<model>/…`: torch/CUDA/transformerlens versions,
  device, seed, artifact revisions. The `Dockerfile` is the reproduction unit.
- **R22 — one shared statistical utility** (n, CIs, multiple-comparison handling) applied
  uniformly across H1/H3/H4. *Researcher-owned:* the choice of test stays with you.

### Clean, self-contained repository
- **R23 — artifact & history policy.** `.git` ≈ 4.0 GB; large/regenerable artifacts move to
  the HF dataset or git-LFS/DVC; checkpoints are never versioned; a one-time history rewrite
  is warranted. The tree states what is versioned-in-git vs stored-externally-and-pinned.
- **R24 — `REPRODUCE.md` + a figure→code→data→revision provenance map** (README is a stub
  with a typo; nothing links `Attribution_Circuits_to_Refusal_Direction.pdf` figures to the
  script+commit+data that made them). Deterministic path from pinned inputs to each artifact.
- **R25 — complete the module inventory.** 14 modules (15 w/ `__init__`); map
  `jailbreak_tracer.py`; fold `attribution_pipeline.py`; resolve the orphaned
  `test_neuronpedia_bridge.py` (restore the module or delete the test).

### Attribution dashboard / viz (gated on P2)
- **R26 — promote it out of the Gemma `pipeline/` fork** to `src/refusal_lens/viz/` (Python
  assembler + FastAPI ablation server + static viewer/patches). The assembler takes
  `(model, run_dir)` off `ModelSpec` + `results/<model>/h4/` — not hardcoded Gemma.
- **R27 — presentation-only.** Node classification / supernode grouping / displayed metrics
  are **library** functions shared with the experiments; the assembler consumes their output
  and never recomputes (else the viz layer drifts from H4 — the metadata-vs-reality risk again).
- **R28 — do not rebuild the renderer.** Keep circuit-tracer's viewer + patch-injection;
  absorb `FRONTEND_ABLATION_PLAN.md` rather than adding a parallel dashboard.
- **R29 — optional rigor view** (researcher's call): a per-graph panel surfacing the R6b
  residual so a browser can see whether a graph passes the invariant.

### Governance
- **R33 — responsible-release datasheet for the jailbreak corpus.** `dataset_v5.json` holds
  working harmful completions; a shareable safety repo needs a datasheet (method, intended
  use, harmful-content warning), a gated-vs-open decision, and license clarity for
  redistributing model-generated content + the vendored fork.

---

## §8 · Sub-project roadmap (sequenced)

Each is a separate `brainstorm → spec → plan` cycle. R-items keyed per phase.

```
P0  Model-agnostic core            [R1 registry+provenance · R2 directions+bit-for-bit · R3 rename ·
                                     R4 chat template · R5 DoD · R6a param-reached test · R7 CI ·
                                     R8 resolve CLT/PLT · R9 pin fork]
      DoD: `--model X` forward pass; one config surface; 0 forks; R6a green in CI;
           no ModelSpec physics field left as placeholder for an in-scope model
        |
        v
P1  Foundation experiments (transcoder-free; all 6 models)
      P1a  Attacks + H2   [R12–R15 · judge R16–R18 in parallel]  -> data/jailbreaks/<model>.json
      P1b  H1 direction across models  -> appendix
      P1c  H3 mediation (correlation + patch + induce)   <-- Georg's non-negotiable
      (parallel supporting: repro infra R19–R22 · repo hygiene R23–R25 · governance R33)
        |
        v
P2  Transcoder migration + H4 attribution (regenerate graphs)
      P2a Tier-1 re-run [R10] · P2b Tier-2 adapters [R11] · R6b golden invariant · viz [R26–R29]
        |
        v
P3  H5 method write-up  +  H6 exploration
```
**Rationale (defensible to Georg):** P1 *is* the H1→H3 chain he says makes tracing
non-nonsensical; attacks (P1a) lead because H3 needs per-model jailbreaks. R6a/R9/R8 sit in
P0 because later correctness silently inherits them; R6b waits for the graphs it checks.

---

## §9 · Open questions for Georg

1. **CLT-vs-PLT history (R8, §4.5).** Existing graphs were "true CLT," yet `config.py`/`clt.py`
   already point at a single-layer `mwhanna` repo. Was the migration config-only (results
   never regenerated)? This freezes the transcoder interface and scopes P2's re-run.
2. **Tier-2 cost/benefit (R11).** Qwen2.5-32B (32B GPU + raw-torch adapter) and DeepSeek-1.5B
   (sae adapter, first-15-layers) worth H4, or H1–H3 only?
3. **H5 method statement.** Who authors the math, to what rigor?
4. **Dataset standardization (R33).** Unify v5 + Tejas v3/v4 as the `ATTACKS` registry; any
   classes to drop as reviewer-fragile? Release gating for the harmful corpus.
5. **H6 scope.** In this reorg, or a follow-on after P0–P2?
6. **Larger-Qwen3 = 14B** confirmed (ct-native ceiling); flag if a 32B Qwen3 is wanted (→ Tier 2).

---

## §10 · Success criteria (definition of done)

- Adding a model is **one `ModelSpec` entry** with **verified (not placeholder) physics** —
  no new folder, no forked script.
- Every hypothesis runs as `experiments/hN/run.py --model X`; no experiment hardcodes a model.
- **One** config surface; H1 extraction in one place; `clt.py` misnomer gone.
- The **R6a methodology invariant passes per model in CI**; R6b passes once H4 exists.
- **Every external artifact and the vendored fork are pinned to immutable revisions**; the
  judge is pinned and its labels frozen; each result carries a run manifest.
- `data/jailbreaks/<model>.json` exists per in-scope model, gated by the H2 control-filter.
- Large/regenerable artifacts live outside git history; `.git` is lean.
- `results/<model>/<hypothesis>/` populates a coverage matrix; stranded branches migrated,
  merged/stale branches deleted; every module mapped.
- A **reproduction guide** gives a deterministic path from pinned inputs to every paper
  figure/table. At that point the repo is working, clean, reproducible, and as self-contained
  as the external dependencies allow — final claims rest on a foundation a reviewer can re-run.

---

*Authoritative umbrella; the addendum (`…-review-addendum.md`, R1–R34) is the review record.
Next step: the P0 sub-project brainstorm → spec. No code until P0's spec is approved.*
