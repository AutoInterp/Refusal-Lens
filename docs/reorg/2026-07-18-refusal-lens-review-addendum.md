# Refusal-Lens Reorg — Review Addendum & Consolidated Recommendations

**Status:** Review of `jb-dataset-refined` north-star · **Companion to:** `docs/reorg/` north-star design
**Audience:** the session executing the reorg · **Nature:** additive — the north-star's Approach B (library + hypothesis-organized experiments + registry) is sound and should be kept as-is; this sharpens it and fills gaps.

---

## §0 · How to use this / governing principles

This is not a replacement plan. The north-star's directory layout, the H1→H6 spine, the load-bearing seam (H1–H3 are transcoder-free; only H4+ needs transcoders), and the sub-project sequencing are all correct. The recommendations below are keyed `R1…R34` and tagged with the plan phase they belong to.

Two principles govern every item:

- **Humans own the science; tooling scaffolds.** The reorg only makes experiments easy to *express, parameterize, run, and reproduce*. It never decides experiment design, what to measure, or how to interpret results. Where a recommendation touches statistics or judging, it concerns making the machinery uniform and reproducible — not choosing the test or the metric.
- **The plan is architecture-complete but under-covers two layers:** (a) the *correctness-provenance* of the attribution physics, which is load-bearing, per-architecture, and in places unverified; and (b) the *reproducibility-and-packaging* layer that turns organized code into a re-runnable artifact. Most gaps below live in those two layers.

A note on "self-contained": true self-containment is impossible (you cannot vendor Gemma's weights). The achievable target is **every external dependency pinned to an immutable revision and documented**, plus the patched fork made durable.

---

## §1 · Start here — P0-blocking items

Do these before or during P0, because later work silently inherits their correctness:

- **R6a** (P0 half) methodology invariant — assert the ModelSpec params actually reach the `attribute()` call + R2 bit-for-bit direction reproduction (the full `Σedges==direct_dot` check is **R6b, P2**; it needs an attribution graph to exist, so it cannot gate P0)
- **R1** ModelSpec extended with verified-vs-placeholder provenance + the full per-arch physics surface
- **R2** consolidate the ≥3 direction implementations, preserving extraction point + dtype
- **R8** resolve CLT-vs-PLT provenance before freezing the transcoder interface
- **R9** pin + freeze the vendored `circuit-tracer` fork and add a per-arch "measurement patch verified" gate

Everything else can proceed as parallel supporting tracks.

---

## §2 · Registry & model-agnostic core (P0)

**R1 — Extend `ModelSpec` beyond the north-star's field list.** The physics that actually matters is bigger and more dangerous than `measurement_layer / position / hook`. Add:
- **Provenance per physics field** (`verified-by-sweep` vs `seeded-placeholder`). Evidence: `scripts/pipeline_qwen/config.py` opens with "UNVERIFIED PLACEHOLDERS" and carries `# TODO: verify via Stage 01 position sweep` / `01b_layer_sweep.py` on `MEASUREMENT_LAYER/POSITION/CAUSAL_LAYER/BEST_SEPARATION_LAYER`. The coverage matrix must not mark a model's H1–H3 "done" while its measurement layer/position are still placeholders.
- **`measurement_position` as a template-anchored *set* with separations**, not a scalar. Gemma uses `[-5,-3,-2]`, Qwen `[-5,-3,-1]`; direction position is `-2` vs `-1`; and `cos(L15 pos=-2, pos=-5) = -0.80` — the direction is strongly position-rotation-sensitive, so the choice is empirical per model.
- **Causal layer and separation layer as distinct fields, each with rationale.** Gemma is L15-causal / L32-separation; attribute at the causally-effective layer (the config already moved `MEASUREMENT_LAYER` 32→15). Every new model needs both measured.
- **Architecture-specific hook resolution.** Gemma-3's 4-RMSNorm block resolves `hook_resid_post` via the *next* layer's `input_layernorm`. This is not a scalar field — it needs an arch-hooks sub-spec, and the vendored fork must have a verified mapping per architecture (Qwen/Llama block structures differ).
- **`hf_revision` and `transcoder_revision`** (see R19).

**R2 — Consolidate the ≥3 H1 direction implementations into one `directions.py`.** The pipeline/emnlp stages currently re-implement or `import config` instead of importing the package. Fold them in, and have the stages import the package. **Preserve the exact extraction point (`hidden_states[L+1]`) and the float64 accumulation** — the mentor note (Bug 3) shows the extraction point is load-bearing. Add a test asserting the consolidated function reproduces the checked-in reference direction bit-for-bit.

**R3 — Rename `clt.py` → `transcoders.py`** and drop "CLT" from docstrings where the loaded artifact is per-layer affine (`…width_16k_l0_small_affine`). The name currently lies about the substance.

**R4 — Lift `pipeline/utils.py:format_prompt` into `models.py`** as the `ModelSpec`-driven chat-template abstraction. Qwen's `enable_thinking=False` is load-bearing and must be a registry field.

**R5 — DoD for P0:** `--model X` runs a trivial forward pass; one config surface (`registry.py`); zero per-model forks.

---

## §3 · Verification harness (P0 — the item that makes the registry safe)

**R6 — Ship the methodology invariant as a regression test, wired into CI. Split by phase (this was over-scoped as wholly P0-blocking).** All three mentor-flagged bugs were *silent* (plausible numbers at the wrong point in the network). The registry adds indirection between config and the `attribute()` call — exactly Bug 2's failure mode (metadata claimed L32/pos=-2 while the call defaulted to post-stack/pos=-1). Two assertions, two phases:

- **R6a (P0-blocking):** assert the attribution was *actually* computed at the ModelSpec's layer/position/hook — that the params **reached the call**, not just the metadata (catches the Bug-2 class, which the registry's config→call indirection reintroduces). Pair it with **R2's bit-for-bit direction reproduction** (H1, checkable now on Gemma with no transcoder). This is what makes the registry safe to use in P0.
- **R6b (P2-gated):** the full `Σ(edges) + baseline == direct_dot` invariant to N decimals (catches Bug-1/Bug-3-class basis mismatches). This *requires an attribution graph to exist*, which does not happen until H4 (P2). Check in a golden reference-values file per model at that point.

**R7 — Bring the science under the CI-guarded test path.** CI (`ci.yml`) already runs ruff + pytest, and CD (`cd.yml`) builds + publishes to *test*-PyPI on release — but `testpaths=["test"]` covers only the package, so the attribution physics in `scripts/` is unguarded. Land R6a in `test/` (or extend `testpaths`) so it runs on every PR; R6b joins once H4 exists.

---

## §4 · Transcoders & the vendored fork (P0 interface; P2 execution)

**R8 — Resolve CLT-vs-PLT provenance *before* P0 freezes the transcoder interface** (north-star §8.1). `clt.py` loads per-layer affine transcoders while the draft's graphs were "true CLT." If the paper figures and current code use different transcoders, then `type: plt|clt|skip` is either a phantom axis you've dropped or a live inconsistency. The decision determines whether the abstraction must genuinely support CLT or only PLT+skip — i.e. it changes the interface.

**R9 — Elevate `vendor/circuit-tracer` to a first-class, per-arch-gated dependency.** The `measurement_hook="hook_resid_post"` fix lives here. The gitlink pins commit `76c7af4b…` (good), but `.gitmodules` also sets `branch = refusal-lens-multi-position-fix` — freeze/remove that to prevent `submodule update --remote` drift — and tag or mirror the commit, since the load-bearing patch sits in a personal fork that could be force-pushed away. Add a coverage-matrix column stricter than "ct-native?": **"residual-stream measurement patch verified for this arch?"** A model can be ct-native for vanilla attribution yet silently fall back to the wrong measurement point without the patch.

**R10 — H4 must be *re-run* under the chosen transcoder per model (regenerate graphs).** This is real P2 work, not a rename. Handle and report the baseline subtraction correctly per model (`Σ edges = direct_dot − baseline`, non-zero at intermediate layers).

**R11 — Tier-2 adapters are unverified.** `KokosDev/qwen25-32b-clt` is raw torch state-dicts whose "clt" name contradicts its own card ("one transcoder per layer"); the EleutherAI skip-transcoder loads via `sae` and covers only the first 15 MLPs (layer-limited H4). Convert both into a circuit-tracer `ReplacementModel`, but verify before relying, and treat them as adapter-gated targets — likely H1–H3-only unless H4 is explicitly in scope for them.

---

## §5 · Attack subsystem & H2 dataset engine (P1a)

**R12 — Unify both lineages under one `ATTACKS` registry.** Categories: framing/competing-objectives, multi-turn, obfuscation, in-context, optimization. Sources: Tejas v3/v4 (`nested_fiction, narrative_sandwich, expert_dialogue, m2s_multiturn, encoding_base64, translation_lowresource`) + Mahmoud v5 (`gcg_per_prompt, many_shot_icl, refusal_suppression[_prefill]`).

**R13 — Apply a *single* control-filter (H2) definition identically across both lineages.** Tejas's harness already carries `kind: control`; v5 is Gemma-wired. Without a unified definition, the jailbreak criterion silently differs by class-of-origin. Orchestration ordering (bootstrap): gen → judge → harvest COMPLY outputs → build many-shot pool → gen/judge → apply control-filter → write `data/jailbreaks/<model>.json` + ASR report.

**R14 — Generators take `(base_prompts, ModelSpec)`.** `gcg` optimizes on the *target* model with per-tokenizer special-token exclusion; `many_shot` consumes the model's *own* comply pool; framing/obfuscation are CPU prompt/chat-template work.

**R15 — ASR is per-model, per-class empirical output.** The human decides which classes to keep; the code only measures. (Gemma spans 0–60% across classes; many-shot was ~0/50 — the latest commit is literally "faithful many-shot re-run = 0/50, robust true negative.")

---

## §6 · The judge — the single biggest reproducibility hole (P1a / supporting)

The judge's labels *define* the H2 dataset that H3 and H4 consume, so the validity of every downstream claim inherits the judge.

**R16 — Pin ONE canonical judge to a specific version.** Today it is inconsistent: the docstring example uses `openrouter/anthropic/claude-haiku-4.5` (API) while `DEFAULT_MODEL` is `ollama_chat/qwen3.6:35b-a3b` (local). Neither is version-pinned.

**R17 — Freeze and commit the judge's outputs as the dataset of record.** Labels must not move when the judge changes or disappears. Cache verdicts into a versioned dataset; downstream reads the frozen labels.

**R18 — Add a judge-calibration check against human labels on a sample.** The dataset's scientific validity rests on judge reliability; the reorg runs the comparison and reports agreement. **Researcher-owned:** the calibration metric, the sample, and whether agreement is *sufficient* are Mahmoud/Tejas/Georg calls — the code only measures.

---

## §7 · Reproducibility infrastructure (supporting tracks)

**R19 — Pin every external HF artifact revision.** `model_loader.py`, `clt.py`, and the configs load models, transcoders, and the pushed graphs dataset by name only — no `revision=<sha>` anywhere. A silent re-upload upstream changes your numbers. Carry `hf_revision`/`transcoder_revision` in the registry (R1).

**R20 — Seed policy + determinism.** Diff-in-means sampling, GCG, the many-shot pool, and generation temperature all inject randomness; the mentor note flags MPS-vs-CUDA and nnsight-vs-transformerlens as behavior-affecting. Set a global seed recorded per run, pin decoding params (temp/top-p), and document the canonical backend + device (results are only bit-comparable within one condition).

**R21 — Per-run manifest written into `results/<model>/…`.** Record torch/CUDA/transformerlens versions, device, seed, and artifact revisions. Make the `Dockerfile` the reproduction unit and reference it from the reproduction guide (R24). The tree records *which* experiment ran, not *under what conditions*.

**R22 — Shared statistical protocol in the library.** `02b_statistical_analysis.py` exists but is not a shared convention. Provide one stats utility (n, confidence intervals, multiple-comparison handling) applied uniformly across H1/H3/H4 so claims report comparably. The choice of test stays with the researchers.

---

## §8 · Clean & self-contained repository (supporting tracks)

**R23 — Artifact and history policy.** `.git` is **~4.0 GB** (verified — far past the point where a one-time history rewrite is optional), with multi-MB outputs committed across **9** run variants — a 16.9 MB residuals `.pt`, 15.7 MB `v5_generations.json`, and the 11.8 MB `attribution_checkpoint.json` blob **committed in 9 run-variant directories** alongside `attribution_results.json` (intermediate checkpoints should never be versioned). Large/regenerable artifacts go to the HF dataset (already used for graphs) or git-LFS/DVC; checkpoints are never versioned; a one-time history rewrite is likely needed. The target tree must state what is versioned-in-git vs stored-externally-and-pinned.

**R24 — Reproduction guide + figure→code→data provenance map.** `README.md` is a 42-line stub with an empty Usage section (and a typo in the pip line), while `Attribution_Circuits_to_Refusal_Direction.pdf` sits in the repo with nothing linking a figure/table to the exact script + commit + data revision that produced it. Add `REPRODUCE.md` or `make table3`-style targets: a deterministic path from pinned inputs to each artifact.

**R25 — Complete the module inventory.** Every existing module needs an explicit home in the target tree. There are **14 modules** (15 with `__init__.py`), not 16 — `jailbreak_tracer.py` (13 KB) is unmapped and `attribution_pipeline.py` (~1.3 KB) is a thin wrapper worth folding in. Also: `test/test_neuronpedia_bridge.py` exists with **no** corresponding `neuronpedia_bridge.py` module — an orphaned test to resolve (restore the module or delete the test).

---

## §9 · Attribution dashboard / viz module (supporting track, gated on P2)

**R26 — It exists but is trapped in the Gemma `pipeline/` fork; promote it to one model-agnostic module.** The viewer is circuit-tracer's graph frontend; the "views and filters" are injected patches in `scripts/pipeline/05_frontend_patches/` (`compare.html`, `compare_multi.html`, `trace.html`, `feature-cart.js`, `subcircuit-panel.js`, `overlap-annotate.js`, `trace-highlight`, `compact-mode`); `assemble_trace_frontend.py` / `assemble_compare_frontend.py` compose viewer + patches + data; `ablation_server.py` (FastAPI) gives live select→intervene→compare. The plumbing (`05_frontend_patches`, `ablation_server`, fetch/push graph) is duplicated across `pipeline/` and `pipeline_qwen/`, and the trace-assembly is Gemma-only. Lift into `src/refusal_lens/viz/` (Python assembler + FastAPI server) plus a static `viewer/` asset dir. The assembler takes `(model, run_dir)` off `ModelSpec` + `results/<model>/h4/` — not hardcoded "Gemma-complement / the 4 flips."

**R27 — Presentation-only constraint (rigor).** `assemble_trace_frontend` currently runs `trace_classifier` and bakes `rl_trace_class` onto nodes at assembly time — a second code path that can drift from what H4/H6 computed (the "metadata vs reality" risk again, in the viz layer). Node classification, supernode grouping, and any displayed metric must be *library* functions shared with the experiments; the assembler consumes their output and never recomputes.

**R28 — Do not rebuild the renderer or add a third viz surface.** Keep circuit-tracer's viewer + patch-injection as the substrate (no from-scratch Streamlit/React app). Absorb `FRONTEND_ABLATION_PLAN.md`'s target architecture rather than superseding it, and avoid a parallel dashboard alongside the existing patches and `figures/`.

**R29 — Optional rigor view (researcher's call):** a per-graph panel surfacing the methodology residual (`Σ edges + baseline` vs `direct_dot`) so anyone browsing a graph can see whether it passes the R6 invariant. The ablation server already has the machinery.

---

## §10 · Consolidation & governance

**R30 — Migrate stranded branch work per north-star §6 before deleting, preserving the hard-won fixes.** The §6 table is good; the executing session must ensure migration does not lose the mentor-flagged fixes (Bugs 1–3) or the verified physics decisions (the L32→L15 attribution-layer move). Delete merged/stale branches (`architectural_improvements`, `dev`, `visualization-figures`) only after confirming nothing unique remains.

**R31 — Keep the "humans own the science" guardrail explicit for the executing session.** It must not drift into making scientific decisions (which layer/position is correct, which classes count, how to interpret a graph). Those are surfaced for Mahmoud/Tejas/Georg to decide; the code makes them easy to express and measure.

**R32 — Every `experiments/hN/` carries the 5-question methodology README** (already in the plan): what/why/expected/results/follow-up. This is where human reasoning lives; code is downstream of it.

**R33 — Responsible-release / data governance for the jailbreak corpus.** `dataset_v5.json` contains working harmful completions. A shareable safety repo needs a datasheet (generation method, intended research use, harmful-content warning), a gated-vs-open release decision, and license clarity for redistributing model-generated content plus the vendored fork.

---

## §11 · Sequencing & definition of done

**Sequencing.** P0-blocking: **R6a**, R1, R2, R8, R9 (plus R3, R4). Parallel supporting tracks that can start immediately: the judge (R16–R18), reproducibility infra (R19–R22), storage + reproduction guide (R23–R25), and governance (R33). P2-gated: R10, R11, **R6b**, and the viz module (R26–R29). Per the north-star's own rule, no code is written until each sub-project's `brainstorm → spec → plan` cycle is approved.

**"Done" means:** adding a model is one `ModelSpec` entry with verified (not placeholder) physics; every hypothesis runs as `experiments/hN/run.py --model X` with no hardcoded model; the methodology invariant passes per model in CI; every external artifact and the vendored fork are pinned to immutable revisions; the judge is pinned and its labels frozen; each result carries a run manifest; large/regenerable artifacts live outside git history; and a reproduction guide gives a deterministic path from pinned inputs to every figure and table in the paper. At that point the repository is working, clean, reproducible, and as self-contained as the external dependencies allow — and the final claims rest on a foundation that a reviewer (or future-you) can re-run and trust.