# Jailbreak-as-Refusal-Suppression: Manual Circuit-Tracing Plan

**Date:** 2026-06-28
**Source:** Weekly check-in with Georg, Mon Jun 22 2026 (`docs/meeting_notes.txt`, esp. 33:00–end)
**Scope:** Gemma-**complement** (#443 zeroed) + **Qwen** attribution graphs only. Ignore the
full (yellow) and outlier (purple) columns — both compute attribution toward the inert #443
direction and are "broken" for this purpose.

---

## The reframed research question

Georg is **not** primarily interested in cataloguing refusal features. The L13–L15 features that
fire on the `model` token and are always followed by a refusal are the *known/easy* part — he
called them "cursed" and "the first layer we may not be super interested in."

What he wants:

> "I'm mostly interested in how **non-refusal features will eventually compute the refusal
> direction**."

Worked example he gave: a role-play jailbreak activates **role-play features** (not refusal-
associated per se) → these must somehow **suppress the refusal direction**. The goal is to see
*how exactly* that suppression happens in the circuit.

Reframe (carries forward, supersedes the batch-16 magnitude/ablation framing):
**a jailbreak = suppression of the refusal direction.** The central deliverable is a **diff of
attribution graphs between a jailbroken prompt and its matched non-jailbroken prompt.**

### The binary-search question to answer first

For a jailbroken vs. non-jailbroken prompt pair, which is true?

- **(a) Suppression of positives** — the jailbreak *reduces* the features that feed **positively**
  into the refusal direction, or
- **(b) Amplification of negatives** — those positive features stay roughly the same, but the
  jailbreak *increases* the "don't-refuse" features that feed **negatively** into the refusal
  direction.

Confirmed live by Georg: **the graphs already contain the negatively-associated features**, so
answering (a) vs (b) needs **no new attribution tooling** — only reading the existing graphs.

---

## Tasks

### Task 0 — Pick the analysis set (small, manual)
- Choose **2–3 prompts** that refuse when bare and comply under a jailbreak (use the
  behavioral/LLM-judge results to confirm a real bare→comply flip; weak jailbreaks are useless
  here — see [[project_dataset_v2_behavioral_findings]]).
- For each, line up the matched **bare** vs **jb_\*** conditions in the 4-way frontend, on
  **complement (green)** and **Qwen** columns.
- Prefer a **role-play / reframe** style jailbreak first — it's Georg's worked example and gives
  the cleanest "non-refusal feature → suppresses refusal" story.

### Task 1 — Establish the refusal-direction "input" feature sets (per prompt, per model)
For the **bare** (refusing) prompt, read off from the refusal-direction node:
- **Positive-into-refusal features** — the L13–15 model-token refusal features (decoder cosine
  ≈ high with r̂). Expect several near-duplicates ("I understand you're grappling…",
  "I cannot fulfill…", etc.).
- **Negative-into-refusal features** — the "don't-refuse / I-want-to-answer" features Georg
  found by scrolling the negative side of the same node.
- Record feature id, layer, position, and sign for each. This is the reference set the jb diff
  is measured against.

### Task 2 — Super-node the redundant refusal features (tooling sub-task)
The many near-duplicate L13–15 model-token refusal features clutter the graph and hide upstream
computation. Combine them into a single **"refusal" super-node** (and similarly a **"don't-refuse"
super-node** for the negatives), then re-trace upstream through the super-node.
- Check whether circuit-tracer's frontend already supports node grouping / super-nodes; if so,
  use it (no code). If not, the minimum viable version is a manual grouping in the write-up
  (list which feature ids collapse into each super-node) — do **not** build heavy tooling before
  confirming it's needed.
- Output: a cleaner upstream view so Task 3 can see what feeds the super-node, not 8 copies of it.

### Task 3 — Diff jailbroken vs non-jailbroken (the core deliverable)
For each prompt pair, compare the bare vs jb attribution graphs and classify the difference as
**(a) suppressed positives** or **(b) amplified negatives** (or a mix — quantify roughly):
- Did the positive-into-refusal super-node's incoming attribution **drop** under the jb? (→ a)
- Did the negative "don't-refuse" super-node's incoming attribution **rise** under the jb? (→ b)
- **Trace the upstream chain**: identify the non-refusal features (role-play, persona, instruction-
  framing, etc.) that newly appear / strengthen under the jb and follow the path by which they
  reach the refusal / don't-refuse super-node. This chain *is* "how non-refusal features compute
  the refusal direction" — the thing Georg actually asked for.

### Task 4 — Write up + form intuition (explicitly requested)
Georg: *"everyone, including me, should just spend a bunch of hours sifting through these refusal
circuits."* Deliverable is a short findings doc per prompt pair:
- (a) vs (b) verdict, with the specific features/super-nodes and attribution magnitudes.
- The upstream non-refusal → refusal-suppression chain, drawn or listed.
- Open intuition notes: how deep/complex is the computation? Heuristic-ish or a real multi-step
  circuit? Candidate steering targets?
- Caveat every causal claim — attribution ≠ causation; flag which claims would need a steering /
  ablation test to confirm (Georg repeatedly noted "we'd need a causal steering test to really
  prove it").

---

## Explicitly OUT of scope (Georg said stop)
- **The outlier (#443)** — "dumb trash direction, spend no time on it." No outlier analysis, no
  hunting for outlier behavior in other model families. Done.
- **Full (yellow) and outlier (purple) columns** — only green (complement) + Qwen matter now.
- **Other model families / largest-transcoder model** — that's an end-of-project, reviewer-
  pleasing translation step. Georg is confident it'll transfer; do it last, not now.
- **Re-labeling / subcircuit / ablation-cart panels** — still deferred (parity with unlabeled
  Qwen graphs).

## Open thread (flagged, not yet assigned)
Georg's unfinished end-of-meeting thought: whether the refusal-associated transcoder features
*themselves* make the circuit low-quality / incomplete. Park it; revisit if Task 3 traces hit
dead ends that look like transcoder artifacts rather than real computation.
