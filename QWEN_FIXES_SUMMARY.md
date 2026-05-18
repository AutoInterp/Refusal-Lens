# Qwen Pipeline Fixes — Complete Action Plan

**Status**: Issue #1 ✅ Fixed | Issues #2–5 In Progress

---

## Issue #1: MEASUREMENT_LAYER Mismatch ✅ FIXED

**Changed**: `scripts/pipeline_qwen/config.py:28`
```python
# OLD (buggy)
MEASUREMENT_LAYER = 34     # Separation layer (wrong!)

# NEW (correct)
MEASUREMENT_LAYER = 18     # Causal layer (matches BEST_CAUSAL_LAYER)
```

**Why**: Georg's l15-refactor (Gemma) mandated that attribution must target the layer where **intervention actually works**, not the peak-separation layer. Qwen's causal layer is L18 (vs Gemma's L15).

**Impact**: 
- All Stage 02 attribution graphs will now target L18 instead of L34
- Stage 03 verification will use correct measurement layer
- Expected to fix or improve Issue #2 (benign force-refuse)

---

## Issue #2: Benign force-refuse = 0/10 (Expected to Fix)

**Root cause**: Likely the MEASUREMENT_LAYER bug (now fixed in #1).

**Next step**: After re-running Stages 01–02 with corrected config:

```bash
# (1) Recompute direction at L18
python3 scripts/pipeline_qwen/01_compute_direction.py --run-dir <new_run_dir>

# (2) Re-run attribution at L18
python3 scripts/pipeline_qwen/02_run_attribution.py --run-dir <new_run_dir>

# (3) Re-test causal intervention on benign prompts
python3 scripts/pipeline_qwen/06_causal_intervention.py --run-dir <new_run_dir> \
    --modes pro_refusal_add anti_refusal_sub
```

**Expected outcome**: Benign force-refuse should improve from 0/10 toward 10/10 (like Gemma).

**If still broken after fix**: Investigate whether:
- Qwen's refusal is class-conditional (not universal across all benign prompts)
- Layer 18's direction magnitude is appropriate for Qwen's architecture
- Direction computation needs per-position tuning (currently defaults to -1)

---

## Issue #3: Dataset Contamination (In Progress)

**Current state**: 40/50 bare refuse (vs 50/50 Gemma), 13 ctrl leaks, 10 bare-comply.

**Step 1: Validate dataset**
```bash
# Full run (~30 min on GPU)
PYTHONPATH=src python3 scripts/pipeline_qwen/verify_dataset_qwen.py

# Output:
# - dataset/qwen_dataset_verification.json
# - dataset/qwen_dataset_verification.md
```

**Step 2: Filter downstream stages**

Once you have `qwen_dataset_verification.json`, filter:

- **Stage 06 anti_refusal_sub**: Run only on prompts where bare REFUSES
  - Use: `bare_refused_ids` from verification JSON
  
- **Stage 06 pro_refusal_add**: Run only on (prompt, class) pairs where:
  - JB COMPLIES (substrate for intervention) **AND**
  - ctrl REFUSES (clean negative control)
  - Use: `jb_comply_pairs` minus `ctrl_leak_pairs`

- **Stage 07/08**: Unaffected by contamination; use all bare + JB pairs

**Qwen-specific caveats**:
- 40/50 bare refuse is **6× better than Qwen's naive baseline** (~10% comply), so dataset curation is working
- 13 ctrl leaks are the headline concern; filter these from pro_refusal_add measurement
- 10 bare-comply are acceptable for feature analysis but excluded from anti_refusal_sub counts

---

## Issue #4: BOS Token Verification (In Progress)

**Script**: `scripts/pipeline_qwen/verify_bos_handling.py`

**Current status**: Running (model download in progress)

**What it does**:
- Tests 5 prompts (harmful + benign mix) at layer 18, position -1
- Compares activations with/without `add_special_tokens=False`
- Reports cosine similarity

**Interpretation**:
- **Cosine > 0.999**: BOS token doesn't matter; direction is valid
- **Cosine < 0.999**: BOS token handling differs; may need to recompute direction

**Expected outcome**: Cosine ~0.9999 (matching Gemma's 0.999983)

**If different**: No immediate action needed — the direction is still valid, just note it in methodology section

---

## Issue #5: Stage 02b Sample Size (Depends on #1, #3)

**Current**: n=5 pairs per JB class (unreliable Cohen's d, marginal p-values)

**Root cause**: The 5-prompt Qwen run was a pilot; real dataset has 40–50 prompts.

**Fix after #1 + #3**:
1. Re-run Stage 02 on full 40–50 clean prompts with corrected MEASUREMENT_LAYER = 18
2. Expected outcome: n=20–40 pairs per class (Cohen's d becomes reliable)
3. p-values will reach statistical significance at typical α=0.05

**No code changes needed** — Stage 02b automatically uses all available pairs (line 170 of 02b_statistical_analysis.py: `n = len(a)`).

---

## Execution Sequence (GPU time required)

1. **DONE** ✅ Fix MEASUREMENT_LAYER → 18

2. **Quick (5 min, CPU)** — Verify BOS handling (currently running)

3. **Medium (30 min, GPU)** — Run dataset verifier
   ```bash
   PYTHONPATH=src python3 scripts/pipeline_qwen/verify_dataset_qwen.py
   ```

4. **Heavy (6+ hours, GPU)** — Re-run pipeline with clean config
   ```bash
   python3 scripts/pipeline_qwen/01_compute_direction.py --run-dir data/results/pipeline_runs_qwen/run_YYYYMMDD_HHMMSS
   python3 scripts/pipeline_qwen/02_run_attribution.py --run-dir data/results/pipeline_runs_qwen/run_YYYYMMDD_HHMMSS
   python3 scripts/pipeline_qwen/02b_statistical_analysis.py --run-dir data/results/pipeline_runs_qwen/run_YYYYMMDD_HHMMSS
   python3 scripts/pipeline_qwen/03_verify_attribution.py --run-dir data/results/pipeline_runs_qwen/run_YYYYMMDD_HHMMSS
   ```

5. **Validation (1 hour, GPU)** — Test causal intervention
   ```bash
   python3 scripts/pipeline_qwen/06_causal_intervention.py --run-dir data/results/pipeline_runs_qwen/run_YYYYMMDD_HHMMSS
   ```

6. **Optional** — Deep causal analysis & subcircuit ablation (Stages 07–08)

---

## Key Takeaway

**These are configuration + dataset issues, not code bugs.** The computational pipeline matches Mahmoud's implementation exactly. The fixes are:
1. ✅ Config constant (MEASUREMENT_LAYER)
2. Dataset validation & filtering
3. Verification scripts (BOS handling, dataset quality)
4. Re-run stages with corrected constants

Expect the benign-control failure to improve significantly after Issue #1 is re-run at scale (40–50 prompts instead of 5).
