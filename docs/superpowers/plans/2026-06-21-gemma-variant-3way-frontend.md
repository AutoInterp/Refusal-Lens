# Gemma Variant Attribution Graphs → 4-Way Frontend Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate Gemma attribution graphs for three refusal-direction variants (complement / full / outlier) and serve them in a 4-column frontend alongside Qwen, so the "complement is the real refusal circuit; full is #443-artifact-dominated; outlier is inert" story is visual.

**Architecture:** A CPU direction-builder writes unit-normalized variant directions into `gemma_var_<v>` run-dirs; a RunPod orchestrator runs Stage-02 attribution (`--save-graphs`, residual-stream hook) per variant, gates each against Georg's committed `net` magnitudes, then packs/annotates/pushes each as its own run to `moon70/refusal-lens-graphs`; a CPU assembler pulls the 3 Gemma runs + the existing Qwen run into one static site with a manifest-driven N-column compare harness (each column = a full gridsnap viewer in `compact` mode).

**Tech Stack:** Python 3.10+ (torch, huggingface_hub), the vendored circuit-tracer fork (branch `refusal-lens-multi-position-fix`), bash, vanilla HTML/CSS/JS (no build step).

## Global Constraints

- **HF dataset repo (push + fetch):** `moon70/refusal-lens-graphs` (dataset). Qwen run already at `runs/run_emnlp_qwen_L18_20260522/`.
- **New HF run-ids:** `run_gemma_complement_L15`, `run_gemma_full_L15`, `run_gemma_outlier_L15`.
- **Slug scheme:** `{prompt_idx:03d}_{condition}_single` (single-mode via `--skip-multi-graph`). Matches Qwen's existing slugs.
- **Attribution settings (all variants):** layer **15**, position **−2**, `--measurement-hook hook_resid_post`, `--backend transformerlens`, `--dtype float32`, `--save-graphs`. Requires `vendor/circuit-tracer` installed editable (`uv pip install -e vendor/circuit-tracer`).
- **Gemma outlier dimension:** **443** (value −2790.53; `‖outlier‖/‖full‖ ≈ 0.90`). Canonical full direction: `data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt` (`dict[layer→tensor(2560)]`).
- **Correctness-gate reference:** `data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json` (`variants.{full,outlier,complement}` = `[{prompt_idx,condition,net}×550]`; complement bare net ≈ +909, full ≈ −48k, outlier ≈ −55k).
- **v1 scope:** NO label baking, NO subcircuit panel for Gemma variants (`--skip-subcircuits`). Parity with Qwen's current unlabeled graphs.
- **Tests:** plain-script style (asserts + a `__main__` runner printing `PASS`/`FAIL`), matching `scripts/emnlp_perm_edit/tests/test_qwen_subcircuit_orchestration.py`; run with `PYTHONPATH=scripts`.

---

## File Structure

**Create**
- `scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py` — variant direction builder (Unit A)
- `scripts/emnlp_perm_edit/verify_variant_nets.py` — correctness gate (Unit B gate)
- `scripts/pipeline/assemble_compare_frontend.py` — fetch + manifest builder (Unit C)
- `scripts/pipeline/05_frontend_patches/compare_config.json` — 4-column config (Unit C)
- `scripts/pipeline/05_frontend_patches/compare_multi.html` — N-column harness (Unit D)
- `scripts/pipeline/05_frontend_patches/compact-mode.css` — column-stack patch (Unit D)
- `scripts/pipeline/05_frontend_patches/compact-mode.js` — column-stack patch (Unit D)
- `scripts/emnlp_perm_edit/runpod_gemma_variants.sh` — orchestrator (Unit B)
- `scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh` — watcher (Unit B)
- `scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh` — GPU smoke (Unit B)
- `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py` — CPU unit tests
- `docs/RUNPOD_GEMMA_VARIANTS_SETUP.md` — runbook

**Modify**
- `scripts/pipeline/push_graph_data.py` — add `--run-name` override
- `scripts/pipeline/utils_viz.py` — inject `compact-mode.{css,js}` in `stage_frontend`

---

## Task 1: Variant direction builder (Unit A)

**Files:**
- Create: `scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py`
- Test: `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`

**Interfaces:**
- Produces: `build_variant_directions(r_full: torch.Tensor) -> dict[str, torch.Tensor]` (keys `full`/`outlier`/`complement`, each unit-normalized 1-D); `load_full_direction(path, layer=15) -> torch.Tensor`; `write_variant_dirs(variants, runs_base, layer=15, pos=-2)`.

- [ ] **Step 1: Write the failing test**

Create `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`:

```python
"""CPU unit tests for the Gemma-variant pipeline (no GPU, no HF)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                 # scripts/emnlp_perm_edit
sys.path.insert(0, str(HERE.parents[2] / "pipeline"))    # scripts/pipeline

from ensure_gemma_variant_directions import build_variant_directions  # noqa: E402


def _fake_r_full(d=2560, spike_dim=443, spike=-2790.53):
    torch.manual_seed(0)
    r = torch.randn(d) * 3.0
    r[spike_dim] = spike
    return r


def test_build_variant_directions():
    r = _fake_r_full()
    v = build_variant_directions(r)
    # all unit-normalized
    for name in ("full", "outlier", "complement"):
        assert abs(float(v[name].norm()) - 1.0) < 1e-5, name
    # outlier is nonzero only at 443
    nz = torch.nonzero(v["outlier"]).flatten().tolist()
    assert nz == [443], nz
    # complement zeros dim 443
    assert float(v["complement"][443]) == 0.0
    # outlier carries ~90% of the (magnitude) norm of full
    assert abs(r[443].abs().item() / r.norm().item() - 0.90) < 0.05
    print("PASS test_build_variant_directions")


if __name__ == "__main__":
    test_build_variant_directions()
    print("ALL PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ensure_gemma_variant_directions'`

- [ ] **Step 3: Write the implementation**

Create `scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py`:

```python
"""Deterministically (re)build the Gemma refusal-direction variants
(full / outlier / complement) and write the unit-normalized vectors into each
gemma_var_<v> run-dir so Stage 02 attributes toward them.

Recipe (matches docs/REFUSAL_DIRECTION_INVESTIGATION_2026-06-16.md):
  r_full      = unnormalized diff-in-means direction at L15 (dict[layer]->tensor)
  outlier_dim = argmax(|r_full|)                      (== 443 for Gemma L15)
  full        = r_full
  outlier     = zeros; outlier[outlier_dim] = r_full[outlier_dim]
  complement  = r_full.clone(); complement[outlier_dim] = 0
each unit-normalized and written to
  gemma_var_<v>/01_direction/{directions/layer_15.pt, positions_L15/pos_-2.pt}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FULL = REPO / "data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt"
SPLIT_STATS = REPO / "data/results/emnlp_perm_edit/phase0_controllability/gemma_outlier_split_stats.json"
RUNS_BASE = REPO / "data/results/pipeline_runs"
LAYER = 15
POS = -2
EXPECT_OUTLIER_DIM = 443


def load_full_direction(path: Path, layer: int = LAYER) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if layer in obj:
            return obj[layer]
        if "direction" in obj:
            return obj["direction"]
        raise ValueError(f"layer {layer} not in direction dict (keys {list(obj)[:5]})")
    if isinstance(obj, torch.Tensor):
        return obj
    raise ValueError(f"unrecognized direction file format: {type(obj)}")


def build_variant_directions(r_full: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return {full, outlier, complement} as UNIT-normalized 1-D tensors."""
    r = r_full.detach().float().flatten()
    outlier_dim = int(r.abs().argmax())
    full = r.clone()
    outlier = torch.zeros_like(r)
    outlier[outlier_dim] = r[outlier_dim]
    complement = r.clone()
    complement[outlier_dim] = 0.0
    return {name: v / v.norm() for name, v in
            (("full", full), ("outlier", outlier), ("complement", complement))}


def write_variant_dirs(variants: dict[str, torch.Tensor], runs_base: Path,
                       layer: int = LAYER, pos: int = POS) -> None:
    for name, unit in variants.items():
        rd = runs_base / f"gemma_var_{name}" / "01_direction"
        (rd / "directions").mkdir(parents=True, exist_ok=True)
        (rd / f"positions_L{layer}").mkdir(parents=True, exist_ok=True)
        existing = rd / "directions" / f"layer_{layer:02d}.pt"
        if existing.exists():
            old = torch.load(existing, map_location="cpu", weights_only=False)
            old = old["direction"] if isinstance(old, dict) and "direction" in old else old
            cos = float(torch.nn.functional.cosine_similarity(
                unit, old.float().flatten(), dim=0))
            assert cos > 0.999, (
                f"{name}: rebuilt direction disagrees with committed file (cos={cos:.4f}); "
                f"refusing to overwrite. Investigate the canonical source.")
        torch.save(unit, rd / "directions" / f"layer_{layer:02d}.pt")
        torch.save(unit, rd / f"positions_L{layer}" / f"pos_{pos:+d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-direction", type=Path, default=DEFAULT_FULL)
    ap.add_argument("--runs-base", type=Path, default=RUNS_BASE)
    ap.add_argument("--check-only", action="store_true",
                    help="Run self-checks but do not write the run-dir files")
    args = ap.parse_args()

    r_full = load_full_direction(args.full_direction)
    variants = build_variant_directions(r_full)

    outlier_dim = int(r_full.float().flatten().abs().argmax())
    assert outlier_dim == EXPECT_OUTLIER_DIM, f"outlier_dim {outlier_dim} != {EXPECT_OUTLIER_DIM}"
    assert float(variants["complement"][EXPECT_OUTLIER_DIM]) == 0.0
    assert torch.nonzero(variants["outlier"]).flatten().tolist() == [EXPECT_OUTLIER_DIM]
    for v in variants.values():
        assert abs(float(v.norm()) - 1.0) < 1e-5
    if SPLIT_STATS.exists():
        norms = json.loads(SPLIT_STATS.read_text())["norms"]
        ratio = norms["outlier"] / norms["full"]
        assert abs(ratio - 0.8998) < 0.02, f"norm ratio {ratio} (expected ~0.90)"
    print(f"self-check OK: outlier_dim={outlier_dim}, unit-normalized, norms match split-stats")

    if args.check_only:
        return
    write_variant_dirs(variants, args.runs_base)
    print(f"wrote directions into gemma_var_{{full,outlier,complement}}/01_direction/ "
          f"(layer_{LAYER:02d}.pt + positions_L{LAYER}/pos_{POS:+d}.pt)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: `PASS test_build_variant_directions` then `ALL PASS`

- [ ] **Step 5: Run the real self-check against committed data (no writes)**

Run: `python3 scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py --check-only`
Expected: `self-check OK: outlier_dim=443, unit-normalized, norms match split-stats`

- [ ] **Step 6: Commit**

```bash
git add scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py
git commit -m "feat(gemma-variants): deterministic variant direction builder + test"
```

---

## Task 2: Correctness gate — `verify_variant_nets.py` (Unit B gate)

**Files:**
- Create: `scripts/emnlp_perm_edit/verify_variant_nets.py`
- Test: `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py` (append)

**Interfaces:**
- Consumes: a fresh `02_attribution/attribution_results.json` and `gemma_var_nets.json`.
- Produces: `extract_nets(attribution_results: dict) -> list[dict]` (each `{prompt_idx,condition,net}`); `compare_nets(new, ref, *, corr_min=0.95, bare_rel_tol=0.25) -> dict` (`{ok, n_paired, corr, bare_mean_new, bare_mean_ref, sign_ok, bare_rel_err}`). CLI exits 0 if `ok` else 1.

- [ ] **Step 1: Write the failing test (append to the test file, before `__main__`)**

```python
from verify_variant_nets import extract_nets, compare_nets  # noqa: E402


def test_extract_and_compare_nets():
    attribution = {"results": [
        {"prompt_idx": 0, "conditions": {
            "bare": {"graphs": {"single": {"net": 900.0}}},
            "jb_fiction": {"graphs": {"single": {"net": 500.0}}}}},
        {"prompt_idx": 1, "conditions": {
            "bare": {"graphs": {"single": {"net": 920.0}}}}},
    ]}
    recs = extract_nets(attribution)
    assert {"prompt_idx": 0, "condition": "bare", "net": 900.0} in recs
    assert len(recs) == 3
    ref = [{"prompt_idx": 0, "condition": "bare", "net": 908.0},
           {"prompt_idx": 0, "condition": "jb_fiction", "net": 480.0},
           {"prompt_idx": 1, "condition": "bare", "net": 915.0}]
    good = compare_nets(recs, ref)
    assert good["ok"] is True, good
    # a sign-flipped / wrong-magnitude run must fail the gate
    bad = compare_nets([{"prompt_idx": 0, "condition": "bare", "net": -48000.0},
                        {"prompt_idx": 1, "condition": "bare", "net": -47000.0}], ref)
    assert bad["ok"] is False, bad
    print("PASS test_extract_and_compare_nets")
```

Add `test_extract_and_compare_nets()` to the `__main__` runner.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'verify_variant_nets'`

- [ ] **Step 3: Write the implementation**

Create `scripts/emnlp_perm_edit/verify_variant_nets.py`:

```python
"""Correctness gate for regenerated variant attribution.

A fresh Stage-02 run targeting a refusal-direction variant must reproduce
Georg's attributed `net` magnitudes (data/.../gemma_var_nets.json). If it does
not, the run used the wrong direction / measurement hook / circuit-tracer build
and the graphs must NOT be trusted. Exit code 0 = pass, 1 = fail (drives the
orchestrator).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_nets(attribution_results: dict) -> list[dict]:
    out = []
    for res in attribution_results.get("results", []):
        idx = res.get("prompt_idx")
        for cond, cdata in (res.get("conditions") or {}).items():
            single = ((cdata or {}).get("graphs") or {}).get("single") or {}
            net = single.get("net")
            if net is not None:
                out.append({"prompt_idx": idx, "condition": cond, "net": float(net)})
    return out


def _bare_mean(records: list[dict]) -> float:
    vals = [r["net"] for r in records if r["condition"] == "bare"]
    return sum(vals) / len(vals) if vals else float("nan")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def compare_nets(new_records: list[dict], ref_records: list[dict], *,
                 corr_min: float = 0.95, bare_rel_tol: float = 0.25) -> dict:
    ref_by_key = {(r["prompt_idx"], r["condition"]): r["net"] for r in ref_records}
    paired = [(r["net"], ref_by_key[(r["prompt_idx"], r["condition"])])
              for r in new_records if (r["prompt_idx"], r["condition"]) in ref_by_key]
    corr = _pearson([a for a, _ in paired], [b for _, b in paired]) if len(paired) >= 2 else float("nan")
    nb, rb = _bare_mean(new_records), _bare_mean(ref_records)
    sign_ok = (nb == 0 and rb == 0) or (nb * rb > 0)
    rel = abs(nb - rb) / max(abs(rb), 1e-9)
    corr_ok = (len(paired) < 2) or (corr >= corr_min)
    ok = bool(corr_ok and sign_ok and rel <= bare_rel_tol)
    return {"ok": ok, "n_paired": len(paired), "corr": corr,
            "bare_mean_new": nb, "bare_mean_ref": rb,
            "sign_ok": sign_ok, "bare_rel_err": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution-results", type=Path, required=True)
    ap.add_argument("--nets-ref", type=Path, required=True)
    ap.add_argument("--variant", required=True, choices=["full", "outlier", "complement"])
    ap.add_argument("--corr-min", type=float, default=0.95)
    ap.add_argument("--bare-rel-tol", type=float, default=0.25)
    a = ap.parse_args()
    new = extract_nets(json.loads(a.attribution_results.read_text()))
    ref = json.loads(a.nets_ref.read_text())["variants"][a.variant]
    r = compare_nets(new, ref, corr_min=a.corr_min, bare_rel_tol=a.bare_rel_tol)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: both `PASS …` lines then `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/emnlp_perm_edit/verify_variant_nets.py scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py
git commit -m "feat(gemma-variants): nets correctness gate + test"
```

---

## Task 3: `push_graph_data.py --run-name` override

**Files:**
- Modify: `scripts/pipeline/push_graph_data.py`
- Test: `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py` (append)

**Interfaces:**
- Produces: `resolve_run_name(run_name_arg: str | None, run_dir: Path) -> str`; new CLI flag `--run-name`.

- [ ] **Step 1: Write the failing test (append, before `__main__`)**

```python
import importlib.util  # noqa: E402


def _load_push_module():
    path = HERE.parents[2] / "pipeline" / "push_graph_data.py"
    spec = importlib.util.spec_from_file_location("push_graph_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_run_name():
    push = _load_push_module()
    assert push.resolve_run_name("run_gemma_complement_L15", Path("/x/gemma_var_complement")) == "run_gemma_complement_L15"
    assert push.resolve_run_name(None, Path("/x/gemma_var_complement")) == "gemma_var_complement"
    print("PASS test_resolve_run_name")
```

Add `test_resolve_run_name()` to the `__main__` runner.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: FAIL with `AttributeError: module 'push_graph_data' has no attribute 'resolve_run_name'`

- [ ] **Step 3: Implement — add the flag, the helper, and use it**

In `scripts/pipeline/push_graph_data.py`, inside `parse_args()` after the `--run-dir` argument, add:

```python
    p.add_argument("--run-name", type=str, default=None,
                   help="Override the HF run-id (default: run-dir basename). "
                        "Use when the local run-dir name differs from the desired "
                        "HF run id, e.g. gemma_var_complement -> run_gemma_complement_L15.")
```

Above `def main():` add the helper:

```python
def resolve_run_name(run_name_arg, run_dir):
    """HF run-id = explicit --run-name override, else the run-dir basename."""
    return run_name_arg or Path(run_dir).name
```

In `main()`, replace `run_name = run_dir.name` with:

```python
    run_name = resolve_run_name(args.run_name, run_dir)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: all `PASS …` lines then `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/push_graph_data.py scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py
git commit -m "feat(push): --run-name override for HF run-id"
```

---

## Task 4: Compare manifest builder (Unit C, pure)

**Files:**
- Create: `scripts/pipeline/assemble_compare_frontend.py` (functions only this task; CLI in Task 5)
- Test: `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py` (append)

**Interfaces:**
- Produces: `parse_condition(slug, mode_suffixes=("single","multi")) -> tuple[str,str] | None`; `build_compare_manifest(columns: list[dict], title: str) -> dict`. Each input column is `{label, dir, model, target, graphs: [{slug, prompt}]}`. Output manifest: `{title, columns:[{label,dir,model,target,slugmap}], prompts:[{idx,text}], conditions:[str]}`. Only `(idx,cond)` present in **all** columns are offered.

- [ ] **Step 1: Write the failing test (append, before `__main__`)**

```python
from assemble_compare_frontend import parse_condition, build_compare_manifest  # noqa: E402


def test_parse_condition():
    assert parse_condition("000_bare_single") == ("000", "bare")
    assert parse_condition("012_jb_fiction_single") == ("012", "jb_fiction")
    assert parse_condition("003_ctrl_analytical") == ("003", "ctrl_analytical")
    assert parse_condition("garbage") is None
    print("PASS test_parse_condition")


def test_build_compare_manifest_intersection():
    colA = {"label": "G-cmpl", "dir": "run_gemma_complement_L15/05_frontend",
            "model": "gemma", "target": "complement",
            "graphs": [{"slug": "000_bare_single", "prompt": "p0"},
                       {"slug": "000_jb_fiction_single", "prompt": "p0"}]}
    colB = {"label": "Qwen", "dir": "run_emnlp_qwen_L18_20260522/05_frontend",
            "model": "qwen", "target": "full",
            "graphs": [{"slug": "000_bare_single", "prompt": "p0"}]}  # no jb_fiction
    m = build_compare_manifest([colA, colB], title="t")
    assert [p["idx"] for p in m["prompts"]] == ["000"]
    assert m["conditions"] == ["bare"]                    # jb_fiction not in ALL columns
    assert m["columns"][0]["slugmap"]["000_bare"] == "000_bare_single"
    assert "000_jb_fiction" not in m["columns"][0]["slugmap"]
    print("PASS test_build_compare_manifest_intersection")
```

Add both to the `__main__` runner.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'assemble_compare_frontend'`

- [ ] **Step 3: Write the implementation (functions)**

Create `scripts/pipeline/assemble_compare_frontend.py`:

```python
"""Assemble an N-column attribution-graph compare site.

Fetches each configured run (Gemma variants + Qwen) from the HF dataset into
one parent dir, builds a compare_manifest.json (shared prompt x condition
options + per-column slug maps), and stages the manifest-driven compare harness.
Pure functions (parse_condition, build_compare_manifest) are unit-tested; the
CLI (main) does fetch + filesystem assembly.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATCHES = Path(__file__).resolve().parent / "05_frontend_patches"
_SLUG_RE = re.compile(r"^(\d+)_(.+)$")


def parse_condition(slug: str, mode_suffixes=("single", "multi")):
    """Return (idx, condition) with any trailing mode suffix stripped, or None."""
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    idx, cond = m.group(1), m.group(2)
    parts = cond.split("_")
    if len(parts) > 1 and parts[-1] in mode_suffixes:
        cond = "_".join(parts[:-1])
    return idx, cond


def build_compare_manifest(columns: list[dict], title: str) -> dict:
    """columns: [{label, dir, model, target, graphs:[{slug, prompt}]}].
    Offers only (idx, cond) present in ALL columns; resolves each column's
    actual slug per (idx, cond)."""
    per_col = []
    for c in columns:
        smap, ptext = {}, {}
        for g in c["graphs"]:
            pc = parse_condition(g["slug"])
            if not pc:
                continue
            idx, cond = pc
            smap[f"{idx}_{cond}"] = g["slug"]
            ptext.setdefault(idx, g.get("prompt", "") or "")
        per_col.append({**c, "smap": smap, "ptext": ptext})

    keysets = [set(col["smap"].keys()) for col in per_col]
    shared = sorted(set.intersection(*keysets)) if keysets else []
    idxs = sorted({k.split("_", 1)[0] for k in shared}, key=int)
    conds = sorted({k.split("_", 1)[1] for k in shared})

    ptext0 = per_col[0]["ptext"] if per_col else {}
    return {
        "title": title,
        "columns": [{"label": col["label"], "dir": col["dir"], "model": col["model"],
                     "target": col["target"],
                     "slugmap": {k: col["smap"][k] for k in shared}}
                    for col in per_col],
        "prompts": [{"idx": i, "text": ptext0.get(i, "")} for i in idxs],
        "conditions": conds,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: all `PASS …` lines then `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/assemble_compare_frontend.py scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py
git commit -m "feat(compare): manifest builder + slug parsing (pure, tested)"
```

---

## Task 5: Compare config + assembler CLI (Unit C)

**Files:**
- Create: `scripts/pipeline/05_frontend_patches/compare_config.json`
- Modify: `scripts/pipeline/assemble_compare_frontend.py` (add `main()` + helpers)

**Interfaces:**
- Consumes: `compare_config.json` (`{title, dataset_repo, columns:[{label, run, model, target}]}`), `build_compare_manifest` (Task 4), `fetch_graph_data.py` (subprocess).
- Produces: `data/results/compare_3way/<run>/05_frontend/...` per column, `data/results/compare_3way/compare_manifest.json`, `data/results/compare_3way/compare.html`.

- [ ] **Step 1: Write the config**

Create `scripts/pipeline/05_frontend_patches/compare_config.json`:

```json
{
  "title": "Refusal circuit: Gemma variants vs Qwen",
  "dataset_repo": "moon70/refusal-lens-graphs",
  "columns": [
    {"label": "Gemma · complement (no #443)", "run": "run_gemma_complement_L15", "model": "gemma", "target": "complement"},
    {"label": "Gemma · full (+#443)",          "run": "run_gemma_full_L15",       "model": "gemma", "target": "full"},
    {"label": "Gemma · outlier #443 only",     "run": "run_gemma_outlier_L15",    "model": "gemma", "target": "outlier"},
    {"label": "Qwen (L18)",                     "run": "run_emnlp_qwen_L18_20260522", "model": "qwen", "target": "full"}
  ]
}
```

- [ ] **Step 2: Add the assembler `main()` to `assemble_compare_frontend.py`**

Append:

```python
def _fetch_run(run: str, dataset_repo: str, out_base: Path) -> None:
    """Shell out to fetch_graph_data.py so each column is a self-contained viewer."""
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "fetch_graph_data.py"),
           "--run", run, "--dataset-repo", dataset_repo, "--out-base", str(out_base)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_column_graphs(run_dir: Path) -> list[dict]:
    """Read a fetched column's data/graph-metadata.json -> [{slug, prompt}]."""
    md = run_dir / "05_frontend" / "data" / "graph-metadata.json"
    meta = json.loads(md.read_text())
    return [{"slug": g["slug"], "prompt": g.get("prompt", "")} for g in meta["graphs"]]


def main():
    ap = argparse.ArgumentParser(description="Assemble the N-column compare site")
    ap.add_argument("--config", type=Path, default=PATCHES / "compare_config.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data/results/compare_3way")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="Reuse already-fetched columns under --out (no HF download)")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    columns = []
    for col in cfg["columns"]:
        run = col["run"]
        if not args.skip_fetch:
            _fetch_run(run, cfg["dataset_repo"], args.out)
        run_dir = args.out / run
        if not (run_dir / "05_frontend" / "data" / "graph-metadata.json").exists():
            print(f"ERROR: no graph-metadata for column '{run}' under {run_dir}")
            sys.exit(1)
        columns.append({"label": col["label"], "dir": f"{run}/05_frontend",
                        "model": col["model"], "target": col["target"],
                        "graphs": _load_column_graphs(run_dir)})

    manifest = build_compare_manifest(columns, title=cfg.get("title", "compare"))
    (args.out / "compare_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(PATCHES / "compare_multi.html", args.out / "compare.html")

    print(f"\nAssembled {len(columns)} columns; "
          f"{len(manifest['prompts'])} shared prompts x {len(manifest['conditions'])} conditions.")
    print(f"Serve:\n  cd {args.out}\n  python3 -m http.server 8000")
    print(f"  open http://localhost:8000/compare.html")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Re-run the unit tests (ensure the new `main`/helpers didn't break imports)**

Run: `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`
Expected: `ALL PASS`

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/assemble_compare_frontend.py scripts/pipeline/05_frontend_patches/compare_config.json
git commit -m "feat(compare): assembler CLI + 4-column config"
```

---

## Task 6: Compact-mode patch + `stage_frontend` injection (Unit D)

**Files:**
- Create: `scripts/pipeline/05_frontend_patches/compact-mode.css`
- Create: `scripts/pipeline/05_frontend_patches/compact-mode.js`
- Modify: `scripts/pipeline/utils_viz.py` (`stage_frontend` injection list + cache-buster tuple)

**Interfaces:**
- Produces: `?compact=1` behavior on the vendored `index.html` — hides `.nav` selector chrome, makes `.cg` fill the column. Injected by `stage_frontend` into every fetched run (harmless without `?compact=1`).

- [ ] **Step 1: Write `compact-mode.css`**

```css
/* Compact column mode for the N-way compare harness. Activated when the viewer
   is loaded with ?compact=1 (compact-mode.js sets html.rl-compact). The parent
   compare bar drives prompt/condition selection, so the per-iframe nav chrome
   is hidden and the gridsnap graph container fills the column. */
html.rl-compact { height: 100%; }
html.rl-compact body { margin: 0; height: 100%; }
html.rl-compact .nav { display: none !important; }
html.rl-compact .cg { height: 100vh; width: 100%; }
```

- [ ] **Step 2: Write `compact-mode.js`**

```javascript
/* Sets html.rl-compact synchronously when ?compact=1 is present, so the
   compact-mode.css rules hide the nav chrome and fill the column regardless of
   when the gridsnap viewer builds its DOM. No-op without the query param. */
(function () {
  try {
    if (new URLSearchParams(location.search).get("compact")) {
      document.documentElement.classList.add("rl-compact");
    }
  } catch (e) { /* older browsers: leave full UI */ }
})();
```

- [ ] **Step 3: Add both to the `stage_frontend` injection in `utils_viz.py`**

In `scripts/pipeline/utils_viz.py`, in the `injection = (...)` string (after the `subcircuit-panel.css` link and the `subcircuit-panel.js` / `feature-cart.js` scripts), add two lines so the full block includes:

```python
            f'<link rel="stylesheet" href="./compact-mode.css{_v("compact-mode.css")}">\n'
            f'<script src="./compact-mode.js{_v("compact-mode.js")}" defer></script>\n'
```

And add `"compact-mode.css", "compact-mode.js"` to the cache-buster refresh tuple (the `for name in (...)` list that currently ends with `"feature-cart.js",`).

- [ ] **Step 4: Validate injection by staging a throwaway frontend**

Run:

```bash
python3 - <<'PY'
import sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, "scripts/pipeline")
from utils_viz import stage_frontend, VENDOR_FRONTEND
tmp = Path(tempfile.mkdtemp())
gd = tmp / "graph_data"; gd.mkdir()
(gd / "graph-metadata.json").write_text('{"graphs": []}')
stage_frontend(graph_data_dir=gd, frontend_out=tmp)
html = (tmp / "index.html").read_text()
assert "compact-mode.css" in html and "compact-mode.js" in html, "compact patch not injected"
assert (tmp / "compact-mode.css").exists() and (tmp / "compact-mode.js").exists()
print("compact-mode injection OK")
shutil.rmtree(tmp)
PY
```

Expected: `compact-mode injection OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/05_frontend_patches/compact-mode.css scripts/pipeline/05_frontend_patches/compact-mode.js scripts/pipeline/utils_viz.py
git commit -m "feat(frontend): compact column mode patch + stage_frontend injection"
```

---

## Task 7: N-column compare harness (Unit D)

**Files:**
- Create: `scripts/pipeline/05_frontend_patches/compare_multi.html`

**Interfaces:**
- Consumes: `./compare_manifest.json` (Task 4 schema) and per-column `./<dir>/index.html?slug=…&compact=1`.

- [ ] **Step 1: Write the harness**

Create `scripts/pipeline/05_frontend_patches/compare_multi.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Refusal-Lens · Circuit Compare</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #222; }
  .toolbar { display: flex; gap: 14px; align-items: center; padding: 10px 18px;
    background: #fafafa; border-bottom: 1px solid #ddd; font-size: 13px; }
  .toolbar h1 { margin: 0; padding-right: 16px; font-size: 14px; font-weight: 600; border-right: 1px solid #ccc; }
  .toolbar select { padding: 4px 8px; border: 1px solid #ccc; border-radius: 3px; font-size: 13px; background: #fff; }
  .toolbar .status { margin-left: auto; color: #666; font-size: 12px; font-family: ui-monospace, monospace; }
  .cols { display: flex; height: calc(100vh - 50px); overflow-x: auto; }
  .col { flex: 1 0 460px; display: flex; flex-direction: column; min-width: 460px; border-left: 1px solid #ddd; }
  .col:first-child { border-left: none; }
  .col-label { padding: 6px 12px; font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .03em; background: #eceff1; color: #263238; border-bottom: 1px solid #ddd;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .col.gemma_complement .col-label { background: #e8f5e9; color: #1b5e20; }
  .col.gemma_full       .col-label { background: #fff3e0; color: #bf360c; }
  .col.gemma_outlier    .col-label { background: #f3e5f5; color: #4a148c; }
  .col.qwen_full        .col-label { background: #e3f2fd; color: #0d47a1; }
  iframe { flex: 1; border: none; background: #fff; }
</style>
</head>
<body>
  <div class="toolbar">
    <h1>Refusal-Lens · Compare</h1>
    <label>Prompt: <select id="prompt-select"></select></label>
    <label>Condition: <select id="cond-select"></select></label>
    <span class="status" id="status">loading…</span>
  </div>
  <div class="cols" id="cols"></div>

<script>
let M;
async function init() {
  const resp = await fetch('./compare_manifest.json');
  if (!resp.ok) throw new Error('compare_manifest.json fetch failed: ' + resp.status);
  M = await resp.json();
  document.querySelector('.toolbar h1').textContent = 'Refusal-Lens · ' + (M.title || 'Compare');

  const ps = document.getElementById('prompt-select');
  for (const p of M.prompts) {
    const o = document.createElement('option');
    o.value = p.idx;
    const t = (p.text || '').slice(0, 55);
    o.text = t ? `${p.idx}: ${t}${p.text.length > 55 ? '…' : ''}` : p.idx;
    ps.appendChild(o);
  }
  const cs = document.getElementById('cond-select');
  for (const c of M.conditions) {
    const o = document.createElement('option');
    o.value = c; o.text = c; cs.appendChild(o);
  }
  const cols = document.getElementById('cols');
  M.columns.forEach((col, i) => {
    const d = document.createElement('div');
    d.className = 'col ' + col.model + '_' + col.target;
    d.innerHTML = `<div class="col-label" title="${col.label}">${col.label}</div>`
                + `<iframe id="frame-${i}" title="${col.label}"></iframe>`;
    cols.appendChild(d);
  });

  ps.addEventListener('change', update);
  cs.addEventListener('change', update);
  const u = new URLSearchParams(location.search);
  if (u.has('prompt')) ps.value = u.get('prompt');
  if (u.has('cond')) cs.value = u.get('cond');
  update();
}

function update() {
  const idx = document.getElementById('prompt-select').value;
  const cond = document.getElementById('cond-select').value;
  const key = `${idx}_${cond}`;
  const parts = [];
  M.columns.forEach((col, i) => {
    const slug = col.slugmap[key];
    const frame = document.getElementById('frame-' + i);
    if (slug) {
      const url = `./${col.dir}/index.html?slug=${encodeURIComponent(slug)}&compact=1`;
      if (frame.src !== new URL(url, location.href).href) frame.src = url;
      parts.push(slug);
    } else {
      frame.removeAttribute('src');
      parts.push('—');
    }
  });
  document.getElementById('status').textContent = parts.join('  |  ');
  const q = new URLSearchParams();
  q.set('prompt', idx); q.set('cond', cond);
  history.replaceState(null, '', '?' + q.toString());
}

init().catch(e => {
  document.getElementById('status').textContent = 'error: ' + e.message;
  console.error(e);
});
</script>
</body>
</html>
```

- [ ] **Step 2: Validate the harness loads against a synthetic manifest (no GPU, no HF)**

Run:

```bash
python3 - <<'PY'
# Sanity: the harness references the manifest + builds iframe URLs as designed.
from pathlib import Path
html = Path("scripts/pipeline/05_frontend_patches/compare_multi.html").read_text()
for needle in ["compare_manifest.json", "col.slugmap[key]", "&compact=1", "prompt-select", "cond-select"]:
    assert needle in html, needle
print("compare_multi.html structure OK")
PY
```

Expected: `compare_multi.html structure OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/pipeline/05_frontend_patches/compare_multi.html
git commit -m "feat(compare): manifest-driven N-column compare harness"
```

---

## Task 8: RunPod orchestrator + watcher + GPU smoke (Unit B)

**Files:**
- Create: `scripts/emnlp_perm_edit/runpod_gemma_variants.sh`
- Create: `scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh`
- Create: `scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh`

**Interfaces:**
- Consumes: Tasks 1–3 (`ensure_gemma_variant_directions.py`, `verify_variant_nets.py`, `push --run-name`), Stage-02/05 pipeline scripts.
- Produces: 3 HF runs `run_gemma_<v>_L15`; a DONE marker `data/results/pipeline_runs/.GEMMA_VARIANTS_DONE`.

- [ ] **Step 1: Write the orchestrator**

Create `scripts/emnlp_perm_edit/runpod_gemma_variants.sh`:

```bash
#!/bin/bash
# Regenerate Gemma variant attribution graphs (complement/full/outlier), gate
# each against Georg's committed nets, pack+annotate, and push to HF.
# Env knobs: DRY_RUN=1 (print plan only), NO_TMUX=1, VARIANTS=..., BATCH=128,
#   NPROMPTS=50, DATASET_REPO=moon70/refusal-lens-graphs.
set -u
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python3}"
VARIANTS="${VARIANTS:-complement full outlier}"
NPROMPTS="${NPROMPTS:-50}"
BATCH="${BATCH:-128}"
DATASET_REPO="${DATASET_REPO:-moon70/refusal-lens-graphs}"
RUNS=data/results/pipeline_runs
NETS_REF=data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json
DONE_MARKER=$RUNS/.GEMMA_VARIANTS_DONE
FAIL_LOG=$RUNS/.GEMMA_VARIANTS_FAILED.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Self-relaunch into tmux unless told not to.
if [ "${NO_TMUX:-0}" != "1" ] && [ -z "${TMUX:-}" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  exec tmux new-session -s gemma_variants "bash $0"
fi

rm -f "$DONE_MARKER" "$FAIL_LOG"

echo "### ensure variant directions ###"
$PY scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py || { echo "direction build FAILED"; exit 2; }

for v in $VARIANTS; do
  RD=$RUNS/gemma_var_$v
  echo "############ VARIANT=$v  $(date) ############"
  ATTR="$PY scripts/pipeline/02_run_attribution.py --run-dir $RD \
    --n-prompts $NPROMPTS --skip-multi-graph --target-layer 15 \
    --single-position-target -2 --measurement-hook hook_resid_post \
    --backend transformerlens --dtype float32 --batch-size $BATCH \
    --save-graphs --resume"
  GATE="$PY scripts/emnlp_perm_edit/verify_variant_nets.py \
    --attribution-results $RD/02_attribution/attribution_results.json \
    --nets-ref $NETS_REF --variant $v"
  VIZ="$PY scripts/pipeline/05_visualize_circuits.py --run-dir $RD \
    --mode single --skip-subcircuits --gzip"
  PUSH="$PY scripts/pipeline/push_graph_data.py --run-dir $RD --source 05_frontend \
    --dataset-repo $DATASET_REPO --run-name run_gemma_${v}_L15"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: $ATTR"; echo "DRY: $GATE"; echo "DRY: $VIZ"; echo "DRY: $PUSH"; continue
  fi

  # Attribution with OOM retry.
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "---- $v attribution attempt $attempt $(date) ----"
    eval $ATTR && { ok=1; break; }
    echo "retry after crash/OOM in 30s..."; sleep 30
  done
  [ "$ok" = "1" ] || { echo "$v: attribution FAILED" | tee -a "$FAIL_LOG"; continue; }

  echo "---- $v nets gate ----"
  if ! eval $GATE; then
    echo "$v: NETS GATE FAILED (graphs not trusted; check fork/hook/direction)" | tee -a "$FAIL_LOG"
    continue
  fi

  echo "---- $v pack+annotate (05) ----"; eval $VIZ || { echo "$v: 05 FAILED" | tee -a "$FAIL_LOG"; continue; }
  echo "---- $v push to HF ($DATASET_REPO/runs/run_gemma_${v}_L15) ----"
  eval $PUSH || { echo "$v: push FAILED" | tee -a "$FAIL_LOG"; continue; }

  echo "---- $v purge .pt to free disk ----"
  rm -rf "$RD/02_attribution/graphs"
  echo "$v DONE $(date)"
done

if [ "${DRY_RUN:-0}" != "1" ]; then
  touch "$DONE_MARKER"
  echo "############ ALL VARIANTS COMPLETE $(date) ############"
  [ -f "$FAIL_LOG" ] && { echo "FAILURES:"; cat "$FAIL_LOG"; }
fi
```

- [ ] **Step 2: Write the GPU smoke (2 prompts, complement only, push dry-run)**

Create `scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh`:

```bash
#!/bin/bash
# 2-prompt GPU smoke for the Gemma variant pipeline: attribution -> nets gate
# (loose) -> 05 pack/annotate -> push --dry-run, into a throwaway run-dir.
set -eu
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python3}"
NETS_REF=data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json
SMOKE=/tmp/gemma_var_smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE/01_direction/directions" "$SMOKE/01_direction/positions_L15"

echo "### stage complement direction into smoke run-dir ###"
cp data/results/pipeline_runs/gemma_var_complement/01_direction/directions/layer_15.pt "$SMOKE/01_direction/directions/"
cp data/results/pipeline_runs/gemma_var_complement/01_direction/positions_L15/pos_-2.pt "$SMOKE/01_direction/positions_L15/"

echo "### attribution (2 prompts) ###"
$PY scripts/pipeline/02_run_attribution.py --run-dir "$SMOKE" \
  --n-prompts 2 --skip-multi-graph --target-layer 15 --single-position-target -2 \
  --measurement-hook hook_resid_post --backend transformerlens --dtype float32 \
  --batch-size 64 --save-graphs --resume

echo "### nets gate (loose: 2-prompt, sign + magnitude only) ###"
$PY scripts/emnlp_perm_edit/verify_variant_nets.py \
  --attribution-results "$SMOKE/02_attribution/attribution_results.json" \
  --nets-ref "$NETS_REF" --variant complement --corr-min -1.0 --bare-rel-tol 0.6

echo "### pack + annotate ###"
$PY scripts/pipeline/05_visualize_circuits.py --run-dir "$SMOKE" --mode single --skip-subcircuits --gzip

echo "### push (dry-run) ###"
$PY scripts/pipeline/push_graph_data.py --run-dir "$SMOKE" --source 05_frontend \
  --dataset-repo moon70/refusal-lens-graphs --run-name run_gemma_complement_L15 --dry-run

echo "SMOKE TEST PASSED"
```

- [ ] **Step 3: Write the watcher**

Create `scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh`:

```bash
#!/bin/bash
# Poll for the DONE marker; on completion commit the small artifacts to the
# branch (the HF push already happened inside the orchestrator).
set -u
cd "$(git rev-parse --show-toplevel)"
MARKER=data/results/pipeline_runs/.GEMMA_VARIANTS_DONE
echo "watching for $MARKER (5 min poll, 8h timeout)…"
for i in $(seq 1 96); do
  if [ -f "$MARKER" ]; then
    echo "DONE marker seen $(date); committing artifacts."
    git add -A data/results/pipeline_runs/gemma_var_*/02_attribution/attribution_results.json \
                data/results/pipeline_runs/gemma_var_*/05_frontend/data/graph-metadata.json 2>/dev/null || true
    git commit -m "gemma variants: attribution summaries + packed metadata" || echo "(nothing to commit)"
    git push origin HEAD || echo "(push failed; push manually)"
    exit 0
  fi
  sleep 300
done
echo "timeout waiting for DONE marker"; exit 1
```

- [ ] **Step 4: Validate scripts parse + the DRY_RUN plan prints (no GPU)**

Run:

```bash
bash -n scripts/emnlp_perm_edit/runpod_gemma_variants.sh && \
bash -n scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh && \
bash -n scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh && echo "syntax OK"
DRY_RUN=1 NO_TMUX=1 bash scripts/emnlp_perm_edit/runpod_gemma_variants.sh 2>&1 | head -40
```

Expected: `syntax OK`, then the direction self-check line and `DRY: …` command lines for all three variants (attribution / gate / 05 / push), no GPU work.

- [ ] **Step 5: Commit**

```bash
git add scripts/emnlp_perm_edit/runpod_gemma_variants.sh scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh
git commit -m "feat(gemma-variants): RunPod orchestrator + watcher + GPU smoke"
```

---

## Task 9: RunPod runbook

**Files:**
- Create: `docs/RUNPOD_GEMMA_VARIANTS_SETUP.md`

- [ ] **Step 1: Write the runbook**

Create `docs/RUNPOD_GEMMA_VARIANTS_SETUP.md` documenting, adapted from `docs/RUNPOD_QWEN_SUBCIRCUITS_SETUP.md`:
- **Pod:** 48 GB card (A40/A6000), 150 GB network volume at `/workspace`, `HF_HOME=/workspace/hf`.
- **Clone + fork install:** `git clone --recurse-submodules`, `git checkout emnlp-perm-edit`, `uv pip install -e .` and **`uv pip install -e vendor/circuit-tracer`** (branch `refusal-lens-multi-position-fix` — required for `hook_resid_post`).
- **Env:** `HF_TOKEN` (WRITE on `moon70/refusal-lens-graphs`), git remote PAT.
- **Preflight:** `nvidia-smi`; `df -h /workspace`; HF read probe on `moon70/refusal-lens-graphs/runs/run_emnlp_qwen_L18_20260522/` (expect 556 files); HF write probe; `git push --dry-run`; `PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`; `python3 scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py --check-only`; `DRY_RUN=1 NO_TMUX=1 bash scripts/emnlp_perm_edit/runpod_gemma_variants.sh`.
- **Smoke:** `bash scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh` → `SMOKE TEST PASSED`.
- **Launch:** `bash scripts/emnlp_perm_edit/runpod_gemma_variants.sh` (self-tmux), plus the watcher in a second window.
- **Cost/time:** ~3–5 h, ~$2–6; per-variant nets gate must print `"ok": true`.
- **After:** assemble + serve locally (Task 10).

- [ ] **Step 2: Commit**

```bash
git add docs/RUNPOD_GEMMA_VARIANTS_SETUP.md
git commit -m "docs: RunPod runbook for Gemma variant regeneration"
```

---

## Task 10: Integration — run, assemble, serve (operational)

This task is executed once the code lands; it is the end-to-end validation, not new code.

- [ ] **Step 1: (RunPod) full regeneration** — per the runbook: fork install → preflight → smoke → `runpod_gemma_variants.sh`. Confirm each variant prints nets-gate `"ok": true` and the three runs appear at `moon70/refusal-lens-graphs/runs/run_gemma_{complement,full,outlier}_L15/`.

- [ ] **Step 2: (local) assemble the compare site**

Run: `python3 scripts/pipeline/assemble_compare_frontend.py`
Expected: 4 columns fetched; `Assembled 4 columns; 50 shared prompts x 11 conditions.`

- [ ] **Step 3: (local) serve + visual check**

Run:

```bash
cd data/results/compare_3way
python3 -m http.server 8000
# open http://localhost:8000/compare.html
```

Verify: prompt + condition pickers populate; all four columns load matched slugs for `000_bare`; compact mode hides the per-column selector and the gridsnap graph fills each column. If a column's panels don't stack legibly, tune `compact-mode.css` selectors (this is the flagged frontend risk) and re-run `assemble_compare_frontend.py --skip-fetch`.

- [ ] **Step 4: Commit any compact-CSS tuning + final docs**

```bash
git add scripts/pipeline/05_frontend_patches/compact-mode.css docs/
git commit -m "chore(compare): finalize compact column styling after visual check"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** Unit A → Task 1; correctness gate → Task 2; push run-id → Task 3; manifest/assembler → Tasks 4–5; compact mode → Task 6; harness → Task 7; orchestrator/watcher/smoke → Task 8; runbook → Task 9; integration → Task 10. v1 non-goals (no labels, no subcircuits) encoded via `--skip-subcircuits` and unlabeled parity.
- **Type consistency:** `build_variant_directions`/`load_full_direction`/`write_variant_dirs` (Task 1) used identically in Task 8 orchestrator; `extract_nets`/`compare_nets` (Task 2) called by the CLI used in Task 8; `resolve_run_name` (Task 3) consumed by `push --run-name` in Tasks 8/smoke; `parse_condition`/`build_compare_manifest` (Task 4) consumed by `main()` in Task 5; `compare_manifest.json` schema produced in Task 4/5 consumed by `compare_multi.html` in Task 7; `compact=1` produced by Task 7 consumed by Task 6 patch.
- **Placeholders:** none — every code step is complete; the one tunable (compact CSS selectors) ships concrete CSS plus an explicit visual-check tuning gate in Task 10.
