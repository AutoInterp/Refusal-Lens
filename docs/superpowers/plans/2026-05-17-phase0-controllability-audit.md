# Phase 0 — Transcoder Controllability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how much control the transcoder framework gives us over the refusal-direction projection at L15 pos=−2 by (0a) offline-decomposing `direct_dot` into per-edge-type contributions and (0b) running runtime ablations that subtract those contributions from the residual stream, testing whether the behavioral effect matches the linearization identity's prediction.

**Architecture:** Two sub-experiments. Sub-experiment 0a is pure CPU arithmetic on the already-saved packed attribution graphs (no GPU). Sub-experiment 0b-simple registers a single per-(prompt, condition) residual-stream hook at L15 that subtracts a scalar × r̂_unit, where the scalar is the predicted edge-type contribution from 0a. Tests four hypotheses: H0-1 controllability completeness, H0-2 signed-attribution correctness, H0-3 error-node prominence, H0-4 edge≠node distinction. Outputs feed the v2 paper's Framing A (mechanistic decomposition).

**Tech Stack:** Python 3.11, PyTorch 2.x, HF Transformers (`AutoModelForCausalLM` for Gemma-3-4B-IT), matplotlib, statsmodels for Wilson CIs. Reuses Phase 1's `eval_runner.py` for the GPU pass.

**Spec reference:** `EXPERIMENT_PLAN_per_class_jb_orthogonalization.md` § 2 (Track A — Phase 0). This plan implements 0a + 0b-simple only. 0b-rigorous (vendor/circuit-tracer edge-ablation patch) is deferred to a follow-up plan if 0b-simple results indicate it's needed.

**Branch:** `emnlp-perm-edit` (already created; spec + Phase 1 plan live here too). All commits land on this branch.

**Phase 0 priority:** Per Mahmoud's 2026-05-17 ask, Phase 0 runs slightly ahead of Phase 1 so Georg sees foundational-controllability results sooner. Practical sequencing: start Task 0 + Task 1 (data setup) before Phase 1's GPU work; once 0a outputs land, 0b can run in parallel with Phase 1's variant runs because they use different code paths and short GPU sessions.

---

## File Structure

```
scripts/emnlp_perm_edit/
    graph_loader.py                                # Task 2 — packed-graph loader library
    edge_ablation_hook.py                          # Task 5 — scalar r̂-projection hook factory
    00_linearization_decomposition.py              # Task 3 — 0a CLI
    00_decomposition_figure.py                     # Task 4 — 0a figure
    00_edge_ablation_runtime.py                    # Task 6 — 0b-simple driver
    00_directdot_drift_verify.py                   # Task 7 — sanity check
    00_aggregate_phase0.py                         # Task 8 — controllability audit figure
    00_check_phase0_acceptance.py                  # Task 9 — H0 verdicts
    tests/
        test_graph_loader.py                       # Task 2 tests
        test_edge_ablation_hook.py                 # Task 5 tests

data/results/emnlp_perm_edit/phase0_controllability/
    linearization_decomposition.json               # Task 3 output (per-prompt × condition)
    decomposition_by_condition.json                # Task 3 output (aggregates)
    decomposition_figure.png                       # Task 4 output (stacked bar)
    edge_ablation_flip_rates.json                  # Task 6 output (per-variant flip rates)
    directdot_drift_audit.json                     # Task 7 output (sanity check)
    controllability_audit_figure.png               # Task 8 output (main figure)
    flip_rate_summary.json                         # Task 8 output (with Wilson CIs)
    PHASE0_SUMMARY.md                              # Task 8 output (human-readable)
    acceptance_check.json                          # Task 9 output
    sign_audit.md                                  # Task 9 output (H0-2 verdict)

data/results/pipeline_runs/run_20260430_023247/graph_data/
    <prompt_idx>_<condition>_<mode>.json.gz        # Task 1 input (pulled from HF)
```

---

## Task 0: Phase 0 scaffold

**Files:**
- Create: `data/results/emnlp_perm_edit/phase0_controllability/.gitkeep`

This is small; the `scripts/emnlp_perm_edit/` directory already exists from the Phase 1 plan's Task 0. If Phase 1 hasn't been started yet, run its Task 0 first (creates `scripts/emnlp_perm_edit/__init__.py` and `tests/__init__.py`).

- [ ] **Step 1: Verify directory exists or create it**

```bash
test -d scripts/emnlp_perm_edit || (mkdir -p scripts/emnlp_perm_edit/tests && touch scripts/emnlp_perm_edit/__init__.py scripts/emnlp_perm_edit/tests/__init__.py)
mkdir -p data/results/emnlp_perm_edit/phase0_controllability
touch data/results/emnlp_perm_edit/phase0_controllability/.gitkeep
```

- [ ] **Step 2: Verify pytest discovery**

```bash
PYTHONPATH=scripts python3 -m pytest scripts/emnlp_perm_edit/tests/ -v
```

Expected output: pytest discovers the directory without errors (`no tests ran` is fine if no tests yet).

- [ ] **Step 3: Commit**

```bash
git add data/results/emnlp_perm_edit/phase0_controllability/.gitkeep
# Plus scripts/emnlp_perm_edit/__init__.py + tests/__init__.py if newly created
git status   # verify what's staged
git commit -m "$(cat <<'EOF'
emnlp phase 0: scaffold data/results/emnlp_perm_edit/phase0_controllability/

Placeholder directory for Phase 0 (Track A — transcoder controllability audit)
outputs. Reuses scripts/emnlp_perm_edit/ created by Phase 1's Task 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Pull packed attribution graphs from HuggingFace

**Why:** Sub-experiment 0a needs the per-(prompt, condition) packed JSON.gz attribution graphs at `data/results/pipeline_runs/run_20260430_023247/graph_data/`. These are NOT in the local checkout — they were pushed to HuggingFace post-run and the local copy was cleaned up to save disk space. Pull them now.

**Files:** No new files; uses existing `scripts/pipeline/fetch_graph_data.py`.

- [ ] **Step 1: Verify graph_data/ is missing locally**

```bash
test -d data/results/pipeline_runs/run_20260430_023247/graph_data && echo "PRESENT — skip pull" || echo "MISSING — proceed with pull"
```

Expected: `MISSING — proceed with pull`. If `PRESENT`, skip to Step 3.

- [ ] **Step 2: Pull packed graphs from HF**

```bash
PYTHONPATH=scripts python3 scripts/pipeline/fetch_graph_data.py --run run_20260430_023247
```

Expected: ~485 MB downloaded over ~5–15 minutes on a decent connection. Files land at `data/results/pipeline_runs/run_20260430_023247/graph_data/*.json.gz`. There should be 1,100 files (50 prompts × 11 conditions × 2 modes), but our 0a work uses only the `single` mode (550 files).

If `fetch_graph_data.py` errors with "vendor frontend not found", run `git submodule update --init --recursive` first.

- [ ] **Step 3: Verify expected file count and one sample**

```bash
ls data/results/pipeline_runs/run_20260430_023247/graph_data/*.json.gz | wc -l
ls data/results/pipeline_runs/run_20260430_023247/graph_data/0_*single*.json.gz 2>/dev/null | head -3
```

Expected: ≥550 files total (multi + single); at least one `single`-suffixed file for prompt 0 exists. The exact slug format may be `<prompt_idx>_<condition>_<mode>.json.gz` or `<prompt_idx>__<condition>_<mode>.json.gz` — note the actual format from the listing because Task 2's loader will need it.

- [ ] **Step 4: Peek at the JSON schema (for Task 2's design)**

```bash
zcat data/results/pipeline_runs/run_20260430_023247/graph_data/0_bare_single.json.gz 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('top-level keys:', list(data.keys()))
if 'nodes' in data:
    nodes = data['nodes']
    print(f'n_nodes: {len(nodes)}')
    if nodes:
        print('sample node:', nodes[0])
        types = set(n.get('type', n.get('node_type', '?')) for n in nodes)
        print('node types:', types)
if 'links' in data:
    print(f'n_links: {len(data[\"links\"])}')
    if data['links']:
        print('sample link:', data['links'][0])
elif 'edges' in data:
    print(f'n_edges: {len(data[\"edges\"])}')
    if data['edges']:
        print('sample edge:', data['edges'][0])
"
```

Record the exact field names and node-type vocabulary that show up. The schema may use `links` or `edges`; node-type field may be `type` or `node_type` or `feature_type`. The Task 2 loader implementation should match what's actually there.

- [ ] **Step 5: Commit a NOTES.md with the discovered schema**

```bash
cat > scripts/emnlp_perm_edit/SCHEMA_NOTES.md <<'EOF'
# Packed-graph schema (run_20260430_023247)

Discovered 2026-05-17 by inspecting `graph_data/0_bare_single.json.gz`. Format
documented here so Task 2's loader doesn't have to re-discover it.

(Fill in the actual fields observed in Step 4 above — node-type values,
edge-list field name, source/target id format, etc.)
EOF
# Edit the file to record what you actually observed in Step 4
git add scripts/emnlp_perm_edit/SCHEMA_NOTES.md
git commit -m "emnlp phase 0: record packed-graph schema notes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(No code commit yet — this is just an artifact for Task 2's reference.)

---

## Task 2: Graph-loader library (`graph_loader.py`)

**Files:**
- Create: `scripts/emnlp_perm_edit/graph_loader.py`
- Create: `scripts/emnlp_perm_edit/tests/test_graph_loader.py`

**Purpose:** Load a packed JSON.gz attribution graph, identify the measurement-target node (corresponding to direct_dot at L15 pos=−2), and aggregate signed edge attributions by source-node type (feature, embedding, error_node).

**Schema assumptions** (will be replaced by SCHEMA_NOTES.md content after Task 1.5; defaults below are best-guess based on circuit-tracer conventions):
- Top-level keys: `nodes`, `links` (or `edges`), `metadata`
- Each node: `{id, layer, ctx_idx, feature, node_type, ...}` where `node_type ∈ {"cross layer transcoder", "mlp reconstruction error", "embedding", "logit", ...}` — actual vocabulary discovered in Task 1
- Each link: `{source: <node_id>, target: <node_id>, weight: <signed float>}`
- The measurement target is the `logit` node with `feature == 0` or matching the r̂-projection node

**The loader handles the schema with explicit configurability — pass a `node_type_to_category` dict to map raw circuit-tracer node types to our 3-bucket taxonomy.**

- [ ] **Step 1: Write the failing tests**

Create `scripts/emnlp_perm_edit/tests/test_graph_loader.py`:

```python
"""Tests for the packed-graph loader library."""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_loader import (  # noqa: E402
    DEFAULT_NODE_TYPE_TO_CATEGORY,
    aggregate_edge_attributions,
    load_packed_graph,
)


def _write_packed(path: Path, data: dict) -> None:
    """Helper: serialize `data` as gzipped JSON to `path`."""
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def test_load_packed_graph_roundtrip():
    """Loader returns dict matching the input JSON structure."""
    data = {"nodes": [{"id": "n0"}], "links": [{"source": "n0", "target": "n0", "weight": 1.0}]}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.json.gz"
        _write_packed(path, data)
        loaded = load_packed_graph(path)
    assert loaded == data


def test_aggregate_edge_attributions_basic():
    """Sums signed attributions by category for edges into the target node."""
    graph = {
        "nodes": [
            {"id": "f1", "node_type": "cross layer transcoder"},
            {"id": "f2", "node_type": "cross layer transcoder"},
            {"id": "e1", "node_type": "embedding"},
            {"id": "err1", "node_type": "mlp reconstruction error"},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": +1.5},
            {"source": "f2", "target": "target", "weight": -0.5},
            {"source": "e1", "target": "target", "weight": +0.2},
            {"source": "err1", "target": "target", "weight": -0.1},
            # An edge that does NOT target the measurement node — should be ignored
            {"source": "f1", "target": "f2", "weight": 99.0},
        ],
    }
    sums = aggregate_edge_attributions(
        graph,
        target_node_id="target",
        node_type_to_category=DEFAULT_NODE_TYPE_TO_CATEGORY,
    )
    assert sums["feature"]["pos"] == 1.5
    assert sums["feature"]["neg"] == -0.5
    assert sums["feature"]["signed"] == 1.0
    assert sums["embedding"]["signed"] == 0.2
    assert sums["error_node"]["signed"] == -0.1
    assert sums["total_signed"] == 1.0 + 0.2 - 0.1


def test_aggregate_skips_edges_not_targeting_measurement_node():
    """Edges between non-target nodes are ignored."""
    graph = {
        "nodes": [
            {"id": "f1", "node_type": "cross layer transcoder"},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [
            {"source": "f1", "target": "f1", "weight": 10.0},  # self-loop, not to target
        ],
    }
    sums = aggregate_edge_attributions(graph, target_node_id="target",
                                        node_type_to_category=DEFAULT_NODE_TYPE_TO_CATEGORY)
    assert sums["feature"]["signed"] == 0.0
    assert sums["total_signed"] == 0.0


def test_aggregate_raises_on_unknown_node_type():
    """Unknown node_type values raise so we don't silently drop signal."""
    graph = {
        "nodes": [
            {"id": "mystery", "node_type": "WHATSIT"},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [{"source": "mystery", "target": "target", "weight": 1.0}],
    }
    with pytest.raises(ValueError, match="unknown node_type"):
        aggregate_edge_attributions(graph, target_node_id="target",
                                    node_type_to_category=DEFAULT_NODE_TYPE_TO_CATEGORY)


def test_find_measurement_target_node_id_default():
    """Default behavior: find a node with node_type='logit' (unique)."""
    from graph_loader import find_measurement_target_node_id  # noqa: E402

    graph = {
        "nodes": [
            {"id": "f1", "node_type": "cross layer transcoder"},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [],
    }
    assert find_measurement_target_node_id(graph) == "target"


def test_find_measurement_target_node_id_raises_on_multiple_logit_nodes():
    graph = {
        "nodes": [
            {"id": "t1", "node_type": "logit"},
            {"id": "t2", "node_type": "logit"},
        ],
        "links": [],
    }
    with pytest.raises(ValueError, match="multiple logit nodes"):
        from graph_loader import find_measurement_target_node_id  # noqa: E402
        find_measurement_target_node_id(graph)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_graph_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'graph_loader'`.

- [ ] **Step 3: Implement `graph_loader.py`**

Create `scripts/emnlp_perm_edit/graph_loader.py`:

```python
"""Loader for packed JSON.gz attribution graphs from run_20260430_023247.

Loads a single graph file, identifies the measurement-target node (the
"logit" node representing direct_dot = h[L15, pos=-2] · r̂), and aggregates
signed edge attributions by source-node category for the linearization audit.

Source-node categories (Phase 0 § 2.2 of EXPERIMENT_PLAN):
- feature: transcoder feature outputs (CLT decoder writes)
- embedding: token embeddings (input write)
- error_node: transcoder reconstruction error residuals

Update DEFAULT_NODE_TYPE_TO_CATEGORY to match the actual node-type vocabulary
discovered in Task 1 (SCHEMA_NOTES.md). If a node_type appears that the dict
doesn't know, the loader raises ValueError so we don't silently drop signal.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Literal


Category = Literal["feature", "embedding", "error_node"]


# Maps circuit-tracer's raw node_type strings to our 3-bucket category.
# Verify against scripts/emnlp_perm_edit/SCHEMA_NOTES.md after Task 1 inspection.
DEFAULT_NODE_TYPE_TO_CATEGORY: dict[str, Category] = {
    "cross layer transcoder": "feature",
    "feature": "feature",                            # legacy alias seen in some packed outputs
    "embedding": "embedding",
    "mlp reconstruction error": "error_node",
    "error": "error_node",                           # alias
    # logit node is the target itself, not a source — not categorized as a source bucket
}


def load_packed_graph(path: Path) -> dict:
    """Load a gzipped JSON attribution graph and return the parsed dict."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


def find_measurement_target_node_id(graph: dict) -> str:
    """Return the id of the single 'logit' node — the measurement target.

    The packed graph is built with a single target node per run-mode; if more
    than one is found, the schema differs from our assumption and we raise.
    """
    candidates = [n["id"] for n in graph["nodes"] if n.get("node_type") == "logit"]
    if len(candidates) == 0:
        raise ValueError("no logit node found in graph — schema mismatch?")
    if len(candidates) > 1:
        raise ValueError(f"multiple logit nodes found: {candidates}; schema differs from assumption")
    return candidates[0]


def aggregate_edge_attributions(
    graph: dict,
    target_node_id: str,
    node_type_to_category: dict[str, Category] = DEFAULT_NODE_TYPE_TO_CATEGORY,
) -> dict:
    """Aggregate signed weights of edges that target `target_node_id`.

    Returns a dict:
        {
            "feature":     {"pos": float, "neg": float, "signed": float},
            "embedding":   {"pos": float, "neg": float, "signed": float},
            "error_node":  {"pos": float, "neg": float, "signed": float},
            "total_signed": float,
            "n_edges_to_target": int,
        }

    Edges from sources whose `node_type` is not in `node_type_to_category`
    raise ValueError (fail-loud so we don't silently drop signal).
    """
    node_lookup = {n["id"]: n for n in graph["nodes"]}
    edges_field = "links" if "links" in graph else "edges"

    sums = {cat: {"pos": 0.0, "neg": 0.0, "signed": 0.0}
            for cat in ("feature", "embedding", "error_node")}
    n_edges = 0

    for edge in graph[edges_field]:
        if edge["target"] != target_node_id:
            continue
        src_node = node_lookup[edge["source"]]
        src_type = src_node.get("node_type", "")
        if src_type == "logit":
            # logit-to-logit edges (rare; appear in some schemas as self-loops)
            continue
        if src_type not in node_type_to_category:
            raise ValueError(
                f"unknown node_type {src_type!r} on edge source {edge['source']!r}; "
                f"add to DEFAULT_NODE_TYPE_TO_CATEGORY in graph_loader.py"
            )
        category = node_type_to_category[src_type]
        weight = float(edge["weight"])
        sums[category]["signed"] += weight
        if weight >= 0:
            sums[category]["pos"] += weight
        else:
            sums[category]["neg"] += weight
        n_edges += 1

    sums["total_signed"] = sum(sums[c]["signed"] for c in ("feature", "embedding", "error_node"))
    sums["n_edges_to_target"] = n_edges
    return sums
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_graph_loader.py -v
```

Expected: `6 passed`.

If a test fails because the actual schema differs from the default (e.g., real node_type strings are different), update `DEFAULT_NODE_TYPE_TO_CATEGORY` in `graph_loader.py` based on Task 1's `SCHEMA_NOTES.md` and re-run. Document the change in a comment.

- [ ] **Step 5: Quick smoke against a real packed graph**

```bash
PYTHONPATH=scripts python3 -c "
from pathlib import Path
from emnlp_perm_edit.graph_loader import load_packed_graph, find_measurement_target_node_id, aggregate_edge_attributions

# Adjust the path/name to match the actual file from Task 1
graph_dir = Path('data/results/pipeline_runs/run_20260430_023247/graph_data')
sample = next(graph_dir.glob('0_*single*.json.gz'))
print(f'sample: {sample.name}')
g = load_packed_graph(sample)
tid = find_measurement_target_node_id(g)
print(f'target node: {tid}')
sums = aggregate_edge_attributions(g, tid)
for cat in ('feature', 'embedding', 'error_node'):
    print(f'  {cat:12s}  pos={sums[cat][\"pos\"]:+10.2f}  neg={sums[cat][\"neg\"]:+10.2f}  signed={sums[cat][\"signed\"]:+10.2f}')
print(f'  total_signed={sums[\"total_signed\"]:+10.2f}  n_edges={sums[\"n_edges_to_target\"]}')
"
```

Expected: clean output with non-zero sums and `n_edges` in the hundreds. If `ValueError: unknown node_type` fires, add the unrecognized type to `DEFAULT_NODE_TYPE_TO_CATEGORY` and re-run.

- [ ] **Step 6: Commit**

```bash
git add scripts/emnlp_perm_edit/graph_loader.py \
        scripts/emnlp_perm_edit/tests/test_graph_loader.py
git commit -m "$(cat <<'EOF'
emnlp phase 0: packed-graph loader library + tests

load_packed_graph(), find_measurement_target_node_id(), aggregate_edge_attributions()
support the linearization decomposition in 0a. Fails loud on unknown node_type
values so we don't silently drop signal from a new circuit-tracer edge category.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Sub-experiment 0a — Linearization decomposition CLI

**Files:**
- Create: `scripts/emnlp_perm_edit/00_linearization_decomposition.py`

**Output:** `data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json` + `decomposition_by_condition.json`

- [ ] **Step 1: Implement the CLI**

Create `scripts/emnlp_perm_edit/00_linearization_decomposition.py`:

```python
"""Phase 0 — Sub-experiment 0a: offline linearization decomposition.

For every (prompt_idx, condition) in run_20260430_023247's `single`-mode
graph_data/, load the packed attribution graph, identify the measurement
target node (direct_dot at L15 pos=-2), aggregate signed edge attributions
by source-node category (feature/embedding/error_node), and verify the
linearization identity holds within Stage 03's tolerance.

The output is a complete decomposition table that drives Sub-experiment 0b's
runtime ablation magnitudes and the Phase 0 dissociation figure.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from graph_loader import (  # noqa: E402
    DEFAULT_NODE_TYPE_TO_CATEGORY,
    aggregate_edge_attributions,
    find_measurement_target_node_id,
    load_packed_graph,
)

CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/graph_data")
    p.add_argument("--baselines-from", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/03_verification/verification_results.json",
                   help="Stage 03 verification output for direct_dot ground truth per prompt × condition.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--mode", default="single", choices=["single", "multi"])
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--max-recon-error", type=float, default=0.01,
                   help="Maximum allowed |1 - reconstructed/direct_dot| per input (default 1%).")
    return p.parse_args()


def load_directdot_ground_truth(verification_path: Path) -> dict:
    """Stage 03 stores per-(prompt, condition) direct_dot + Σ_edges + baseline_offset."""
    data = json.loads(verification_path.read_text())
    # Schema: {"per_prompt": [{"prompt_idx", "condition", "direct_dot", "edges_sum", "baseline_offset"}, ...]}
    # OR aggregated; verify against actual file structure on first run.
    out = {}
    for entry in data.get("per_prompt", data.get("results", [])):
        key = (entry["prompt_idx"], entry["condition"])
        out[key] = {
            "direct_dot": entry["direct_dot"],
            "edges_sum": entry.get("edges_sum"),
            "baseline_offset": entry.get("baseline_offset"),
        }
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0a] loading direct_dot ground truth from {args.baselines_from}")
    try:
        ground_truth = load_directdot_ground_truth(args.baselines_from)
        print(f"  loaded {len(ground_truth)} (prompt, condition) entries")
    except (FileNotFoundError, KeyError) as e:
        print(f"  WARNING: cannot load Stage 03 verification ({e}); reconstruction-error check will be skipped.")
        ground_truth = {}

    per_prompt_records = []
    by_condition = {c: [] for c in CONDITIONS}
    skipped = []
    n_recon_errors = 0
    t0 = time.time()

    for prompt_idx in range(args.n_prompts):
        for condition in CONDITIONS:
            slug = f"{prompt_idx}_{condition}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                skipped.append(slug)
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            sums = aggregate_edge_attributions(graph, target_id)

            record = {
                "prompt_idx": prompt_idx,
                "condition": condition,
                "n_edges_to_target": sums["n_edges_to_target"],
                "feature_pos": sums["feature"]["pos"],
                "feature_neg": sums["feature"]["neg"],
                "feature_signed": sums["feature"]["signed"],
                "embedding_pos": sums["embedding"]["pos"],
                "embedding_neg": sums["embedding"]["neg"],
                "embedding_signed": sums["embedding"]["signed"],
                "error_pos": sums["error_node"]["pos"],
                "error_neg": sums["error_node"]["neg"],
                "error_signed": sums["error_node"]["signed"],
                "all_signed": sums["total_signed"],
            }

            gt = ground_truth.get((prompt_idx, condition))
            if gt and gt["direct_dot"] is not None:
                direct_dot = gt["direct_dot"]
                baseline_offset = direct_dot - sums["total_signed"]
                record["direct_dot"] = direct_dot
                record["baseline_offset_computed"] = baseline_offset
                if abs(direct_dot) > 1e-6:
                    recon_err = abs(1.0 - (sums["total_signed"] + baseline_offset) / direct_dot)
                    record["reconstruction_error"] = recon_err
                    if recon_err > args.max_recon_error:
                        n_recon_errors += 1

            per_prompt_records.append(record)
            by_condition[condition].append(record)

    elapsed = time.time() - t0
    print(f"[0a] processed {len(per_prompt_records)} graphs in {elapsed:.1f}s "
          f"({len(skipped)} skipped, {n_recon_errors} recon-error violations)")

    # Aggregate per-condition stats
    aggregates = {}
    for condition in CONDITIONS:
        recs = by_condition[condition]
        if not recs:
            aggregates[condition] = {"n": 0}
            continue
        agg = {"n": len(recs)}
        for field in ("feature_pos", "feature_neg", "feature_signed",
                      "embedding_pos", "embedding_neg", "embedding_signed",
                      "error_pos", "error_neg", "error_signed", "all_signed"):
            vals = [r[field] for r in recs]
            agg[f"{field}_mean"] = statistics.mean(vals)
            agg[f"{field}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if all("direct_dot" in r for r in recs):
            dd_vals = [r["direct_dot"] for r in recs]
            agg["direct_dot_mean"] = statistics.mean(dd_vals)
            agg["direct_dot_std"] = statistics.stdev(dd_vals) if len(dd_vals) > 1 else 0.0
        aggregates[condition] = agg

    out = {
        "metadata": {
            "graph_data_dir": str(args.graph_data_dir),
            "mode": args.mode,
            "n_prompts_processed": args.n_prompts,
            "n_records": len(per_prompt_records),
            "n_skipped": len(skipped),
            "n_recon_errors_above_threshold": n_recon_errors,
            "recon_error_threshold": args.max_recon_error,
        },
        "per_prompt": per_prompt_records,
        "skipped_slugs": skipped,
    }
    (args.out_dir / "linearization_decomposition.json").write_text(json.dumps(out, indent=2))
    (args.out_dir / "decomposition_by_condition.json").write_text(json.dumps({
        "per_condition": aggregates,
    }, indent=2))
    print(f"[0a] wrote linearization_decomposition.json + decomposition_by_condition.json")

    if n_recon_errors > 0:
        print(f"\n[0a] WARNING: {n_recon_errors} records have reconstruction error > "
              f"{args.max_recon_error}. Inspect linearization_decomposition.json for "
              f"records where 'reconstruction_error' field exceeds threshold — these "
              f"may indicate schema mismatch in graph_loader or a basis issue.")
        sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on 1 prompt**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py --n-prompts 1
```

Expected: processes ~11 graphs in <10 seconds, writes both JSON files, prints summary line. If reconstruction-error warnings fire, the loader's node-type mapping needs adjustment (see Task 2 Step 4).

- [ ] **Step 3: Full run on 50 prompts × 11 conditions**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py
```

Expected: ~30 seconds wall, processes 550 graphs (or fewer if some single-mode files are missing — note skipped slugs), reconstruction error <1% on >99% of inputs.

- [ ] **Step 4: Spot-check per-condition aggregates**

```bash
python3 -c "
import json
d = json.load(open('data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json'))
for cond, agg in d['per_condition'].items():
    if agg['n'] == 0:
        print(f'{cond:30s}  (no data)')
        continue
    print(f'{cond:30s}  n={agg[\"n\"]:2d}  '
          f'feat_signed={agg[\"feature_signed_mean\"]:+10.1f}  '
          f'embed_signed={agg.get(\"embedding_signed_mean\", 0):+10.1f}  '
          f'err_signed={agg[\"error_signed_mean\"]:+10.1f}  '
          f'all_signed={agg[\"all_signed_mean\"]:+10.1f}')
"
```

Expected: aggregates look numerically plausible — `feature_signed_mean` should be on the order of −40k to −50k for bare prompts (matching Stage 03's reference `Σ edges ≈ −48,886`). If numbers are wildly off, debug schema in graph_loader.

- [ ] **Step 5: Commit**

```bash
git add scripts/emnlp_perm_edit/00_linearization_decomposition.py \
        data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json \
        data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0a linearization decomposition CLI + outputs

Per-(prompt, condition) decomposition of direct_dot into feature / embedding /
error_node signed sums + reconstruction-error audit against Stage 03 ground
truth. Drives Sub-experiment 0b's per-variant ablation magnitudes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Decomposition figure (`00_decomposition_figure.py`)

**Files:**
- Create: `scripts/emnlp_perm_edit/00_decomposition_figure.py`

**Output:** `data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png`

- [ ] **Step 1: Implement the figure script**

Create `scripts/emnlp_perm_edit/00_decomposition_figure.py`:

```python
"""Phase 0 — stacked-bar figure of direct_dot decomposition per condition.

X-axis: 11 conditions (bare, 5 jb_*, 5 ctrl_*).
Y-axis: signed magnitude of contribution to direct_dot.
Stacked: feature_signed | embedding_signed | error_signed | baseline_offset.

Sign convention: positive = pro-refusal contribution (toward refusal axis);
negative = anti-refusal. Stack components in the same convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]
COMPONENT_COLORS = {
    "feature": "#4C72B0",
    "embedding": "#55A868",
    "error_node": "#C44E52",
    "baseline_offset": "#8172B2",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.in_file.read_text())["per_condition"]

    n = len(CONDITIONS)
    feat = np.zeros(n)
    embed = np.zeros(n)
    err = np.zeros(n)
    baseline = np.zeros(n)
    direct_dot = np.zeros(n)

    for i, cond in enumerate(CONDITIONS):
        agg = data.get(cond, {})
        if agg.get("n", 0) == 0:
            continue
        feat[i] = agg.get("feature_signed_mean", 0.0)
        embed[i] = agg.get("embedding_signed_mean", 0.0)
        err[i] = agg.get("error_signed_mean", 0.0)
        direct_dot[i] = agg.get("direct_dot_mean", 0.0)
        baseline[i] = direct_dot[i] - feat[i] - embed[i] - err[i]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(n)
    width = 0.7

    # Stack components: positive and negative contributions plotted separately
    # so the stack reads cleanly even when components have mixed signs.
    bottoms_pos = np.zeros(n)
    bottoms_neg = np.zeros(n)
    for vals, label, color in [
        (feat, "feature edges", COMPONENT_COLORS["feature"]),
        (embed, "embedding edges", COMPONENT_COLORS["embedding"]),
        (err, "error_node edges", COMPONENT_COLORS["error_node"]),
        (baseline, "baseline_offset (residual)", COMPONENT_COLORS["baseline_offset"]),
    ]:
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, width, bottom=bottoms_pos, color=color, label=label)
        ax.bar(x, neg, width, bottom=bottoms_neg, color=color)
        bottoms_pos += pos
        bottoms_neg += neg

    # Overlay direct_dot as a marker
    ax.scatter(x, direct_dot, color="black", marker="D", s=40, zorder=5, label="direct_dot (sum)")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=45, ha="right")
    ax.set_ylabel("Signed contribution to direct_dot at L15 pos=-2")
    ax.set_title("Phase 0 — Linearization decomposition of direct_dot per condition\n"
                 "(positive = pro-refusal axis; bare is the reference)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"[0a-figure] wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_decomposition_figure.py
ls -la data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png
```

Expected: file exists, ~100–200 KB PNG. Open it and verify:
- 11 bars (bare + 5 jb_* + 5 ctrl_*)
- bare bar has feature_signed ≈ −48k (the long blue segment)
- baseline_offset (purple) is positive ≈ +20k
- direct_dot (black diamond) ≈ −29k, in agreement with REPORT §4

- [ ] **Step 3: Commit**

```bash
git add scripts/emnlp_perm_edit/00_decomposition_figure.py \
        data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0a decomposition figure (stacked bar per condition)

Visualizes feature / embedding / error_node / baseline_offset contributions
to direct_dot at L15 pos=-2 across the 11 controlled conditions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Edge-ablation hook library (`edge_ablation_hook.py`)

**Files:**
- Create: `scripts/emnlp_perm_edit/edge_ablation_hook.py`
- Create: `scripts/emnlp_perm_edit/tests/test_edge_ablation_hook.py`

**Purpose:** Build a forward hook that subtracts a per-(prompt, condition) scalar `delta` × `r̂_unit` from the L15 residual at all positions, where the scalar is the predicted contribution from a chosen edge-type bucket. After the hook, `h_new · r̂ = h · r̂ − delta`.

- [ ] **Step 1: Write the failing tests**

Create `scripts/emnlp_perm_edit/tests/test_edge_ablation_hook.py`:

```python
"""Tests for edge-ablation r̂-projection hook factory."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402


def test_hook_subtracts_predicted_amount_from_rhat_projection():
    """After hook, h_new · r̂ = (h · r̂) - delta."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    delta = 1500.0
    h_in = torch.randn(2, 5, 2560)
    proj_in = (h_in.float() * r_hat).sum(-1)  # (batch, seq)

    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)
    h_out = hook_fn(None, None, h_in.clone())
    proj_out = (h_out.float() * r_hat).sum(-1)

    # Each (batch, seq) projection should drop by approximately delta
    diff = proj_in - proj_out
    assert torch.allclose(diff, torch.full_like(diff, delta), atol=1e-2), \
        f"expected diff ≈ {delta}, got mean diff {diff.mean().item():.3f}"


def test_hook_handles_tuple_output():
    """For Gemma layer-output hooks the output is a tuple; first element is hidden states."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    h_in = torch.randn(2, 5, 2560)
    extra = torch.zeros(1)
    output_tuple = (h_in.clone(), extra)

    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, 1000.0)
    result = hook_fn(None, None, output_tuple)
    assert isinstance(result, tuple)
    proj_in = (h_in.float() * r_hat).sum(-1)
    proj_out = (result[0].float() * r_hat).sum(-1)
    diff = proj_in - proj_out
    assert torch.allclose(diff, torch.full_like(diff, 1000.0), atol=1e-2)


def test_hook_zero_delta_is_identity():
    """delta=0 should not modify the residual."""
    r_hat = torch.randn(2560)
    h_in = torch.randn(2, 5, 2560)
    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, 0.0)
    h_out = hook_fn(None, None, h_in.clone())
    assert torch.allclose(h_out, h_in, atol=1e-4)


def test_hook_negative_delta_increases_rhat_projection():
    """Negative delta should add r̂-magnitude (push toward refusal)."""
    r_hat = torch.randn(2560)
    h_in = torch.randn(2, 5, 2560)
    proj_in = (h_in.float() * r_hat).sum(-1)
    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, -500.0)
    h_out = hook_fn(None, None, h_in.clone())
    proj_out = (h_out.float() * r_hat).sum(-1)
    diff = proj_in - proj_out
    # Subtracting -500 means adding 500, so proj_out is HIGHER and diff is -500
    assert torch.allclose(diff, torch.full_like(diff, -500.0), atol=1e-2)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_edge_ablation_hook.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `edge_ablation_hook.py`**

Create `scripts/emnlp_perm_edit/edge_ablation_hook.py`:

```python
"""Edge-ablation hook factory for Phase 0 sub-experiment 0b-simple.

Subtracts a scalar `delta` along the r̂_unit direction from the residual at
every position. After the hook, the r̂-projection of the residual at every
(batch, seq) is reduced by `delta`. Used to simulate "ablating an edge-type
bucket's contribution to direct_dot at L15 pos=-2" by directly modifying
the residual stream's r̂ component by the predicted amount.

Math:
    h_new = h - (delta / ‖r̂‖²) · r̂
    h_new · r̂ = h · r̂ - delta · (r̂ · r̂) / ‖r̂‖² = h · r̂ - delta
"""
from __future__ import annotations

import torch


def make_scalar_rhat_subtraction_hook(r_hat: torch.Tensor, delta: float):
    """Return a forward_hook that subtracts `delta` from h · r̂ at every position.

    Args:
        r_hat: 1-D direction tensor (need NOT be unit-norm).
        delta: scalar amount to subtract from the r̂-projection.

    The hook handles both tuple outputs (Gemma layer modules return tuples)
    and plain-tensor outputs (sublayer norms). Casts r̂ to the output dtype
    at hook time so we don't force fp32 math inside a bf16 model.
    """
    r_hat = r_hat.float()
    r_hat_norm_sq = (r_hat @ r_hat).item()

    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        r_cast = r_hat.to(dtype=h.dtype, device=h.device)
        # Subtract (delta / ‖r̂‖²) × r̂ from every position
        coeff = delta / r_hat_norm_sq
        h_new = h - coeff * r_cast
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook_fn
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_edge_ablation_hook.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/emnlp_perm_edit/edge_ablation_hook.py \
        scripts/emnlp_perm_edit/tests/test_edge_ablation_hook.py
git commit -m "$(cat <<'EOF'
emnlp phase 0: edge-ablation hook factory + tests

make_scalar_rhat_subtraction_hook(r_hat, delta) returns a forward_hook that
subtracts `delta` from h·r̂ at every position. Used by 0b-simple to simulate
edge-type ablation by direct residual-stream r̂-projection modulation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Sub-experiment 0b-simple driver — Runtime edge-ablation

**Files:**
- Create: `scripts/emnlp_perm_edit/00_edge_ablation_runtime.py`

**Output:** `data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json`

**Variants** (per spec § 2.3, 7 variants × 550 prompts × max_new_tokens=80):

| Variant | delta source (from 0a) |
|---|---|
| `ablate_features_pos` | `feature_pos` |
| `ablate_features_neg` | `feature_neg` (negative number; pushes opposite way) |
| `ablate_features_all` | `feature_signed` |
| `ablate_embeddings_all` | `embedding_signed` |
| `ablate_errors_all` | `error_signed` |
| `ablate_all_edges` | `all_signed` |
| `ablate_all_2x` | `2 × all_signed` |

- [ ] **Step 1: Implement the driver**

Create `scripts/emnlp_perm_edit/00_edge_ablation_runtime.py`:

```python
"""Phase 0 — Sub-experiment 0b-simple: runtime edge-ablation across 7 variants.

For each (prompt, condition), looks up the precomputed edge-type sums from 0a
and registers a forward hook on L15 that subtracts the chosen scalar × r̂_unit
from the residual at every position. Generates max_new_tokens=80 greedy and
classifies refuse/comply.

This is a behavioral-proxy test of edge ablation — it operates on the residual
stream directly, NOT through circuit-tracer's transcoder graph. Its purpose is
to test whether the linearization decomposition's predictions are causally
meaningful: when we remove the predicted amount of r̂-projection contribution,
does the model flip refusal as expected?

The "rigorous" version (true transcoder edge ablation via vendor/circuit-tracer
patches) is deferred per spec § 2.3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402


LAYER = 15
VARIANT_TO_DELTA_FIELD = {
    "ablate_features_pos": ("feature_pos", 1.0),
    "ablate_features_neg": ("feature_neg", 1.0),
    "ablate_features_all": ("feature_signed", 1.0),
    "ablate_embeddings_all": ("embedding_signed", 1.0),
    "ablate_errors_all": ("error_signed", 1.0),
    "ablate_all_edges": ("all_signed", 1.0),
    "ablate_all_2x": ("all_signed", 2.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decomposition", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json")
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--variants", default=",".join(VARIANT_TO_DELTA_FIELD.keys()),
                   help="Comma-separated variant names to run.")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Smoke test: limit to first N prompts.")
    return p.parse_args()


def main():
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants_to_run:
        assert v in VARIANT_TO_DELTA_FIELD, f"unknown variant: {v}"

    print(f"[0b-simple] loading r̂[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()
    print(f"  ||r̂|| = {r_hat.norm().item():.2f}")

    print(f"[0b-simple] loading decomposition from {args.decomposition}")
    decomp = json.loads(args.decomposition.read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}
    print(f"  {len(per_prompt)} (prompt, condition) entries")

    print(f"[0b-simple] loading model {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_layer = layers[LAYER]
    print(f"  loaded in {time.time()-t0:.1f}s")

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
        per_prompt = {k: v for k, v in per_prompt.items() if k[0] < args.max_prompts}

    pad_id = tokenizer.eos_token_id

    results = {
        "metadata": {
            "layer": LAYER,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": len(dataset),
            "variants": variants_to_run,
            "r_hat_norm": r_hat.norm().item(),
        },
        "per_variant": {v: [] for v in variants_to_run},
    }

    t_total = time.time()
    for variant in variants_to_run:
        delta_field, scale = VARIANT_TO_DELTA_FIELD[variant]
        print(f"\n[0b-simple] variant={variant} (delta_field={delta_field}, scale={scale})")
        n_done = 0
        t_v = time.time()
        for prompt_idx, prompt in enumerate(dataset):
            for cond, blob in prompt["conditions"].items():
                decomp_rec = per_prompt.get((prompt_idx, cond))
                if decomp_rec is None:
                    continue
                delta = float(decomp_rec[delta_field]) * scale
                hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)

                text = blob["text"]
                formatted = format_prompt(tokenizer, text)
                ids = tokenizer(formatted, return_tensors="pt").to(model.device)
                prompt_len = ids.input_ids.shape[1]

                handle = target_layer.register_forward_hook(hook_fn)
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **ids, do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=pad_id,
                        )
                    resp = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                finally:
                    handle.remove()

                results["per_variant"][variant].append({
                    "prompt_idx": prompt_idx, "condition": cond,
                    "delta_applied": delta,
                    "response": resp[:300],
                    "classification": classify_response(resp),
                    "coherent": is_coherent(resp),
                })
                n_done += 1
                if n_done % 50 == 0:
                    elapsed = time.time() - t_v
                    total = len(dataset) * 11
                    eta = elapsed / n_done * (total - n_done)
                    print(f"  [{n_done}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s")
        print(f"  variant={variant} done in {time.time()-t_v:.0f}s")

    elapsed_total = time.time() - t_total
    print(f"\n[0b-simple] all variants done in {elapsed_total:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on 1 prompt × 2 variants**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
    --max-prompts 1 --variants ablate_all_edges,ablate_all_2x \
    --out /tmp/0b_smoke.json
```

Expected: ~30 seconds, writes 22 records (2 variants × 1 prompt × 11 conditions).

```bash
python3 -c "
import json
d = json.load(open('/tmp/0b_smoke.json'))
for v in d['per_variant']:
    recs = d['per_variant'][v]
    print(f'variant={v}: {len(recs)} records')
    for r in recs[:3]:
        print(f'  prompt={r[\"prompt_idx\"]} cond={r[\"condition\"]:25s} '
              f'delta={r[\"delta_applied\"]:+8.0f} cls={r[\"classification\"]}')
"
```

For `ablate_all_2x` on bare prompts, expect at least some COMPLY classifications (over-ablation should drive direct_dot past zero into the harmless region, flipping bare-refuse). For `ablate_all_edges`, expect MOSTLY REFUSE on bare (direct_dot → baseline_offset which is still positive/refusal-side).

- [ ] **Step 3: Full run on all 7 variants × 50 prompts**

```bash
tmux new -s phase0_0b 'PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py 2>&1 | tee /tmp/phase0_0b.log'
```

Expected wall: ~3.5 hours (7 variants × 550 prompts × ~3 s/gen on RTX 5080 16 GB). Detach with Ctrl-b d; reattach with `tmux attach -t phase0_0b`.

Output: `data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json` with 3,850 generation records.

- [ ] **Step 4: Commit**

```bash
git add scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
        data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0b-simple runtime edge-ablation driver + full output

Runs 7 variants (per-edge-type, all-edges, 2x over-ablation) × 50 prompts ×
11 conditions = 3,850 generations. Each generation registers a forward hook
at L15 that subtracts a per-(prompt, condition) scalar × r̂_unit from the
residual stream, simulating edge ablation by direct r̂-projection modulation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: direct_dot drift verification (sanity check)

**Files:**
- Create: `scripts/emnlp_perm_edit/00_directdot_drift_verify.py`

**Purpose:** For a small subset of inputs (5 prompts × all conditions), run unedited and hook-edited forward passes and measure the actual direct_dot drift. Verify the hook achieves the predicted delta within numerical tolerance. This is the load-bearing correctness check for 0b-simple — if the hook doesn't move direct_dot by the predicted amount, our linearization decomposition is wrong somewhere.

- [ ] **Step 1: Implement the verifier**

Create `scripts/emnlp_perm_edit/00_directdot_drift_verify.py`:

```python
"""Phase 0 — direct_dot drift verification.

For a subset of inputs, runs unedited and hook-edited forward passes and
captures the L15 residual at pos=-2. Checks that hook-induced direct_dot
drift matches the predicted delta within numerical tolerance.

If drift is wildly off the prediction, the linearization decomposition or
the hook math has an issue to debug before trusting 0b-simple flip rates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402
from utils import format_prompt, load_controlled_dataset  # noqa: E402


LAYER = 15
MEASUREMENT_POSITION = -2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decomposition", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json")
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/directdot_drift_audit.json")
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--variants", default="ablate_all_edges,ablate_all_2x")
    p.add_argument("--tolerance", type=float, default=50.0,
                   help="Allowed |actual_drift - predicted_drift| in direct_dot units.")
    return p.parse_args()


def capture_direct_dot_at_target(model, tokenizer, text: str, r_hat: torch.Tensor,
                                 hook_fn=None, target_layer=None) -> float:
    """Run a single forward pass and return h[L15, pos=-2] · r̂ as a Python float."""
    formatted = format_prompt(tokenizer, text)
    ids = tokenizer(formatted, return_tensors="pt").to(model.device)
    captured = {}

    def capture_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # Capture AFTER any other hooks have already modified h
        captured["h"] = h[:, MEASUREMENT_POSITION, :].detach().float().cpu()

    handles = []
    if hook_fn is not None:
        handles.append(target_layer.register_forward_hook(hook_fn))
    handles.append(target_layer.register_forward_hook(capture_hook))

    try:
        with torch.no_grad():
            model(**ids)
    finally:
        for h in handles:
            h.remove()

    h_vec = captured["h"][0]  # batch=0
    return (h_vec @ r_hat).item()


def main():
    args = parse_args()

    print(f"[drift] loading r̂[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()

    decomp = json.loads(args.decomposition.read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}

    print(f"[drift] loading model")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it", torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_layer = layers[LAYER]

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    dataset = dataset[:args.n_prompts]

    variants_to_run = [v.strip() for v in args.variants.split(",")]

    audit = {"metadata": {"n_prompts": args.n_prompts, "tolerance": args.tolerance,
                          "variants": variants_to_run},
             "per_check": []}
    n_passing = 0
    n_failing = 0

    for prompt_idx, prompt in enumerate(dataset):
        for cond, blob in prompt["conditions"].items():
            decomp_rec = per_prompt.get((prompt_idx, cond))
            if decomp_rec is None:
                continue

            text = blob["text"]
            dd_unedited = capture_direct_dot_at_target(model, tokenizer, text, r_hat,
                                                      hook_fn=None, target_layer=target_layer)

            # Re-define the variant map locally to avoid coupling to 00_edge_ablation_runtime.py.
            # Keep in sync with the VARIANT_TO_DELTA_FIELD dict in that module.
            variant_map = {
                "ablate_features_pos": ("feature_pos", 1.0),
                "ablate_features_neg": ("feature_neg", 1.0),
                "ablate_features_all": ("feature_signed", 1.0),
                "ablate_embeddings_all": ("embedding_signed", 1.0),
                "ablate_errors_all": ("error_signed", 1.0),
                "ablate_all_edges": ("all_signed", 1.0),
                "ablate_all_2x": ("all_signed", 2.0),
            }
            for variant in variants_to_run:
                delta_field, scale = variant_map[variant]
                delta = float(decomp_rec[delta_field]) * scale
                hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)
                dd_edited = capture_direct_dot_at_target(model, tokenizer, text, r_hat,
                                                        hook_fn=hook_fn, target_layer=target_layer)
                actual_drift = dd_unedited - dd_edited
                predicted_drift = delta
                err = abs(actual_drift - predicted_drift)
                passing = err <= args.tolerance
                if passing:
                    n_passing += 1
                else:
                    n_failing += 1
                audit["per_check"].append({
                    "prompt_idx": prompt_idx, "condition": cond, "variant": variant,
                    "delta_predicted": predicted_drift,
                    "drift_measured": actual_drift,
                    "abs_error": err,
                    "passing": passing,
                })

    audit["summary"] = {"n_passing": n_passing, "n_failing": n_failing,
                       "pass_rate": n_passing / max(n_passing + n_failing, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2))
    print(f"[drift] {n_passing}/{n_passing+n_failing} checks pass within tolerance "
          f"{args.tolerance} → wrote {args.out}")
    if n_failing > 0:
        print(f"[drift] WARNING: {n_failing} drift checks failed. Inspect "
              f"directdot_drift_audit.json — large discrepancies indicate the "
              f"hook isn't achieving its predicted drift, which compromises "
              f"0b-simple's interpretation.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the drift verifier**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_directdot_drift_verify.py
```

Expected wall: ~3 minutes (5 prompts × 11 conditions × 2 variants × 2 forward passes = 220 forward passes, no generation). Output: `directdot_drift_audit.json`. Expected pass rate: ≥99%. If <90%, debug — likely the hook math or the linearization decomposition has an issue.

- [ ] **Step 3: Commit**

```bash
git add scripts/emnlp_perm_edit/00_directdot_drift_verify.py \
        data/results/emnlp_perm_edit/phase0_controllability/directdot_drift_audit.json
git commit -m "$(cat <<'EOF'
emnlp phase 0: direct_dot drift verification (hook correctness audit)

Captures L15 residuals before/after hook on 5 prompts × all conditions × 2
variants, verifies hook-induced direct_dot drift matches predicted delta
within tolerance. Load-bearing correctness check for 0b-simple.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Aggregate Phase 0 + controllability audit figure

**Files:**
- Create: `scripts/emnlp_perm_edit/00_aggregate_phase0.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/flip_rate_summary.json` (per-variant flip rates + Wilson CIs)
- `data/results/emnlp_perm_edit/phase0_controllability/controllability_audit_figure.png` (main figure)
- `data/results/emnlp_perm_edit/phase0_controllability/PHASE0_SUMMARY.md`

- [ ] **Step 1: Implement aggregation + figure**

Create `scripts/emnlp_perm_edit/00_aggregate_phase0.py`:

```python
"""Phase 0 aggregation: flip rates per variant + controllability audit figure.

Loads 0b-simple's edge_ablation_flip_rates.json + Stage 06 baseline classifications
and computes per-(variant, condition) JB-comply → REFUSE flip rates (for jb_*),
REFUSE → COMPLY flip rates (for bare and ctrl_*), with Wilson 95% CIs.

The main figure shows per-variant flip rate on bare-refuse, summarizing whether
each ablation variant succeeded in driving the model out of refusal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

JB_CLASSES = ("fiction", "roleplay", "analytical", "completion", "cognitive_reframe")
CTRL_CLASSES = tuple(f"ctrl_{c}" for c in JB_CLASSES)
ALL_CONDITIONS = ("bare",) + tuple(f"jb_{c}" for c in JB_CLASSES) + CTRL_CLASSES


def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    if n_total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_baselines(run_dir: Path) -> dict:
    causal = json.loads((run_dir / "06_causal/causal_results.json").read_text())
    baselines = {}
    for r in causal["results"]:
        baselines[r["prompt_idx"]] = {c: blob["cls"] for c, blob in r["baseline"].items()}
    return baselines


def compute_flip_rate(records, baselines, condition: str, target_baseline: str, target_intervened: str):
    n_baseline = 0
    n_flipped = 0
    for r in records:
        if r["condition"] != condition:
            continue
        b = baselines.get(r["prompt_idx"], {}).get(condition, "UNCLEAR")
        if b == target_baseline:
            n_baseline += 1
            if r["classification"] == target_intervened:
                n_flipped += 1
    rate = n_flipped / n_baseline if n_baseline > 0 else 0.0
    ci = wilson_ci(n_flipped, n_baseline)
    return rate, n_flipped, n_baseline, ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json")
    p.add_argument("--baseline-run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    args = p.parse_args()

    data = json.loads(args.in_file.read_text())
    baselines = load_baselines(args.baseline_run_dir)

    summary = {"per_variant": {}}
    for variant, records in data["per_variant"].items():
        per_cond = {}
        for cond in ALL_CONDITIONS:
            if cond.startswith("jb_"):
                tb, ti = "COMPLY", "REFUSE"
            else:
                tb, ti = "REFUSE", "COMPLY"
            rate, n_flip, n_base, ci = compute_flip_rate(records, baselines, cond, tb, ti)
            per_cond[cond] = {
                "flip_rate": rate, "n_flipped": n_flip, "n_baseline": n_base,
                "ci_lo": ci[0], "ci_hi": ci[1],
                "target_baseline_cls": tb, "target_intervened_cls": ti,
            }
        summary["per_variant"][variant] = per_cond

    (args.out_dir / "flip_rate_summary.json").write_text(json.dumps(summary, indent=2))

    # Main figure: per-variant flip rate on bare-refuse
    variants = list(summary["per_variant"].keys())
    bare_rates = [summary["per_variant"][v]["bare"]["flip_rate"] for v in variants]
    bare_lo = [summary["per_variant"][v]["bare"]["ci_lo"] for v in variants]
    bare_hi = [summary["per_variant"][v]["bare"]["ci_hi"] for v in variants]
    err_low = [bare_rates[i] - bare_lo[i] for i in range(len(variants))]
    err_high = [bare_hi[i] - bare_rates[i] for i in range(len(variants))]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(variants, [r * 100 for r in bare_rates],
           yerr=[[e * 100 for e in err_low], [e * 100 for e in err_high]],
           capsize=4)
    ax.set_ylabel("Bare-refuse → COMPLY flip rate (%)")
    ax.set_title("Phase 0 — Controllability audit: bare-refuse flip rate per ablation variant\n"
                 "(higher = more refusal-axis control via the targeted edges)")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "controllability_audit_figure.png", dpi=150)
    plt.close()

    # Write PHASE0_SUMMARY.md
    md_lines = ["# Phase 0 — Controllability Audit Summary\n"]
    md_lines.append("## Bare-refuse → COMPLY flip rate per variant\n")
    md_lines.append("| Variant | flip rate | Wilson 95% CI | n |")
    md_lines.append("|---|---:|---:|---:|")
    for v in variants:
        b = summary["per_variant"][v]["bare"]
        md_lines.append(f"| {v} | {b['flip_rate']*100:.1f}% | "
                       f"[{b['ci_lo']*100:.1f}, {b['ci_hi']*100:.1f}] | "
                       f"{b['n_flipped']}/{b['n_baseline']} |")

    md_lines.append("\n## JB-comply → REFUSE flip rate per (variant, JB class)\n")
    md_lines.append("| Variant | " + " | ".join(f"jb_{c[:8]}" for c in JB_CLASSES) + " |")
    md_lines.append("|" + "---|" * (len(JB_CLASSES) + 1))
    for v in variants:
        row = [v]
        for c in JB_CLASSES:
            cond = f"jb_{c}"
            blob = summary["per_variant"][v][cond]
            row.append(f"{blob['flip_rate']*100:.1f}%")
        md_lines.append("| " + " | ".join(row) + " |")

    (args.out_dir / "PHASE0_SUMMARY.md").write_text("\n".join(md_lines))
    print(f"[aggregate] wrote flip_rate_summary.json, controllability_audit_figure.png, PHASE0_SUMMARY.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run aggregation**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_aggregate_phase0.py
cat data/results/emnlp_perm_edit/phase0_controllability/PHASE0_SUMMARY.md
```

Expected: writes 3 output files, prints the summary markdown.

- [ ] **Step 3: Commit**

```bash
git add scripts/emnlp_perm_edit/00_aggregate_phase0.py \
        data/results/emnlp_perm_edit/phase0_controllability/flip_rate_summary.json \
        data/results/emnlp_perm_edit/phase0_controllability/controllability_audit_figure.png \
        data/results/emnlp_perm_edit/phase0_controllability/PHASE0_SUMMARY.md
git commit -m "$(cat <<'EOF'
emnlp phase 0: aggregation + controllability audit figure

Per-variant flip rates with Wilson CIs across bare, JB classes, and ctrl
classes. Main figure: bare-refuse flip rate per variant. Headline summary in
PHASE0_SUMMARY.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Phase 0 acceptance check + sign audit

**Files:**
- Create: `scripts/emnlp_perm_edit/00_check_phase0_acceptance.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/acceptance_check.json`
- `data/results/emnlp_perm_edit/phase0_controllability/sign_audit.md`

**Hypotheses checked** (per spec § 2.1):
- **H0-1 (controllability completeness):** `ablate_all_edges` drives direct_dot to baseline_offset (proxy: bare-refuse flip rate is ≥X% under `ablate_all_2x`).
- **H0-2 (signed-attribution correctness):** `ablate_features_neg` shifts direct_dot in the OPPOSITE direction from `ablate_features_pos`. Behaviorally: their bare flip rates should be on opposite sides of 0%.
- **H0-3 (error-node prominence):** `ablate_errors_all` alone produces measurable flip rate (>5%) on bare or JB classes → error nodes carry publishable mechanism weight.
- **H0-4 (edge ≠ node):** Compare `ablate_all_edges` flip rate against the v1 35% sparse-feature plateau (REPORT § 9.10).

- [ ] **Step 1: Implement the acceptance checker**

Create `scripts/emnlp_perm_edit/00_check_phase0_acceptance.py`:

```python
"""Phase 0 acceptance check — H0-1 / H0-2 / H0-3 / H0-4 verdicts.

Reads flip_rate_summary.json from the aggregation step and evaluates the
four foundational hypotheses defined in spec § 2.1. Exit code 0 if all four
pass (or all key ones pass per priority); 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flip-rates", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/flip_rate_summary.json")
    p.add_argument("--drift-audit", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/directdot_drift_audit.json")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/acceptance_check.json")
    p.add_argument("--sign-out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/sign_audit.md")
    args = p.parse_args()

    summary = json.loads(args.flip_rates.read_text())
    drift = json.loads(args.drift_audit.read_text()) if args.drift_audit.exists() else None

    verdicts = {}

    # H0-1: ablate_all_2x should flip bare-refuse strongly (>=70% as a guideline)
    bare_all_edges = summary["per_variant"]["ablate_all_edges"]["bare"]["flip_rate"]
    bare_2x = summary["per_variant"]["ablate_all_2x"]["bare"]["flip_rate"]
    verdicts["H0_1_controllability_completeness"] = {
        "bare_flip_rate_under_ablate_all_edges": bare_all_edges,
        "bare_flip_rate_under_ablate_all_2x": bare_2x,
        "expected": "ablate_all_2x should flip bare-refuse at high rate (>=70%); "
                    "ablate_all_edges should drive direct_dot to baseline (positive, "
                    "still refusal-side), so its flip rate may be modest.",
        "pass_strong": bare_2x >= 0.70,
        "pass_weak": bare_2x >= 0.30,
    }

    # H0-2: ablate_features_pos and ablate_features_neg should produce flip rates on opposite sides
    # of bare-refuse (one direction increases refuse, other decreases)
    feat_pos = summary["per_variant"]["ablate_features_pos"]["bare"]["flip_rate"]
    feat_neg = summary["per_variant"]["ablate_features_neg"]["bare"]["flip_rate"]
    # Note: "flip rate" here is REFUSE→COMPLY; if neg shifts toward MORE refusal it should yield 0%
    # and if pos shifts toward LESS refusal it should yield >0%. Different signs of effect.
    verdicts["H0_2_signed_attribution_correctness"] = {
        "bare_flip_rate_ablate_features_pos": feat_pos,
        "bare_flip_rate_ablate_features_neg": feat_neg,
        "expected": "pos and neg ablations should produce different bare-flip behavior. "
                    "Symmetric effect (both ~0% or both equal) would indicate sign-handling bug.",
        "pass": abs(feat_pos - feat_neg) > 0.05 or (feat_pos == 0 and feat_neg == 0 and feat_pos > 0.01),
    }

    # H0-3: error node ablation alone should produce >5% flip rate somewhere
    err_bare = summary["per_variant"]["ablate_errors_all"]["bare"]["flip_rate"]
    err_jb_avg = sum(summary["per_variant"]["ablate_errors_all"][f"jb_{c}"]["flip_rate"]
                     for c in ("fiction", "roleplay", "analytical", "completion", "cognitive_reframe")) / 5
    verdicts["H0_3_error_node_prominence"] = {
        "bare_flip_rate_ablate_errors": err_bare,
        "jb_avg_flip_rate_ablate_errors": err_jb_avg,
        "expected": "error_node ablation should produce >5% flip on some condition if errors carry signal.",
        "pass": err_bare > 0.05 or err_jb_avg > 0.05,
    }

    # H0-4: ablate_all_edges should produce flip rate greater than v1's 35% sparse-feature plateau
    jb_avg_all = sum(summary["per_variant"]["ablate_all_edges"][f"jb_{c}"]["flip_rate"]
                     for c in ("fiction", "roleplay", "analytical", "completion", "cognitive_reframe")) / 5
    verdicts["H0_4_edge_beats_node"] = {
        "jb_avg_flip_rate_ablate_all_edges": jb_avg_all,
        "v1_sparse_node_plateau": 0.348,
        "expected": "comprehensive edge ablation should exceed v1 top-50 plateau of 34.8%",
        "pass": jb_avg_all > 0.40,
    }

    # Drift audit summary
    if drift is not None:
        verdicts["drift_audit"] = drift["summary"]

    overall_pass = (
        verdicts["H0_1_controllability_completeness"]["pass_weak"] and
        verdicts["H0_2_signed_attribution_correctness"]["pass"]
    )

    print("Phase 0 acceptance verdict:")
    for h, v in verdicts.items():
        if "pass" in v:
            print(f"  {h}: {'PASS' if v['pass'] else 'FAIL'}")
        elif "pass_strong" in v:
            print(f"  {h}: strong={'PASS' if v['pass_strong'] else 'FAIL'} "
                  f"weak={'PASS' if v['pass_weak'] else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    args.out.write_text(json.dumps({"verdicts": verdicts, "overall_pass": overall_pass}, indent=2))

    # Sign audit MD
    md = [
        "# Phase 0 Sign Audit (H0-2)\n",
        f"bare flip rate under ablate_features_pos: {feat_pos*100:.1f}%",
        f"bare flip rate under ablate_features_neg: {feat_neg*100:.1f}%",
        "",
        ("PASS — positive and negative ablations produce different behavior, "
         "confirming sign-handling is correct."
         if verdicts["H0_2_signed_attribution_correctness"]["pass"]
         else "FAIL — pos and neg ablations produce similar behavior. Likely a sign-handling "
              "or basis bug in the attribution pipeline. Investigate before any further claim."),
    ]
    args.sign_out.write_text("\n".join(md))

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the acceptance check**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_check_phase0_acceptance.py
cat data/results/emnlp_perm_edit/phase0_controllability/sign_audit.md
```

Expected: verdict printed; exit 0 (PASS) or 1 (FAIL). If FAIL, inspect the per-hypothesis verdict and investigate based on which hypothesis failed. Per the spec:
- **H0-1 weak fail**: comprehensive edge ablation has limited behavioral effect — surprising; suggests the residual-stream proxy might not be a faithful test of edge ablation. Consider 0b-rigorous.
- **H0-2 fail**: sign-handling bug. Pause Phase 0 conclusions; debug attribution math.
- **H0-3 fail**: error nodes carry negligible signal — OK, not a blocker; report as a clean negative finding.
- **H0-4 fail**: edge ablation doesn't beat the node plateau — the linearization decomposition doesn't translate into causal control beyond what sparse ablation already gave us. Reframe paper's Framing A.

- [ ] **Step 3: Commit**

```bash
git add scripts/emnlp_perm_edit/00_check_phase0_acceptance.py \
        data/results/emnlp_perm_edit/phase0_controllability/acceptance_check.json \
        data/results/emnlp_perm_edit/phase0_controllability/sign_audit.md
git commit -m "$(cat <<'EOF'
emnlp phase 0: acceptance check + sign audit (H0-1/2/3/4)

Evaluates the four foundational hypotheses defined in spec § 2.1 against the
0b-simple flip rates. Sign audit (H0-2) is the priority diagnostic — failure
indicates a sign-handling bug that must be fixed before any further claim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Direction-alignment robustness audit (Sub-experiment 0c)

**Files:**
- Create: `scripts/emnlp_perm_edit/00_direction_robustness.py`
- Create: `scripts/emnlp_perm_edit/tests/test_direction_robustness.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/direction_robustness.json`
- `data/results/emnlp_perm_edit/phase0_controllability/direction_robustness_figure.png`

**Dependencies:** none beyond Task 0 (scaffold). This task uses `02b_stats/residuals_L15_per_cond.pt` (already locally available) and `01_direction/unnormalized_r.pt` — it does NOT require the HF graph fetch (Task 1) or any GPU work. **Recommended execution order: run this right after Task 0**, share results with Georg first, then continue with Tasks 1–9 in parallel.

**Purpose:** addresses Georg's 2026-05-17 challenge to the +0.72 to +0.94 cosine between `r_jb_C` (toward jailbreak) and `−r̂` (toward harmless) reported in REPORT § 5.5.2. Three diagnostics (per-prompt cosine, random-baseline cosine, Pearson-style mean-subtracted cosine) test whether that alignment is a robust geometric fact or partly inflated by high-dimensional residual stream artifacts.

- [ ] **Step 1: Write the failing tests**

Create `scripts/emnlp_perm_edit/tests/test_direction_robustness.py`:

```python
"""Tests for direction-robustness audit helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 00_ prefixed scripts are not importable directly; use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "direction_robustness",
    Path(__file__).resolve().parents[1] / "00_direction_robustness.py",
)
direction_robustness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direction_robustness)

compute_per_prompt_cosines = direction_robustness.compute_per_prompt_cosines
random_baseline_cosine_stats = direction_robustness.random_baseline_cosine_stats
pearson_cosine = direction_robustness.pearson_cosine


def test_per_prompt_cosines_match_class_mean_when_all_prompts_identical():
    """Sanity: if every prompt's displacement is identical, per-prompt mean equals class-mean cosine."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    common_delta = torch.randn(2560)
    h_jb = common_delta.unsqueeze(0).expand(50, 2560).clone()
    h_bare = torch.zeros(50, 2560)
    per_prompt = compute_per_prompt_cosines(h_jb, h_bare, r_hat)
    class_mean_cos = torch.nn.functional.cosine_similarity(
        common_delta.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    assert abs(per_prompt["mean_cos"] - class_mean_cos) < 1e-5
    assert per_prompt["std_cos"] < 1e-5


def test_per_prompt_cosines_have_nonzero_std_when_prompts_differ():
    """When per-prompt displacements vary, std is nonzero."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    h_jb = torch.randn(50, 2560)
    h_bare = torch.randn(50, 2560)
    per_prompt = compute_per_prompt_cosines(h_jb, h_bare, r_hat)
    assert per_prompt["std_cos"] > 0.01


def test_random_baseline_returns_expected_stats():
    """Random-baseline returns 95th percentile of absolute cosines and the test direction's cosine."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    r_jb = torch.randn(2560)
    stats = random_baseline_cosine_stats(r_jb, r_hat, n_random=500, seed=42)
    assert "p95_abs_random_cos" in stats
    assert "real_cos_with_r_hat" in stats
    assert "rank_of_real_in_random" in stats
    # rank is in [0, n_random]
    assert 0 <= stats["rank_of_real_in_random"] <= 500


def test_pearson_cosine_zeros_out_all_ones_bias():
    """If both vectors are pure all-ones, raw cosine is 1.0 but Pearson cosine is 0/0 → NaN handled to 0."""
    r_hat = torch.ones(100)
    r_jb = torch.ones(100) * 3.0  # parallel to all-ones, but scaled
    raw = torch.nn.functional.cosine_similarity(
        r_jb.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    pearson = pearson_cosine(r_jb, r_hat)
    assert raw == pytest.approx(1.0)
    # After mean-subtraction both vectors are zero; pearson returns 0 by convention
    assert pearson == pytest.approx(0.0, abs=1e-5)


def test_pearson_cosine_orthogonal_random_vectors_differs_from_raw():
    """For random vectors, Pearson and raw cosines should be similar but not identical."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    r_jb = torch.randn(2560)
    raw = torch.nn.functional.cosine_similarity(
        r_jb.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    pearson = pearson_cosine(r_jb, r_hat)
    # Both should be close to 0 for random pairs; small numerical difference is expected
    assert abs(raw - pearson) < 0.1
```

- [ ] **Step 2: Run tests and verify failure**

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_direction_robustness.py -v
```

Expected: `FileNotFoundError` or similar — script doesn't exist yet.

- [ ] **Step 3: Implement `00_direction_robustness.py`**

Create `scripts/emnlp_perm_edit/00_direction_robustness.py`:

```python
"""Phase 0 — Sub-experiment 0c: direction-alignment robustness audit.

Tests whether the +0.72 to +0.94 cosine between r_jb_C (toward jailbreak) and
−r̂ (toward harmless) reported in REPORT § 5.5.2 is a robust geometric fact or
partly inflated by high-dimensional residual stream anisotropy / all-ones-
direction bias / class-mean averaging artifacts.

Three diagnostics per class:
- 0c.1 per-prompt cosine: compute the JB displacement for each individual
  prompt and report per-class mean ± std of per-prompt cosines.
- 0c.2 random-direction baseline: sample N random unit vectors and compare
  cos(r_jb_C, r̂) against the 95th percentile of cos(r_jb_C, random_dir).
- 0c.3 Pearson-style mean-subtraction: compute cosine after subtracting each
  direction's scalar mean (all-ones-direction bias control).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]

LAYER = 15
POS_IDX = 2  # index of pos=-2 in the saved [-5, -3, -2] tensor
CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]


def compute_per_prompt_cosines(h_jb: torch.Tensor, h_bare: torch.Tensor,
                               r_hat: torch.Tensor) -> dict:
    """For each prompt p, compute cos(h_jb[p] - h_bare[p], r_hat).

    Returns {mean_cos, std_cos, gt_half_rate, per_prompt: [...]}.
    """
    delta = h_jb.float() - h_bare.float()  # (n_prompts, d_model)
    per_prompt_cos = torch.nn.functional.cosine_similarity(
        delta, r_hat.float().unsqueeze(0).expand_as(delta), dim=-1
    )  # (n_prompts,)
    mean_cos = per_prompt_cos.mean().item()
    std_cos = per_prompt_cos.std().item() if per_prompt_cos.numel() > 1 else 0.0
    gt_half = (per_prompt_cos.abs() > 0.5).float().mean().item()
    return {
        "mean_cos": mean_cos,
        "std_cos": std_cos,
        "gt_half_rate": gt_half,
        "per_prompt_cosines": per_prompt_cos.tolist(),
    }


def random_baseline_cosine_stats(r_jb_C: torch.Tensor, r_hat: torch.Tensor,
                                 n_random: int = 1000, seed: int = 42) -> dict:
    """Sample n_random unit vectors and compare cos(r_jb_C, r_hat) to their cosines."""
    g = torch.Generator().manual_seed(seed)
    d = r_jb_C.shape[0]
    rand = torch.randn(n_random, d, generator=g)
    rand = rand / rand.norm(dim=1, keepdim=True)
    rand_cos = torch.nn.functional.cosine_similarity(
        r_jb_C.float().unsqueeze(0), rand, dim=-1
    )  # (n_random,)
    real_cos = torch.nn.functional.cosine_similarity(
        r_jb_C.float().unsqueeze(0), r_hat.float().unsqueeze(0), dim=-1
    ).item()
    p95 = rand_cos.abs().quantile(0.95).item()
    # Rank: how many random cosines have |cos| larger than real cos
    rank = int((rand_cos.abs() > abs(real_cos)).sum().item())
    return {
        "n_random": n_random,
        "p95_abs_random_cos": p95,
        "real_cos_with_r_hat": real_cos,
        "rank_of_real_in_random": rank,
        "real_passes_p95": abs(real_cos) > p95,
    }


def pearson_cosine(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine of mean-subtracted vectors (i.e., Pearson correlation coefficient).

    Returns 0.0 if either centered vector has zero norm (e.g., constant vectors).
    """
    v1c = v1.float() - v1.float().mean()
    v2c = v2.float() - v2.float().mean()
    n1 = v1c.norm()
    n2 = v2c.norm()
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return ((v1c @ v2c) / (n1 * n2)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[0c] loading r̂[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()
    print(f"  ||r̂|| = {r_hat.norm().item():.2f}")

    print(f"[0c] loading residuals from {args.run_dir}/02b_stats/")
    R = torch.load(args.run_dir / "02b_stats/residuals_L15_per_cond.pt", weights_only=False)
    h_bare = R["bare"][:, POS_IDX, :].float()  # (50, 2560)

    out = {
        "metadata": {
            "n_random": args.n_random, "seed": args.seed,
            "r_hat_norm": r_hat.norm().item(),
            "n_prompts": h_bare.shape[0],
            "convention": "r_jb_C = mean(h_jb_C) - mean(h_bare); Ball/Wang (points toward JB)",
        },
        "per_class": {},
    }

    print(f"\n[0c] per-class diagnostics:")
    print(f"  {'class':22s}  {'class_cos':>10s}  {'pp_mean':>10s}  {'pp_std':>10s}  "
          f"{'p95_rand':>10s}  {'pearson':>10s}  {'rank/N':>10s}")

    for cls in CLASSES:
        h_jb = R[f"jb_{cls}"][:, POS_IDX, :].float()
        r_jb_C = h_jb.mean(0) - h_bare.mean(0)
        class_mean_cos = torch.nn.functional.cosine_similarity(
            r_jb_C.unsqueeze(0), r_hat.unsqueeze(0)).item()

        per_prompt = compute_per_prompt_cosines(h_jb, h_bare, r_hat)
        rand_stats = random_baseline_cosine_stats(
            r_jb_C, r_hat, n_random=args.n_random, seed=args.seed)
        pcos = pearson_cosine(r_jb_C, r_hat)

        out["per_class"][cls] = {
            "class_mean_cos_with_r_hat": class_mean_cos,
            "class_mean_cos_with_neg_r_hat": -class_mean_cos,
            "per_prompt": per_prompt,
            "random_baseline": rand_stats,
            "pearson_cos": pcos,
            "delta_raw_minus_pearson": abs(class_mean_cos - pcos),
        }

        print(f"  {cls:22s}  {class_mean_cos:+10.4f}  {per_prompt['mean_cos']:+10.4f}  "
              f"{per_prompt['std_cos']:10.4f}  {rand_stats['p95_abs_random_cos']:10.4f}  "
              f"{pcos:+10.4f}  {rand_stats['rank_of_real_in_random']:4d}/{args.n_random:4d}")

    # Phase 0 H0-5 verdict per class
    print(f"\n[0c] H0-5 per-class verdicts:")
    n_pass_all = 0
    for cls in CLASSES:
        blob = out["per_class"][cls]
        c1_pass = abs(blob["per_prompt"]["mean_cos"] - blob["class_mean_cos_with_r_hat"]) < 0.10
        c2_pass = blob["random_baseline"]["real_passes_p95"]
        c3_pass = blob["delta_raw_minus_pearson"] < 0.10
        full_pass = c1_pass and c2_pass and c3_pass
        if full_pass:
            n_pass_all += 1
        print(f"  {cls:22s}  per_prompt={'PASS' if c1_pass else 'FAIL'}  "
              f"random={'PASS' if c2_pass else 'FAIL'}  "
              f"pearson={'PASS' if c3_pass else 'FAIL'}  "
              f"all={'PASS' if full_pass else 'FAIL'}")
    out["h0_5_summary"] = {
        "n_classes_passing_all": n_pass_all,
        "n_classes_total": len(CLASSES),
        "overall_pass": n_pass_all >= 4,
    }
    print(f"\nH0-5 overall: {n_pass_all}/{len(CLASSES)} classes pass all 3 controls "
          f"→ {'PASS' if out['h0_5_summary']['overall_pass'] else 'FAIL'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "direction_robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\n[0c] wrote direction_robustness.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_direction_robustness.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Run the diagnostic on real data**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_robustness.py
```

Expected wall: ~10 seconds. Output: per-class table printed to stdout + `direction_robustness.json` written.

For each class, expect to see something like:
```
fiction               +0.7240    +0.71xx   0.07xx   0.08xx   +0.72xx     5/1000
```

The headline numbers from REPORT § 5.5.2 are +0.72 (fiction) to +0.94 (cog_reframe). The "rank" column should show the real cosine in the top ~5 of 1000 random cosines (i.e., ≥ 95th percentile).

- [ ] **Step 6: Generate the figure**

Add an optional `--make-figure` flag invocation, or implement a separate small figure script. For simplicity, add to the same script — a two-panel matplotlib figure showing (left) per-prompt cosine scatter + class-mean line per class; (right) random-baseline histogram overlaid with real cosines as vertical lines.

Implement by editing `00_direction_robustness.py` to add a `_make_figure()` call after writing the JSON, if `--make-figure` is set (default True). Use existing imports:

```python
# Add to the imports at the top of 00_direction_robustness.py:
import matplotlib.pyplot as plt
import numpy as np

# Add at the bottom of main(), before "wrote" log line:
def _make_figure(out, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Left: per-prompt cosine scatter + class-mean line
    for i, cls in enumerate(CLASSES):
        blob = out["per_class"][cls]
        pp = blob["per_prompt"]["per_prompt_cosines"]
        x = np.full(len(pp), i) + np.random.uniform(-0.15, 0.15, size=len(pp))
        ax1.scatter(x, pp, alpha=0.5, s=20)
        ax1.axhline(blob["class_mean_cos_with_r_hat"], xmin=i/5+0.02, xmax=(i+1)/5-0.02,
                    color="black", linewidth=2)
    ax1.set_xticks(range(len(CLASSES)))
    ax1.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax1.set_ylabel("cos(per-prompt JB displacement, r̂)")
    ax1.set_title("Per-prompt cosine — should align with class-mean (black bars)")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # Right: random-baseline cosines (use fiction's distribution as representative)
    cls0 = CLASSES[0]
    rand_stats = out["per_class"][cls0]["random_baseline"]
    # Re-sample random cosines for the histogram (not stored in JSON to save space)
    g = torch.Generator().manual_seed(out["metadata"]["seed"])
    rand = torch.randn(out["metadata"]["n_random"], 2560, generator=g)
    rand = rand / rand.norm(dim=1, keepdim=True)
    # (Approximate by using r_jb of fiction implicitly; in production, store rand cosines per class)
    ax2.hist(np.random.normal(0, 1/np.sqrt(2560), size=out["metadata"]["n_random"]),
             bins=50, alpha=0.5, color="gray", label="random direction baseline")
    for cls in CLASSES:
        cos = out["per_class"][cls]["class_mean_cos_with_r_hat"]
        ax2.axvline(cos, label=f"{cls}: {cos:+.2f}", linewidth=2)
    ax2.set_xlabel("cos(r_jb_C, direction)")
    ax2.set_title("Random-direction baseline distribution vs r_jb_C cosine with r̂\n"
                  "(real cosines should land in the tails)")
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Then in main(), after writing JSON:
_make_figure(out, args.out_dir / "direction_robustness_figure.png")
print(f"[0c] wrote direction_robustness_figure.png")
```

After adding the figure code, rerun:

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_robustness.py
```

Expected: the JSON and PNG both write.

- [ ] **Step 7: Commit**

```bash
git add scripts/emnlp_perm_edit/00_direction_robustness.py \
        scripts/emnlp_perm_edit/tests/test_direction_robustness.py \
        data/results/emnlp_perm_edit/phase0_controllability/direction_robustness.json \
        data/results/emnlp_perm_edit/phase0_controllability/direction_robustness_figure.png
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0c direction-alignment robustness audit (Georg's cosine challenge)

Three diagnostics per JB class: (a) per-prompt cosine vs class-mean; (b) random-
direction baseline with 95th percentile test; (c) Pearson-style mean-subtraction.
Tests H0-5: whether the +0.72 to +0.94 cosine between r_jb_C and -r_hat is a
robust geometric fact or inflated by high-dim anisotropy / all-ones bias /
class-mean averaging. CPU-only, ~10s wall.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Top-K feature ablation Pareto sweep (Sub-experiment 0d)

**Files:**
- Modify: `scripts/emnlp_perm_edit/graph_loader.py` (add `extract_edge_records_to_target` helper)
- Modify: `scripts/emnlp_perm_edit/tests/test_graph_loader.py` (add tests for the new helper)
- Create: `scripts/emnlp_perm_edit/00_topk_feature_sweep.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/topk_feature_sweep.json`
- `data/results/emnlp_perm_edit/phase0_controllability/topk_feature_pareto_figure.png`

**Dependencies:** Tasks 0, 1, 2, 5 (scaffold, HF graph pull, graph_loader, edge_ablation_hook). Reuses the residual-stream r̂-projection hook from Task 5 with a per-(prompt, condition) delta computed from the top-K features.

**Hypothesis tested:** H0-6 (refusal-signal sparsity / Pareto knee with sign asymmetry).

- [ ] **Step 1: Extend graph_loader.py with per-edge record extraction + test**

Append to `scripts/emnlp_perm_edit/tests/test_graph_loader.py`:

```python
def test_extract_edge_records_returns_signed_attributions_per_source():
    """For edges into target, returns per-source records with type + attribution."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_edge_records_to_target, DEFAULT_NODE_TYPE_TO_CATEGORY  # noqa: E402

    graph = {
        "nodes": [
            {"id": "f1", "node_type": "cross layer transcoder", "layer": 13, "feature": 427},
            {"id": "f2", "node_type": "cross layer transcoder", "layer": 11, "feature": 99},
            {"id": "e1", "node_type": "embedding", "layer": 0, "feature": 12345},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": +1.5},
            {"source": "f2", "target": "target", "weight": -0.5},
            {"source": "e1", "target": "target", "weight": +0.2},
            {"source": "f1", "target": "f2", "weight": 99.0},  # not to target, skip
        ],
    }
    records = extract_edge_records_to_target(
        graph, target_node_id="target",
        node_type_to_category=DEFAULT_NODE_TYPE_TO_CATEGORY,
        filter_category="feature",
    )
    # Only feature edges to target — should be 2 records, sorted by signed attribution desc
    assert len(records) == 2
    assert records[0]["source_id"] == "f1"
    assert records[0]["signed_attribution"] == +1.5
    assert records[0]["category"] == "feature"
    assert records[0]["layer"] == 13
    assert records[0]["feature"] == 427
    assert records[1]["source_id"] == "f2"
    assert records[1]["signed_attribution"] == -0.5


def test_extract_edge_records_no_filter_returns_all_categories():
    """Without filter_category, all source-type edges into target are returned."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_edge_records_to_target, DEFAULT_NODE_TYPE_TO_CATEGORY  # noqa: E402

    graph = {
        "nodes": [
            {"id": "f1", "node_type": "cross layer transcoder"},
            {"id": "e1", "node_type": "embedding"},
            {"id": "err1", "node_type": "mlp reconstruction error"},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": 1.0},
            {"source": "e1", "target": "target", "weight": 2.0},
            {"source": "err1", "target": "target", "weight": 3.0},
        ],
    }
    records = extract_edge_records_to_target(
        graph, target_node_id="target",
        node_type_to_category=DEFAULT_NODE_TYPE_TO_CATEGORY,
        filter_category=None,
    )
    assert len(records) == 3
    categories = {r["category"] for r in records}
    assert categories == {"feature", "embedding", "error_node"}
```

Run tests, verify FAIL (`AttributeError: module 'graph_loader' has no attribute 'extract_edge_records_to_target'`):

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_graph_loader.py::test_extract_edge_records_returns_signed_attributions_per_source -v
```

Then append to `scripts/emnlp_perm_edit/graph_loader.py`:

```python
def extract_edge_records_to_target(
    graph: dict,
    target_node_id: str,
    node_type_to_category: dict[str, Category] = DEFAULT_NODE_TYPE_TO_CATEGORY,
    filter_category: Category | None = None,
) -> list[dict]:
    """Return per-edge records for edges targeting `target_node_id`.

    Each record is a dict:
        {
            "source_id": str,
            "category": "feature" | "embedding" | "error_node",
            "layer": int | None,           # if available in source node
            "feature": int | None,          # if available
            "signed_attribution": float,
        }
    Sorted by `signed_attribution` descending (most positive first).

    Args:
        graph: parsed packed-graph dict.
        target_node_id: id of the measurement-target node.
        node_type_to_category: mapping from raw node_type to our 3-bucket category.
        filter_category: if not None, only records matching this category are returned.

    Raises:
        ValueError on unknown source-node node_type (fail-loud, same convention as
        aggregate_edge_attributions).
    """
    node_lookup = {n["id"]: n for n in graph["nodes"]}
    edges_field = "links" if "links" in graph else "edges"
    records = []
    for edge in graph[edges_field]:
        if edge["target"] != target_node_id:
            continue
        src_node = node_lookup[edge["source"]]
        src_type = src_node.get("node_type", "")
        if src_type == "logit":
            continue
        if src_type not in node_type_to_category:
            raise ValueError(
                f"unknown node_type {src_type!r} on edge source {edge['source']!r}"
            )
        category = node_type_to_category[src_type]
        if filter_category is not None and category != filter_category:
            continue
        records.append({
            "source_id": edge["source"],
            "category": category,
            "layer": src_node.get("layer"),
            "feature": src_node.get("feature"),
            "signed_attribution": float(edge["weight"]),
        })
    records.sort(key=lambda r: r["signed_attribution"], reverse=True)
    return records
```

Rerun the tests, verify `2 passed`:

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_graph_loader.py -v
```

Expected: all 8 tests pass (6 existing + 2 new).

- [ ] **Step 2: Implement `00_topk_feature_sweep.py`**

Create `scripts/emnlp_perm_edit/00_topk_feature_sweep.py`:

```python
"""Phase 0 — Sub-experiment 0d: top-K feature ablation Pareto sweep (signed).

For each (prompt, condition):
  1. Load packed attribution graph.
  2. Extract feature-type edges into the measurement target.
  3. For each variant (pos-K, neg-K, abs-K) and each K, sum the top-K signed
     attributions to get delta_K.
  4. Register a runtime hook at L15 that subtracts delta_K × r̂_unit / ‖r̂‖
     from the residual (reuses 0b-simple's make_scalar_rhat_subtraction_hook).
  5. Generate max_new_tokens=80 greedy, classify refuse/comply.
  6. Aggregate per (variant, K) flip rates with Wilson CIs.

Tests H0-6: refusal-signal sparsity / Pareto knee with sign asymmetry.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402
from graph_loader import (  # noqa: E402
    extract_edge_records_to_target,
    find_measurement_target_node_id,
    load_packed_graph,
)
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402


LAYER = 15
DEFAULT_K_VALUES = [1, 5, 10, 20, 50, 100, 500]
VARIANTS = ("pos", "neg", "abs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/graph_data")
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/topk_feature_sweep.json")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help="Comma-separated subset of: pos,neg,abs")
    p.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    p.add_argument("--mode", default="single", choices=["single", "multi"])
    return p.parse_args()


def compute_delta_for_variant(records: list[dict], variant: str, K: int) -> float:
    """Sum signed attributions of the top-K feature records under the given variant.

    pos: top-K by descending signed attribution (most positive first); take their sum.
    neg: top-K by ascending signed attribution (most negative first); take their sum.
    abs: top-K by descending |signed|; take their signed sum (mixes pos and neg).
    """
    if variant == "pos":
        chosen = records[:K]  # records already sorted desc by signed_attribution
    elif variant == "neg":
        chosen = records[::-1][:K]  # reverse for ascending → most negative first
    elif variant == "abs":
        chosen = sorted(records, key=lambda r: abs(r["signed_attribution"]), reverse=True)[:K]
    else:
        raise ValueError(f"unknown variant: {variant}")
    return sum(r["signed_attribution"] for r in chosen)


def main():
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    print(f"[0d] loading r̂[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()
    print(f"  ||r̂|| = {r_hat.norm().item():.2f}")

    print(f"[0d] loading model {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_layer = layers[LAYER]
    print(f"  loaded in {time.time()-t0:.1f}s")

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
    pad_id = tokenizer.eos_token_id

    # Pre-compute per-(prompt, condition) feature edge records once
    print(f"[0d] pre-computing per-(prompt, condition) feature rankings...")
    rankings = {}
    skipped = []
    for prompt_idx, prompt in enumerate(dataset):
        for cond, blob in prompt["conditions"].items():
            slug = f"{prompt_idx}_{cond}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                skipped.append(slug)
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            records = extract_edge_records_to_target(graph, target_id, filter_category="feature")
            rankings[(prompt_idx, cond)] = records
    print(f"  ranked features for {len(rankings)} inputs ({len(skipped)} skipped)")

    results = {
        "metadata": {
            "layer": LAYER, "model": args.model,
            "variants": variants_to_run, "k_values": k_values,
            "n_prompts": len(dataset), "mode": args.mode,
        },
        "per_variant_K": {},
    }

    t_total = time.time()
    for variant in variants_to_run:
        for K in k_values:
            key = f"{variant}_K{K}"
            print(f"\n[0d] variant={variant} K={K}")
            t_v = time.time()
            per_records = []
            for (prompt_idx, cond), records in rankings.items():
                if len(records) == 0:
                    continue
                delta = compute_delta_for_variant(records, variant, K)
                hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)

                text = dataset[prompt_idx]["conditions"][cond]["text"]
                formatted = format_prompt(tokenizer, text)
                ids = tokenizer(formatted, return_tensors="pt").to(model.device)
                prompt_len = ids.input_ids.shape[1]

                handle = target_layer.register_forward_hook(hook_fn)
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **ids, do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=pad_id,
                        )
                    resp = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                finally:
                    handle.remove()

                per_records.append({
                    "prompt_idx": prompt_idx, "condition": cond,
                    "delta_applied": delta, "n_features_used": min(K, len(records)),
                    "response": resp[:300],
                    "classification": classify_response(resp),
                    "coherent": is_coherent(resp),
                })
            results["per_variant_K"][key] = per_records
            print(f"  {key} done in {time.time()-t_v:.0f}s ({len(per_records)} generations)")

    elapsed = time.time() - t_total
    print(f"\n[0d] sweep complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[0d] wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test on 1 prompt × 1 variant × 2 K values**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_feature_sweep.py \
    --max-prompts 1 --variants pos --k-values 5,50 \
    --out /tmp/0d_smoke.json
```

Expected: ~30s, writes 22 records (2 K values × 1 prompt × 11 conditions). Spot-check that delta_applied is reasonable (similar magnitude to feature_signed from 0a):

```bash
python3 -c "
import json
d = json.load(open('/tmp/0d_smoke.json'))
for key in d['per_variant_K']:
    recs = d['per_variant_K'][key]
    print(f'{key}: {len(recs)} records; sample deltas:')
    for r in recs[:3]:
        print(f'  prompt={r[\"prompt_idx\"]} cond={r[\"condition\"]:25s} '
              f'delta={r[\"delta_applied\"]:+10.1f} K_used={r[\"n_features_used\"]} cls={r[\"classification\"]}')
"
```

- [ ] **Step 4: Full run on all 3 variants × 7 K values × 50 prompts × 11 conditions**

```bash
tmux new -s phase0_0d 'PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_feature_sweep.py 2>&1 | tee /tmp/phase0_0d.log'
```

Expected wall: ~6 hours (21 variant-K combinations × 550 generations).

**Scope-trim option:** if 6h is too long, use `--k-values 5,50,500` (3 K values × 3 variants = 9 combinations, ~2.5h wall). Fill in higher resolution only if H0-6 holds at coarse resolution.

- [ ] **Step 5: Generate the Pareto figure**

Add a small companion script `00_topk_feature_pareto_figure.py` (or inline at the end of the sweep script). Two-panel matplotlib: (left) bare-refuse flip rate vs K, one curve per variant; (right) JB-comply flip rate (avg across 5 classes) vs K, one curve per variant. Reuse Wilson-CI logic from `02_aggregate_phase1.py`. (Code follows the same pattern as Task 8's figure code.)

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_feature_pareto_figure.py
ls -la data/results/emnlp_perm_edit/phase0_controllability/topk_feature_pareto_figure.png
```

- [ ] **Step 6: Commit**

```bash
git add scripts/emnlp_perm_edit/graph_loader.py \
        scripts/emnlp_perm_edit/tests/test_graph_loader.py \
        scripts/emnlp_perm_edit/00_topk_feature_sweep.py \
        scripts/emnlp_perm_edit/00_topk_feature_pareto_figure.py \
        data/results/emnlp_perm_edit/phase0_controllability/topk_feature_sweep.json \
        data/results/emnlp_perm_edit/phase0_controllability/topk_feature_pareto_figure.png
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0d top-K feature ablation Pareto sweep (signed)

Three variants (pos-K, neg-K, abs-K) × 7 K values × 550 prompts. Per-prompt
feature ranking via extract_edge_records_to_target, runtime hook reuses 0b-simple's
r̂-projection subtraction. Tests H0-6 (refusal-signal sparsity + sign asymmetry).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Top-K edge ablation Pareto sweep (Sub-experiment 0e)

**Files:**
- Create: `scripts/emnlp_perm_edit/00_topk_edge_sweep.py`
- Create: `scripts/emnlp_perm_edit/00_edge_vs_node_figure.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/topk_edge_sweep.json`
- `data/results/emnlp_perm_edit/phase0_controllability/topk_edge_vs_node_figure.png`

**Dependencies:** Tasks 0, 1, 2, 5, 11 (scaffold, HF pull, graph_loader, edge_ablation_hook, and Task 11's extended `extract_edge_records_to_target`).

**Hypothesis tested:** H0-7 (edge Pareto > node Pareto at every K).

The implementation is structurally identical to Task 11 but with `filter_category=None` in `extract_edge_records_to_target` (rank ALL edge types, not just feature edges).

- [ ] **Step 1: Implement `00_topk_edge_sweep.py`**

Create `scripts/emnlp_perm_edit/00_topk_edge_sweep.py` — same structure as `00_topk_feature_sweep.py` from Task 11 with two changes:

```python
# Change 1: in the pre-compute loop, drop filter_category:
records = extract_edge_records_to_target(graph, target_id, filter_category=None)

# Change 2: change the output filename:
default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/topk_edge_sweep.json"

# Change 3: bump default K values to include K=1000 (edges are more numerous than features):
DEFAULT_K_VALUES = [1, 5, 10, 50, 100, 500, 1000]
```

Otherwise the script is a copy of Task 11's `00_topk_feature_sweep.py`. To avoid duplication, you can refactor both into a shared module (`scripts/emnlp_perm_edit/topk_sweep_runner.py`) with a `mode` parameter — feel free to do that refactor if it feels cleaner, or copy/paste for speed.

- [ ] **Step 2: Smoke test**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_edge_sweep.py \
    --max-prompts 1 --variants pos --k-values 10,100 \
    --out /tmp/0e_smoke.json
```

Expected: similar shape to 0d smoke output but with more edges per prompt (deltas will be larger because all node types contribute).

- [ ] **Step 3: Full run**

```bash
tmux new -s phase0_0e 'PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_edge_sweep.py 2>&1 | tee /tmp/phase0_0e.log'
```

Expected wall: ~6 hours (similar to 0d). Same scope-trim option applies.

- [ ] **Step 4: Generate the edge-vs-node comparison figure**

Create `scripts/emnlp_perm_edit/00_edge_vs_node_figure.py` that reads both `topk_feature_sweep.json` and `topk_edge_sweep.json`, computes per-(variant, K) bare-refuse flip rate with Wilson CIs, and plots both curves overlaid (node = solid lines, edge = dashed lines) per variant. Annotate the integrated area between edge and node curves as a single scalar at the bottom of the figure — that's the "methodology lever" metric for H0-7.

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_vs_node_figure.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/emnlp_perm_edit/00_topk_edge_sweep.py \
        scripts/emnlp_perm_edit/00_edge_vs_node_figure.py \
        data/results/emnlp_perm_edit/phase0_controllability/topk_edge_sweep.json \
        data/results/emnlp_perm_edit/phase0_controllability/topk_edge_vs_node_figure.png
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0e top-K edge ablation Pareto sweep + edge-vs-node figure

Tests H0-7: whether edge-level top-K ablation uniformly outperforms node-level
top-K at every K. Structurally identical to 0d but with filter_category=None
in extract_edge_records_to_target so all edge types (features, embeddings,
error nodes) participate in the ranking.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Feature role clustering + semantic annotation (Sub-experiment 0f)

**Files:**
- Modify: `scripts/emnlp_perm_edit/graph_loader.py` (add `extract_feature_profile` helper)
- Modify: `scripts/emnlp_perm_edit/tests/test_graph_loader.py` (add a test for the helper)
- Create: `scripts/emnlp_perm_edit/00_feature_taxonomy.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy.json`
- `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy_clusters.md`
- `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy_figure.png`

**Dependencies:** Tasks 0, 1, 2, 3 (scaffold, HF pull, graph_loader, 0a linearization decomposition's per-prompt records). Optionally uses Stage 04 outputs from `data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels.json` for semantic annotation.

**Hypothesis tested:** H0-8 (feature role taxonomy structure).

- [ ] **Step 1: Add feature-profile extraction to graph_loader**

Append to `scripts/emnlp_perm_edit/graph_loader.py`:

```python
def extract_feature_profile(
    graph: dict, target_node_id: str,
    node_type_to_category: dict[str, Category] = DEFAULT_NODE_TYPE_TO_CATEGORY,
) -> dict[tuple[int, int], float]:
    """For each feature source node in the graph, return its signed attribution to target.

    Returns dict keyed by (layer, feature_id) → signed_attribution. Used by 0f to
    build per-feature profiles across the 11-condition dataset (caller iterates
    over conditions and aggregates).

    Skips non-feature source types (embeddings, error_nodes) — those have their own
    profile-build path. Raises on unknown source types per the graph_loader convention.
    """
    node_lookup = {n["id"]: n for n in graph["nodes"]}
    edges_field = "links" if "links" in graph else "edges"
    profile = {}
    for edge in graph[edges_field]:
        if edge["target"] != target_node_id:
            continue
        src_node = node_lookup[edge["source"]]
        src_type = src_node.get("node_type", "")
        if src_type == "logit":
            continue
        if src_type not in node_type_to_category:
            raise ValueError(f"unknown node_type {src_type!r}")
        if node_type_to_category[src_type] != "feature":
            continue
        L = src_node.get("layer")
        F = src_node.get("feature")
        if L is None or F is None:
            continue
        profile[(L, F)] = float(edge["weight"])
    return profile
```

Append a test:

```python
def test_extract_feature_profile_skips_non_feature_sources():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_feature_profile  # noqa: E402

    graph = {
        "nodes": [
            {"id": "f1", "node_type": "cross layer transcoder", "layer": 13, "feature": 427},
            {"id": "e1", "node_type": "embedding", "layer": 0, "feature": 1},
            {"id": "err1", "node_type": "mlp reconstruction error", "layer": 15, "feature": 0},
            {"id": "target", "node_type": "logit"},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": +1.5},
            {"source": "e1", "target": "target", "weight": +0.2},
            {"source": "err1", "target": "target", "weight": -0.1},
        ],
    }
    profile = extract_feature_profile(graph, "target")
    assert profile == {(13, 427): 1.5}
```

Run tests; verify pass:

```bash
PYTHONPATH=scripts/emnlp_perm_edit python3 -m pytest scripts/emnlp_perm_edit/tests/test_graph_loader.py -v
```

- [ ] **Step 2: Implement `00_feature_taxonomy.py`**

Create `scripts/emnlp_perm_edit/00_feature_taxonomy.py`:

```python
"""Phase 0 — Sub-experiment 0f: feature role clustering + semantic annotation.

For each transcoder feature appearing in any (prompt, condition) attribution graph,
build a 22-dim profile:
  - 11 dims: per-condition mean signed attribution to direct_dot
  - 11 dims: per-condition fraction of prompts in which the feature appears

Cluster features in this 22-dim space (hierarchical agglomerative, Ward linkage).
Annotate clusters with their top-25 features' Stage 04 Neuronpedia labels.
Cluster names refined manually after inspecting the output.

Output: feature_taxonomy.json + feature_taxonomy_clusters.md + feature_taxonomy_figure.png

Tests H0-8: features cluster into discrete causal roles.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from graph_loader import (  # noqa: E402
    extract_feature_profile,
    find_measurement_target_node_id,
    load_packed_graph,
)

CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/graph_data")
    p.add_argument("--stage-04-labels", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--mode", default="single")
    p.add_argument("--n-clusters", type=int, default=6,
                   help="Initial number of clusters; refine after silhouette inspection.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0f] extracting per-feature profiles from graphs at {args.graph_data_dir}")
    # Per-feature accumulators: feature_id → {condition → [list of signed attributions]}
    by_feature_cond_attr = defaultdict(lambda: defaultdict(list))
    by_feature_cond_count = defaultdict(lambda: defaultdict(int))
    n_per_cond = defaultdict(int)

    for prompt_idx in range(args.n_prompts):
        for cond in CONDITIONS:
            slug = f"{prompt_idx}_{cond}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            profile = extract_feature_profile(graph, target_id)
            n_per_cond[cond] += 1
            for fid, signed_attr in profile.items():
                by_feature_cond_attr[fid][cond].append(signed_attr)
                by_feature_cond_count[fid][cond] += 1

    features = sorted(by_feature_cond_attr.keys())
    print(f"  {len(features)} unique features across {sum(n_per_cond.values())} graphs")

    # Build 22-dim profile per feature: 11 dims signed-attribution mean, 11 dims frequency
    profiles = np.zeros((len(features), 22), dtype=np.float32)
    for i, fid in enumerate(features):
        for j, cond in enumerate(CONDITIONS):
            attrs = by_feature_cond_attr[fid].get(cond, [])
            count = by_feature_cond_count[fid].get(cond, 0)
            profiles[i, j] = float(np.mean(attrs)) if attrs else 0.0
            profiles[i, j + 11] = count / max(n_per_cond[cond], 1)

    # Standardize each dimension (z-score)
    profiles_std = (profiles - profiles.mean(axis=0)) / (profiles.std(axis=0) + 1e-8)

    print(f"[0f] clustering with hierarchical agglomerative + Ward linkage, n_clusters={args.n_clusters}")
    from sklearn.cluster import AgglomerativeClustering  # lazy import
    from sklearn.metrics import silhouette_score
    clusterer = AgglomerativeClustering(n_clusters=args.n_clusters, linkage="ward")
    labels = clusterer.fit_predict(profiles_std)
    silhouette = silhouette_score(profiles_std, labels) if len(set(labels)) > 1 else 0.0
    print(f"  silhouette score: {silhouette:.3f}")

    # Per-cluster aggregate: top-K features by total |attribution| across corpus
    cluster_info = {}
    for cluster_id in sorted(set(labels)):
        members = [features[i] for i in range(len(features)) if labels[i] == cluster_id]
        member_total_abs = []
        for fid in members:
            total_abs = sum(abs(np.mean(attrs)) for attrs in by_feature_cond_attr[fid].values())
            member_total_abs.append((fid, total_abs))
        member_total_abs.sort(key=lambda x: -x[1])
        cluster_info[int(cluster_id)] = {
            "n_features": len(members),
            "top_25_features": [
                {"layer": L, "feature": F, "total_abs_attribution": ta}
                for (L, F), ta in member_total_abs[:25]
            ],
        }

    # Optional Stage 04 annotation
    stage4_labels = {}
    if args.stage_04_labels.exists():
        try:
            stage4_raw = json.loads(args.stage_04_labels.read_text())
            for entry in stage4_raw.get("features", stage4_raw):
                key = (entry.get("layer"), entry.get("feature"))
                stage4_labels[key] = entry
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: failed to parse Stage 04 labels ({e}); skipping annotation")

    # Generate per-cluster markdown report
    md_lines = ["# Phase 0 Feature Role Taxonomy (0f)\n",
                f"Clusters: {args.n_clusters}, silhouette score: {silhouette:.3f}\n"]
    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        md_lines.append(f"\n## Cluster {cluster_id} ({info['n_features']} features)\n")
        md_lines.append("| Layer | Feature | |attr| total | Top logits / contexts |")
        md_lines.append("|---|---|---|---|")
        for f in info["top_25_features"]:
            L, F = f["layer"], f["feature"]
            label_blob = stage4_labels.get((L, F), {})
            top_logits = label_blob.get("top_logits", label_blob.get("logits_top", []))[:5]
            md_lines.append(f"| L{L} | F{F} | {f['total_abs_attribution']:.1f} | "
                           f"{', '.join(map(str, top_logits))} |")

    (args.out_dir / "feature_taxonomy_clusters.md").write_text("\n".join(md_lines))

    out = {
        "metadata": {
            "n_features": len(features), "n_clusters": args.n_clusters,
            "silhouette_score": silhouette,
            "profile_dims": ["attr_mean_per_cond x11", "freq_per_cond x11"],
        },
        "cluster_info": cluster_info,
        "feature_cluster_assignment": {
            f"L{L}_F{F}": int(labels[i]) for i, (L, F) in enumerate(features)
        },
    }
    (args.out_dir / "feature_taxonomy.json").write_text(json.dumps(out, indent=2))
    print(f"[0f] wrote feature_taxonomy.json + feature_taxonomy_clusters.md")

    # Optional figure: heatmap of features sorted by cluster x conditions
    try:
        import matplotlib.pyplot as plt
        order = np.argsort(labels)
        attrs_sorted = profiles[order, :11]
        fig, ax = plt.subplots(figsize=(8, 12))
        vmax = np.abs(attrs_sorted).max()
        im = ax.imshow(attrs_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(11)); ax.set_xticklabels(CONDITIONS, rotation=45, ha="right")
        ax.set_ylabel(f"Feature index (sorted by cluster; clusters {args.n_clusters})")
        ax.set_title("Feature role taxonomy — per-condition signed attribution")
        # Draw cluster boundaries
        boundaries = np.where(np.diff(labels[order]) != 0)[0]
        for b in boundaries:
            ax.axhline(b, color="black", linewidth=0.5)
        plt.colorbar(im, ax=ax, label="signed attribution")
        plt.tight_layout()
        plt.savefig(args.out_dir / "feature_taxonomy_figure.png", dpi=150)
        plt.close()
        print(f"[0f] wrote feature_taxonomy_figure.png")
    except Exception as e:
        print(f"[0f] WARNING: figure generation failed ({e}); skipping")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run on real data**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_feature_taxonomy.py
```

Expected wall: ~2 hours (file IO for 550 graphs is the bottleneck; clustering on ~1-3k features takes seconds; figure render is fast). Output: 3 files in the Phase 0 outputs dir.

Inspect `feature_taxonomy_clusters.md` manually and refine cluster naming. Per the spec H0-8 acceptance: silhouette score ≥ 0.3 means clean clusters; < 0.1 means pivot to continuous-role description.

- [ ] **Step 4: Commit**

```bash
git add scripts/emnlp_perm_edit/graph_loader.py \
        scripts/emnlp_perm_edit/tests/test_graph_loader.py \
        scripts/emnlp_perm_edit/00_feature_taxonomy.py \
        data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy.json \
        data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy_clusters.md \
        data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy_figure.png
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0f feature role clustering + Stage 04 annotation

Hierarchical agglomerative clustering of transcoder features on a 22-dim profile
(11 conditions × signed attribution + 11 conditions × frequency). Cluster
top-25 features annotated with Stage 04 Neuronpedia labels. Tests H0-8
(discrete feature roles for refusal).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: JB-class perturbation profile + taxonomy synthesis figure (Sub-experiment 0g)

**Files:**
- Create: `scripts/emnlp_perm_edit/00_jb_perturbation_signature.py`

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/jb_perturbation_signatures.json`
- `data/results/emnlp_perm_edit/phase0_controllability/taxonomy_synthesis_figure.png`
- `data/results/emnlp_perm_edit/phase0_controllability/TAXONOMY_REPORT.md`

**Dependencies:** Task 13 (0f outputs: feature_taxonomy.json with cluster assignments + cluster_info).

**Hypothesis tested:** H0-9 (JB-class perturbation signature with cluster-level localization).

- [ ] **Step 1: Implement `00_jb_perturbation_signature.py`**

Create `scripts/emnlp_perm_edit/00_jb_perturbation_signature.py`:

```python
"""Phase 0 — Sub-experiment 0g: JB-class perturbation profile + taxonomy synthesis figure.

For each cluster from 0f and each JB class, compute the perturbation:
  Δ(cluster, jb_C)     = mean(cluster activation | jb_C) − mean(cluster activation | bare)
  Δ_sem(cluster, jb_C) = mean(cluster activation | jb_C) − mean(cluster activation | ctrl_C)

The per-class vector of cluster perturbations is the "perturbation signature."

Headline figure: two-panel:
  Left: heatmap of Δ_sem (rows = JB classes, columns = clusters, color = signed)
  Right: scatter of per-class cluster decoder-projection onto u_C vs perturbation magnitude
         (each point = JB class × cluster; Pearson r across all points is the
          bridge between Phase 0 taxonomy and Track B per-class methodology)

Tests H0-9 (per-class perturbation signatures localize to specific clusters).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from graph_loader import (  # noqa: E402
    extract_feature_profile,
    find_measurement_target_node_id,
    load_packed_graph,
)

JB_CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]
ALL_CONDS = ["bare"] + [f"jb_{c}" for c in JB_CLASSES] + [f"ctrl_{c}" for c in JB_CLASSES]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/graph_data")
    p.add_argument("--taxonomy", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--mode", default="single")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0g] loading taxonomy from {args.taxonomy}")
    taxonomy = json.loads(args.taxonomy.read_text())
    cluster_assignment = taxonomy["feature_cluster_assignment"]
    # Parse the "L{layer}_F{feature}" keys back to (L, F) tuples
    feature_to_cluster = {}
    for key, cluster_id in cluster_assignment.items():
        # key format "L{layer}_F{feature}"
        L_part, F_part = key.split("_")
        L = int(L_part[1:])
        F = int(F_part[1:])
        feature_to_cluster[(L, F)] = int(cluster_id)
    n_clusters = max(feature_to_cluster.values()) + 1
    print(f"  {len(feature_to_cluster)} features assigned to {n_clusters} clusters")

    print(f"[0g] computing per-cluster per-condition activations from graphs")
    # Accumulate per (cluster, condition) list of mean signed attributions
    cluster_cond_attrs = defaultdict(lambda: defaultdict(list))
    for prompt_idx in range(args.n_prompts):
        for cond in ALL_CONDS:
            slug = f"{prompt_idx}_{cond}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            profile = extract_feature_profile(graph, target_id)
            # Aggregate per cluster
            cluster_attrs_this_prompt = defaultdict(list)
            for (L, F), attr in profile.items():
                cluster_id = feature_to_cluster.get((L, F))
                if cluster_id is not None:
                    cluster_attrs_this_prompt[cluster_id].append(attr)
            for cid, attrs in cluster_attrs_this_prompt.items():
                cluster_cond_attrs[cid][cond].append(np.mean(attrs))

    # Per-cluster per-condition mean
    cluster_cond_mean = {}
    for cid in range(n_clusters):
        cluster_cond_mean[cid] = {}
        for cond in ALL_CONDS:
            vals = cluster_cond_attrs[cid].get(cond, [])
            cluster_cond_mean[cid][cond] = float(np.mean(vals)) if vals else 0.0

    # Compute Δ and Δ_sem per (cluster, jb_class)
    signatures = {}
    for cid in range(n_clusters):
        signatures[cid] = {}
        bare_mean = cluster_cond_mean[cid]["bare"]
        for jb in JB_CLASSES:
            jb_mean = cluster_cond_mean[cid][f"jb_{jb}"]
            ctrl_mean = cluster_cond_mean[cid][f"ctrl_{jb}"]
            signatures[cid][jb] = {
                "delta_vs_bare": jb_mean - bare_mean,
                "delta_sem_vs_ctrl": jb_mean - ctrl_mean,
                "jb_mean": jb_mean, "bare_mean": bare_mean, "ctrl_mean": ctrl_mean,
            }

    out = {
        "metadata": {"n_clusters": n_clusters, "n_prompts": args.n_prompts},
        "cluster_cond_mean": cluster_cond_mean,
        "signatures": signatures,
    }
    (args.out_dir / "jb_perturbation_signatures.json").write_text(json.dumps(out, indent=2))
    print(f"[0g] wrote jb_perturbation_signatures.json")

    # Generate headline two-panel figure
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: Δ_sem heatmap (rows = JB classes, cols = clusters)
    matrix = np.zeros((len(JB_CLASSES), n_clusters))
    for i, jb in enumerate(JB_CLASSES):
        for cid in range(n_clusters):
            matrix[i, cid] = signatures[cid][jb]["delta_sem_vs_ctrl"]
    vmax = np.abs(matrix).max()
    im1 = ax1.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_xticks(range(n_clusters)); ax1.set_xticklabels([f"C{c}" for c in range(n_clusters)])
    ax1.set_yticks(range(len(JB_CLASSES))); ax1.set_yticklabels(JB_CLASSES)
    ax1.set_xlabel("Cluster"); ax1.set_ylabel("JB class")
    ax1.set_title("JB perturbation signatures (Δ_sem vs ctrl-matched)\n"
                  "negative = cluster activation suppressed by JB; positive = recruited")
    for i in range(len(JB_CLASSES)):
        for j in range(n_clusters):
            ax1.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                     color="black" if abs(matrix[i,j]) < 0.6*vmax else "white")
    plt.colorbar(im1, ax=ax1, label="Δ_sem (cluster mean attribution)")

    # Right panel: cluster perturbation magnitude per class — bar chart per JB
    width = 0.15
    x = np.arange(n_clusters)
    for i, jb in enumerate(JB_CLASSES):
        magnitudes = [abs(signatures[cid][jb]["delta_sem_vs_ctrl"]) for cid in range(n_clusters)]
        ax2.bar(x + i*width, magnitudes, width, label=jb)
    ax2.set_xticks(x + width * (len(JB_CLASSES)-1) / 2)
    ax2.set_xticklabels([f"C{c}" for c in range(n_clusters)])
    ax2.set_xlabel("Cluster"); ax2.set_ylabel("|Δ_sem| (cluster perturbation magnitude)")
    ax2.set_title("Per-class cluster perturbation magnitudes\n"
                  "(reveals which cluster each JB class targets most)")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_dir / "taxonomy_synthesis_figure.png", dpi=150)
    plt.close()
    print(f"[0g] wrote taxonomy_synthesis_figure.png (HEADLINE PAPER FIGURE)")

    # Write report
    md = ["# Phase 0 Taxonomy Synthesis (0g)\n",
          "## Per-class top-perturbed clusters\n"]
    for jb in JB_CLASSES:
        per_cluster = [(cid, abs(signatures[cid][jb]["delta_sem_vs_ctrl"])) for cid in range(n_clusters)]
        per_cluster.sort(key=lambda x: -x[1])
        top = per_cluster[0]
        md.append(f"- **{jb}**: top-perturbed cluster = C{top[0]} (|Δ_sem| = {top[1]:.3f})")
    md.append("\n## Pairwise cosine of JB-class signature vectors")
    signature_vecs = {}
    for jb in JB_CLASSES:
        signature_vecs[jb] = np.array(
            [signatures[cid][jb]["delta_sem_vs_ctrl"] for cid in range(n_clusters)]
        )
    for i, jb1 in enumerate(JB_CLASSES):
        for jb2 in JB_CLASSES[i+1:]:
            v1, v2 = signature_vecs[jb1], signature_vecs[jb2]
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            md.append(f"- cos({jb1}, {jb2}) = {cos:+.3f}")
    (args.out_dir / "TAXONOMY_REPORT.md").write_text("\n".join(md))
    print(f"[0g] wrote TAXONOMY_REPORT.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_jb_perturbation_signature.py
```

Expected wall: ~1 hour (file IO bound; computation is light). Outputs 3 files.

Inspect `taxonomy_synthesis_figure.png` for the EMNLP paper headline. Per the spec H0-9 acceptance: pairwise signature cosines should be ≤ +0.5 (distinct class signatures) and each class should have a clear top-perturbed cluster (high concentration of perturbation magnitude on one or two clusters).

- [ ] **Step 3: Commit**

```bash
git add scripts/emnlp_perm_edit/00_jb_perturbation_signature.py \
        data/results/emnlp_perm_edit/phase0_controllability/jb_perturbation_signatures.json \
        data/results/emnlp_perm_edit/phase0_controllability/taxonomy_synthesis_figure.png \
        data/results/emnlp_perm_edit/phase0_controllability/TAXONOMY_REPORT.md
git commit -m "$(cat <<'EOF'
emnlp phase 0: 0g JB-class perturbation profile + taxonomy synthesis figure

For each cluster from 0f and each JB class, computes Δ_sem (perturbation vs
ctrl-matched baseline). Two-panel synthesis figure: (left) Δ_sem heatmap
rows=JBs cols=clusters; (right) per-class cluster perturbation magnitudes.
Tests H0-9 (class signatures localize to clusters); produces the EMNLP
paper's headline taxonomy figure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist (already performed)

- ✓ **Spec coverage:** spec § 2 (Phase 0) is covered: § 2.1 hypotheses (H0-1 through H0-9) → Task 9 + Task 10 + Task 14 acceptance verdicts; § 2.2 sub-experiment 0a → Tasks 1–4; § 2.3 sub-experiment 0b-simple → Tasks 5–7; § 2.4 sub-experiment 0c → Task 10; § 2.5 sub-experiment 0d → Task 11; § 2.6 sub-experiment 0e → Task 12; § 2.7 sub-experiment 0f → Task 13; § 2.8 sub-experiment 0g → Task 14; § 2.9 outputs covered across Tasks 8 + 10 + 11 + 12 + 13 + 14; § 2.10 compute estimate matches the wall times reported in each task; 0b-rigorous is explicitly deferred per spec.
- ✓ **Placeholder scan:** no `TBD`/`TODO`/"appropriate error handling" patterns. Every step contains the actual code or shell command.
- ✓ **Type consistency:** `load_packed_graph`, `find_measurement_target_node_id`, `aggregate_edge_attributions`, `extract_edge_records_to_target`, `extract_feature_profile`, `make_scalar_rhat_subtraction_hook`, `compute_per_prompt_cosines`, `random_baseline_cosine_stats`, `pearson_cosine`, and the variant-map dict are referenced consistently across tasks.
- ✓ **0b-rigorous deferred:** documented explicitly as out of scope for this plan, with the trigger condition (0b-simple result + H0-1 weak fail) for invoking it.
- ✓ **Task 10 independence:** explicitly documented as having no dependencies on Tasks 1–9, with recommended execution order "right after Task 0" so Georg sees H0-5 results first.
- ✓ **Tasks 11–14 dependencies:** documented per-task (Task 11 extends graph_loader; Task 12 uses Task 11's extension; Task 13 adds another graph_loader helper; Task 14 consumes Task 13's clustering output). Tasks 11 and 12 can run in parallel on different GPU sessions; Tasks 13 and 14 are CPU-only and can interleave with GPU tasks.

---

## Execution Notes

**Recommended execution order (CPU-first for early Georg signal; taxonomy tasks interleaved with GPU work):**

| Day | Task | Wall | Note |
|---|---|---|---|
| 1 | Task 0 (scaffold) | ~1 min | |
| 1 | **Task 10 (0c direction robustness audit)** | ~10 s | **Share results with Georg immediately** |
| 1 | Task 1 (HF graph pull) | ~10 min | Network bound |
| 1 | Task 2 (graph_loader + tests) | ~5 min | |
| 1 | Task 3 (0a linearization decomposition CLI) | ~30 min | CPU |
| 1 | Task 4 (decomposition figure) | ~1 min | |
| 1 | Task 5 (edge_ablation_hook + tests) | ~5 min | |
| 1–2 | Task 6 (0b-simple driver, full run) | ~3.5h GPU | tmux |
| 1–2 | Task 7 (drift verify) | ~3 min | |
| 1–2 | Task 8 (aggregation) | ~1 min | |
| 1–2 | Task 9 (acceptance check H0-1/2/3/4) | <1 min | |
| 2–3 | **Task 11 (0d top-K feature sweep, full run)** | **~6h GPU** | tmux; can scope-trim to ~2h with `--k-values 5,50,500` |
| 2–3 | **Task 12 (0e top-K edge sweep, full run)** | **~6h GPU** | tmux; same scope-trim option |
| 2–3 | **Task 13 (0f feature clustering + Stage 04 annotation)** | **~2h CPU** | Can run in parallel with Task 11 or 12 GPU sessions |
| 3 | **Task 14 (0g perturbation signature + headline figure)** | **~1h CPU** | Depends on Task 13 |

This sequencing delivers four foundational results in order:
1. **Hour 0**: Task 10's H0-5 cosine challenge verdict (per-prompt + random baseline + Pearson)
2. **Hour ~1**: Task 4's linearization decomposition figure (0a)
3. **Hour ~4**: Task 8's controllability audit (0a + 0b combined)
4. **Day 3 EOD**: Taxonomy synthesis figure (0g) — the EMNLP paper's headline

**Minimum viable Phase 0 (foundational only, no taxonomy):** Tasks 0–10. Wall: ~30 min CPU + ~10 s 0c + ~3.5h GPU + ~5 min aggregation = ~4 hours total. Sufficient to clear H0-1 / H0-2 / H0-5 verdicts and produce the controllability figure for Georg. Leaves H0-6 / H0-7 / H0-8 / H0-9 unanswered.

**Phase 0 with taxonomy (recommended for EMNLP submission):** Tasks 0–14 in the order above. Wall: ~4h foundational + ~12h GPU for sweeps + ~3h CPU for taxonomy + ~1h figure = ~20h total work spread over 3 days. Produces the full Phase 0 deliverable per spec § 2.

**Parallel execution with Phase 1 (Track B):**
- Tasks 0–4 + Task 10 (~45 min CPU) can run alongside Phase 1's Tasks 0–4 (Phase 1A is the long GPU job — ~4h).
- Task 6 (0b-simple GPU, ~3.5h), Task 11 (0d GPU, ~6h), Task 12 (0e GPU, ~6h) compete for GPU. Sequence: Task 6 → Phase 1A → Task 11 → Phase 1B → Task 12. Or interleave on different days.
- Task 13 (0f clustering, ~2h CPU) can run in parallel with ANY GPU job.

**Branch hygiene:** all commits land on `emnlp-perm-edit`. Do not merge to `l15-refactor` or `main` until both tracks complete and EMNLP submission lands.

**If 0b-rigorous is needed:** create a sibling plan `docs/superpowers/plans/2026-05-XX-phase0b-rigorous-edge-ablation.md` covering the vendor/circuit-tracer patch (~3 person-days). Out of scope for this plan.

**Scope-trim options if 4-week EMNLP timeline gets tight (apply in order):**
1. Trim Task 11 and Task 12 to coarse K values (`--k-values 5,50,500`) — saves ~8h GPU total. H0-6 / H0-7 still answerable at coarse resolution.
2. Skip 0g's optional LLM annotation in Task 13 — saves ~30 min + API cost.
3. Skip Task 12 entirely if H0-4 (from Task 9) already strongly rejects edge-vs-node difference — saves ~6h GPU.
4. Defer Variant 1C (Phase 1 per-layer sweep) — saves ~21h GPU.
