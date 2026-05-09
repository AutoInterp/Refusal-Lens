# Refusal-Lens — ICML 2026 Mech-Interp Workshop Code Submission

Code accompanying the anonymous submission *"Jailbreaks Edit the Refusal
Direction: Per-Class Causal Decomposition and Direction-Aligned Attribution"*
(see `ICML2026_v2.tex`).

This repository reproduces the paper's numerical results on
**Gemma-3-4B-IT** with a controlled `50 prompts × 11 conditions` suite. The
headline causal claims are:

- A bidirectional refusal axis at `ℓ=15` (89/89 pro-refusal flips, 49/50
  anti-refusal flips, 10/10 benign force-refusals).
- Per-class jailbreak direction subtraction flips **93.3%** of
  jailbreak-comply prompts, vs. **52.8%** for the class-averaged universal
  vector at the same dose.
- A magnitude-matched **fiction**-class vector flips **50/50** refusing
  prompts, exceeding the canonical Arditi −r̂ intervention (49/50).
- Sparse subcircuits constructed from direction-aligned attribution graphs
  recover at most **~35%** of what the directional intervention recovers.

## Repository layout

```
src/refusal_lens/         Python package: model loading, direction extraction,
                          attribution glue, refusal classifier, supernode
                          analysis.
scripts/pipeline/         Numbered stages reproducing the paper:
  01_compute_direction.py     direction extraction (§ 2.2)
  02_run_attribution.py       direction-aligned CLT attribution
  02b_statistical_analysis.py per-class cosines, magnitudes, Cohen's d (§ 3.2)
  03_verify_attribution.py    linearization-identity sanity check
  04_label_features.py        Gemma-Scope dashboard feature labels
  06_causal_intervention.py   bidirectional Arditi intervention (§ 3.1)
  07_identify_subcircuits.py  rule-based subcircuit construction
  08_ablate_subcircuits.py    sparse-feature ablation experiment
  08_renorm_baselines.py      Stage 06-aligned baseline normalisation
  08_top_n_sweep.py           Tier-1 per-prompt top-N sweep (Fig. F5)
scripts/analysis/         Per-class JB-vector interventions (§ 3.3, § 3.4)
                          and the paper-figure generator
                          (`generate_paper_figures.py` produces F1, F1B, F3-F6).
test/                     Unit tests for the `refusal_lens` package.
scripts/pipeline/tests/   End-to-end / pipeline-local tests
                          (`test_pipeline_local.py`).
dataset/                  Controlled 50×11 prompt suite + the harmful/harmless
                          splits used to extract r.
data/results/pipeline_runs/run_20260430_023247/
                          Canonical paper run (full intermediate artefacts).
data/results/pipeline_runs/run_20260430_023247_{canonical,full,topN}_*/
                          Subcircuit-ablation sweep variants used for Fig. F5.
figures/                  Paper figures (PDFs + PNG previews) and the recovery
                          summary tables/CSVs.
vendor/circuit-tracer.tar.gz
                          Vendored fork of Anthropic's circuit-tracer with
                          the multi-position-measurement patch the paper
                          requires. Extract before installing (see Setup).
ICML2026_v2.tex           Anonymous paper source.
```

## Setup

```bash
# Python 3.10 recommended; tested on 3.10.12.
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[gpu]

# Extract and install the vendored circuit-tracer fork:
mkdir -p vendor && tar -xzf vendor/circuit-tracer.tar.gz -C vendor
pip install -e vendor/circuit-tracer
```

CUDA 12.1+ is required for the GPU stages (Stage 02 attribution and
Stage 06 causal intervention). All non-GPU analysis stages
(02b / 03 / 04 / 07 / 08* / figure generation) run on CPU.

Optional Docker workflow: `Dockerfile.local` provides a CUDA 12.1 base image
with the runtime dependencies preinstalled (build instructions in the
file's header comment).

## Quick verification — runs without GPU

The canonical paper run (`run_20260430_023247/`) and the subcircuit-ablation
sweep variants are committed under `data/results/pipeline_runs/`. The
following commands reproduce every paper figure from those artefacts in
under five minutes on a laptop:

```bash
# Regenerate F1, F1B, F3-F6 + recovery_table.csv + recovery_with_ci.json
PYTHONPATH=src python3 scripts/analysis/generate_paper_figures.py

# Sanity-check the per-class causal flip rates (Tab. 2, Tab. 3 in the paper)
python3 -c "
import json, pathlib
RUN = pathlib.Path('data/results/pipeline_runs/run_20260430_023247')
print('Stage 06 bidirectional flip rates:')
print(json.dumps(json.load(open(RUN/'06_causal/causal_summary.json'))['summary'], indent=2))
print()
print('Per-class JB-vector causal results:')
r = json.load(open(RUN/'06_causal/jb_vector_intervention_per_class_results.json'))
for k in ('experiment_a_mitigate_jb_subtract_per_class_rjb', 'experiment_b_induce_jb_add_per_class_rjb'):
    print(k)
    print(json.dumps(r[k]['summary'], indent=2))
"

# Run the unit + pipeline-local test suites
PYTHONPATH=src python3 -m pytest test/ -v -W ignore::DeprecationWarning
PYTHONPATH=src python3 scripts/pipeline/tests/test_pipeline_local.py --stage all
```

## Paper section ↔ script map

| Paper section                                   | Script(s)                                                | Output(s)                                                                           |
| ---                                             | ---                                                      | ---                                                                                 |
| § 2.2 / § 3 Replication of the 1-D refusal axis | `06_causal_intervention.py`                              | `06_causal/causal_summary.json`, `intervention_symmetry.png`                        |
| § 3.2 Per-class displacement (Tab. 1, Fig. 1)   | `02b_statistical_analysis.py`, `generate_paper_figures.fig1` | `02b_stats/direction_alignment.json`, `figures/F1_per_class_geometry.{pdf,png}` |
| § 3.2 Raw vs. semantic (App. F1B)               | `generate_paper_figures.fig1b`                           | `figures/F1B_raw_vs_semantic_cosines.{pdf,png}`                                     |
| § 3.3 Mitigate experiment (Tab. 2)              | `scripts/analysis/jb_vector_intervention_per_class.py`   | `06_causal/jb_vector_intervention_per_class_results.json` (Exp. A)                  |
| § 3.4 Induce experiment (Tab. 3)                | `scripts/analysis/jb_vector_intervention_per_class.py`   | same file (Exp. B)                                                                  |
| § (attribution) Direction-aligned graphs        | `02_run_attribution.py`                                  | `02_attribution/`                                                                   |
| § (attribution) Linearization sanity            | `03_verify_attribution.py`                               | `03_verification/`                                                                  |
| § (subcircuits) Rule-based construction         | `07_identify_subcircuits.py`                             | `07_subcircuits/subcircuits.json`, contrast figures                                 |
| § (ablation) Subcircuit ablation, Fig. F5       | `08_ablate_subcircuits.py` + `08_renorm_baselines.py` + `generate_paper_figures.fig5` | `08_ablation/ablation_summary*.json`, `figures/F5_recovery_vs_features_pareto.png` |
| § (ablation) Tier-1 top-N sweep                 | `08_top_n_sweep.py`                                      | feeds Fig. F5 and `recovery_table.csv`                                              |

## Running the pipeline from scratch

The full pipeline takes ≈ 24 GPU-hours on a single H100 SXM (Stage 02
attribution dominates). All commands assume `PYTHONPATH=src` and a
Hugging Face token in `HF_TOKEN`.

```bash
RUN=data/results/pipeline_runs/$(date -u +run_%Y%m%d_%H%M%S)

# 1. Per-layer refusal direction
python3 scripts/pipeline/01_compute_direction.py --output-dir "$RUN"

# 2. Direction-aligned CLT attribution at L15 (multi-position)
python3 scripts/pipeline/02_run_attribution.py --run-dir "$RUN"

# 2b. Per-class statistics + cosines/magnitudes
python3 scripts/pipeline/02b_statistical_analysis.py --run-dir "$RUN"

# 3. Linearization-identity verification (≥ 45/50 prompts must match)
python3 scripts/pipeline/03_verify_attribution.py --run-dir "$RUN"

# 4. Feature labels via Gemma-Scope HF dashboards
python3 scripts/pipeline/04_label_features.py --run-dir "$RUN"

# 6. Bidirectional causal intervention at L15
python3 scripts/pipeline/06_causal_intervention.py --run-dir "$RUN"

# Per-class JB vector causal experiments (Exp. A and Exp. B)
python3 scripts/analysis/jb_vector_intervention_per_class.py --run-dir "$RUN"

# 7. Subcircuit construction + 8. Ablation (Fig. F5 inputs)
python3 scripts/pipeline/07_identify_subcircuits.py --run-dir "$RUN"
python3 scripts/pipeline/08_ablate_subcircuits.py --run-dir "$RUN"
python3 scripts/pipeline/08_renorm_baselines.py --run-dir "$RUN"
python3 scripts/pipeline/08_top_n_sweep.py --run-dir "$RUN"

# Regenerate every paper figure from the produced artefacts
python3 scripts/analysis/generate_paper_figures.py
```

## Pre-computed artefacts on Hugging Face

The committed `data/results/pipeline_runs/run_20260430_023247/` directory
contains the JSON / PNG / Markdown summaries used to regenerate every paper
figure (≈ 280 MB total). The raw `.pt` attribution graphs (≈ 75 GB) are not
committed; an anonymous Hugging Face dataset hosts the full artefact bundle
and can be pulled with the helper script:

```bash
python3 scripts/anon_hf_download.py --hf-repo <anonymous-org>/refusal-lens-icml2026-data
```

The repository identifier and download instructions are listed in the
paper's reproducibility appendix.

## Citation

The paper reference appears in `ICML2026_v2.tex`. Anonymous submission;
please cite as the venue requires once authorship is de-anonymised.

## License

MIT — see `LICENSE`.
