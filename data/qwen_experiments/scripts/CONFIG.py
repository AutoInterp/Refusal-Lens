"""
Centralized configuration for Qwen3-4B refusal experiments.

Mirror of `data/tejas_experiments/scripts/` for Gemma-3-4B-IT, adapted for
Qwen/Qwen3-4B + mwhanna/qwen3-4b-transcoders.

Import this from every script:
    from CONFIG import *
"""
from pathlib import Path

# =============================================================================
# MODEL
# =============================================================================
# NOTE: Qwen3-4B is an instruction-tuned causal LM (not multimodal like Gemma3).
# This means `model.config.hidden_size` works directly (no `text_config`).
MODEL_NAME = "Qwen/Qwen3-4B"

# Transcoders for MLP feature decomposition.
#   HuggingFace:  https://huggingface.co/mwhanna/qwen3-4b-transcoders
#   Neuronpedia:  https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/qwen3-4b/
# TODO: Verify the exact subpath after inspecting the HF repo. The Gemma analogue
# was: "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine".
TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"
TRANSCODER_SUBPATH = ""  # FILL IN after inspecting the repo

# =============================================================================
# ARCHITECTURE
# =============================================================================
# Qwen3-4B has 36 transformer blocks; Gemma-3-4B-IT had 34.
# `output_hidden_states` returns N_LAYERS+1 tensors (embedding + N layers).
N_LAYERS = 36

# d_model — read at runtime via `model.config.hidden_size`.
# Hardcoded fallback for reference: Qwen3-4B hidden_size = 2560.
HIDDEN_SIZE = 2560

# =============================================================================
# REFUSAL-DIRECTION HYPERPARAMETERS
# =============================================================================
# IMPORTANT: These were empirically tuned for Gemma-3-4B-IT and MUST be
# re-validated for Qwen3-4B. Run `01_compute_direction_and_sanity.py` first
# to scan positions × layers and pick the best.
#
# Gemma reference (for comparison only — do not assume same for Qwen):
#   GEMMA_BEST_POSITION = -2   # the literal "model" token in Gemma's template
#   GEMMA_BEST_LAYER    = 32   # strongest separation
#   GEMMA_CAUSAL_LAYER  = 15   # causally effective for steering
#
# Qwen's chat template ends with `<|im_start|>assistant\n`, so the last few
# tokens are different. Position -2 in Qwen is the `\n` after `assistant`,
# not a `model` token. Re-tune!
QWEN_BEST_POSITION = -1   # FILL IN after running script 01
QWEN_BEST_LAYER = 34      # FILL IN after running script 01
QWEN_CAUSAL_LAYER = None    # FILL IN after running script 15/16/17

# Positions to sweep when computing the direction (last N tokens of prompt).
SWEEP_POSITIONS = [-5, -4, -3, -2, -1]

# Numerical-precision settings (Georg's corrections — keep these for Qwen).
ACCUMULATION_DTYPE = "float64"   # never bfloat16 for the direction itself
PADDING_SIDE = "left"

# =============================================================================
# DATA
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"

HARMFUL_TRAIN = DATASET_DIR / "refusal_direction_dataset" / "splits" / "harmful_train.json"
HARMLESS_TRAIN = DATASET_DIR / "refusal_direction_dataset" / "splits" / "harmless_train.json"
CONTROLLED_DATASET = DATASET_DIR / "refusal_lens_controlled_dataset.json"

N_TRAIN_SAMPLES = 64

# =============================================================================
# OUTPUT PATHS
# =============================================================================
QWEN_DIR = PROJECT_ROOT / "data" / "qwen_experiments"
RESULTS_DIR = QWEN_DIR / "results"
RESULTS_V2_DIR = QWEN_DIR / "results_v2"
FIGURES_DIR = QWEN_DIR / "figures"

for d in (RESULTS_DIR, RESULTS_V2_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CHAT TEMPLATE HELPER
# =============================================================================
def format_prompt(text: str, tokenizer) -> str:
    """Apply Qwen's chat template with assistant-generation prompt.

    enable_thinking=False matches Gemma-3-4B-IT (no thinking mode). With the
    default enable_thinking=True the template appends `<think>\\n` after
    `<|im_start|>assistant\\n`, which shifts every trailing-token position
    the refusal-direction sweep analyzes. The transcoders were trained on
    Qwen/Qwen3-4B with no enforced thinking-mode prefix.
    """
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def get_hidden_size(model) -> int:
    """Qwen has flat config; Gemma3 has nested text_config. This handles both."""
    if hasattr(model.config, "text_config"):
        return model.config.text_config.hidden_size
    return model.config.hidden_size


def get_decoder_layers(model):
    """Return the iterable of transformer blocks (handles wrapper differences)."""
    # Qwen3 layout: model.model.layers
    # Gemma3 layout: model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers
