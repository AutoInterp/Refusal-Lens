#!/bin/bash
# RunPod Setup for Refusal-Lens Foundation Branch
# ================================================
# GPU: RTX 4090 (24GB VRAM) or better
# Container disk: 50GB minimum (transcoders ~12GB)
#
# Usage:
#   bash scripts/runpod_setup.sh
#
# After setup, run experiments:
#   python scripts/validate_tejas_replication.py
#   python scripts/generate_comparison_report.py

set -euo pipefail

echo "========================================"
echo "Refusal-Lens RunPod Setup"
echo "========================================"

# Environment
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

# Clone with submodules if not already cloned
if [ ! -d "/workspace/Refusal-Lens" ]; then
    echo "Cloning repository..."
    git clone --recurse-submodules -b foundation \
        https://github.com/AutoInterp/Refusal-Lens.git /workspace/Refusal-Lens
else
    echo "Repository already exists, updating..."
    cd /workspace/Refusal-Lens
    git fetch origin foundation
    git checkout foundation
    git pull origin foundation
    git submodule update --init --recursive
fi

cd /workspace/Refusal-Lens

# Install with pinned versions matching Tejas's environment
echo "Installing dependencies..."
pip install -e ".[runpod]"

# Verify installation
echo ""
echo "========================================"
echo "Verification"
echo "========================================"

echo "1. Package import..."
python -c "import refusal_lens; print('   refusal_lens OK')"

echo "2. Circuit-tracer import..."
python -c "
from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.attribution.targets import CustomTarget
import inspect
sig = inspect.signature(attribute)
has_ml = 'measurement_layer' in sig.parameters
print(f'   circuit-tracer OK (measurement_layer support: {has_ml})')
if not has_ml:
    print('   WARNING: measurement_layer not supported! Check fork branch.')
"

echo "3. GPU check..."
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU detected!'
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'   GPU: {name}, VRAM: {mem:.1f}GB')
assert mem >= 20, f'Need >= 20GB VRAM, got {mem:.1f}GB'
"

# Pre-download model tokenizer (model weights download on first use)
echo "4. Pre-downloading tokenizer..."
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-3-4b-it')
print(f'   Tokenizer cached ({tok.vocab_size} tokens)')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Log in to HuggingFace:  huggingface-cli login"
echo "  2. Run validation:         python scripts/validate_tejas_replication.py"
echo "  3. Generate report:        python scripts/generate_comparison_report.py"
echo ""
echo "Tip: If you get 'Disk quota exceeded', ensure container disk is 50GB."
