# syntax=docker/dockerfile:1.6
#
# Refusal-Lens pipeline runtime image.
#
# Ships PyTorch 2.4.1 + CUDA 12.1 + cuDNN 9, clones the repo with
# submodules on the branch of your choice, installs the Stage 01-08
# Python dependencies (pinned to match pyproject.toml [runpod] extras),
# and installs the vendored circuit-tracer (which carries the
# measurement_layer/measurement_position patch) in editable mode.
#
# Build:
#   docker build -t refusal-lens:l15 \
#       --build-arg BRANCH=l15-refactor .
#
# Run (interactive, GPU, persistent outputs + HF cache):
#   docker run --rm -it --gpus=all --ipc=host --shm-size=16g \
#       -v "$PWD/outputs":/workspace/outputs \
#       -v "$PWD/hf-cache":/workspace/hf-cache \
#       -e HF_TOKEN="$HF_TOKEN" \
#       -e GITHUB_TOKEN="$GITHUB_TOKEN" \
#       -e GIT_USER_NAME="Mahmoud Shabana" \
#       -e GIT_USER_EMAIL="algoversemechinterp@gmail.com" \
#       refusal-lens:l15
#
# H100 alternative base (faster on Hopper; rebuild if switching):
#   FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ARG REPO_URL=https://github.com/AutoInterp/Refusal-Lens.git
ARG BRANCH=l15-refactor

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

# System packages: git for clone/submodules + commits, curl/jq for HF pushes,
# tini for clean PID-1 signal handling, openssh-client for optional SSH remotes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl jq ca-certificates tini openssh-client less vim \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone the repo on the target branch with its submodules (vendor/circuit-tracer
# carries the refusal-lens-measurement-patch needed for L15 attribution).
RUN git clone --recurse-submodules --branch "${BRANCH}" "${REPO_URL}" Refusal-Lens \
    && cd Refusal-Lens \
    && git submodule update --init --recursive

WORKDIR /workspace/Refusal-Lens

# Python dependencies. Version pins mirror pyproject.toml [runpod] extras so
# RunPod-verified outputs are bit-for-bit reproducible here. circuit-tracer
# installed editable from the submodule — not from git URL — so whatever
# commit the submodule points to is what gets used.
RUN pip install --upgrade pip \
    && pip install \
        "transformers==4.57.3" \
        "huggingface_hub==0.36.2" \
        "accelerate" \
        "nnsight==0.6.1" \
        "safetensors" \
        "einops" \
        "matplotlib" \
        "scipy" \
        "pandas" \
        "seaborn" \
        "upsetplot" \
        "hf_transfer" \
        "tqdm" \
        "ipython" \
    && pip install -e vendor/circuit-tracer

# Convenience: any stage script can be invoked directly as `python
# scripts/pipeline/02_run_attribution.py ...` without needing to export
# PYTHONPATH each time.
ENV PYTHONPATH=/workspace/Refusal-Lens/scripts/pipeline:/workspace/Refusal-Lens/src

# Mount points (bind from host at `docker run` time):
#   /workspace/outputs   → Pipeline run_YYYYMMDD_HHMMSS/ directories
#   /workspace/hf-cache  → HuggingFace model/tokenizer cache (reuse across runs)
# Optional override:
#   /workspace/Refusal-Lens  → mount host repo for live-edit / branch-switch
#                              without rebuilding (host must have submodules
#                              initialized: `git submodule update --init
#                              --recursive` on the host before bind-mounting).
VOLUME ["/workspace/outputs", "/workspace/hf-cache"]

# Entrypoint configures git credentials from env vars (if present) and then
# hands off to the command. Keeps secrets out of image layers.
COPY <<'EOF' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -e

if [ -n "${GIT_USER_NAME:-}" ]; then
    git config --global user.name "${GIT_USER_NAME}"
fi
if [ -n "${GIT_USER_EMAIL:-}" ]; then
    git config --global user.email "${GIT_USER_EMAIL}"
fi
if [ -n "${GITHUB_TOKEN:-}" ] && [ -d /workspace/Refusal-Lens/.git ]; then
    cd /workspace/Refusal-Lens
    # Token-embedded origin for pushes. Idempotent — safe to re-run.
    git remote set-url origin \
        "https://oauth2:${GITHUB_TOKEN}@github.com/AutoInterp/Refusal-Lens.git"
    cd /workspace
fi

# HF_TOKEN is auto-consumed by huggingface_hub; no login step needed.
# Verify the GPU is actually visible.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo "------------------"
fi

exec "$@"
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
