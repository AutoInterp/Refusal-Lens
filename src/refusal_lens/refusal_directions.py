"""
Refusal direction computation (Arditi et al. 2024)

Compute \tilde{r} = E[x|harmful] - E[x|harmless] via difference-in-means or PCA.
This direction is the attribution target for circuit-tracer integration.

Requires optional deps: torch, transformers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def _require_torch() -> None:
    if not HAS_TORCH:
        msg = (
            "torch is required for refusal direction computation. "
            "Install with: pip install refusal-lens[steering]"
        )
        raise ImportError(msg)
    
# data classes
@dataclass
class RefusalDirectionResult:
    """Refusal direction for a single layer"""

    direction: torch.Tensor # unit-norm vector, shape (d_model,)
    layer: int
    method: str # 'mean' or 'pca'
    separation: float # ||E[harmful] - E[harmless]|| before normalization

@dataclass
class RefusalDirectionSet:
    """Collection of refusal directions across layers"""

    directions: dict[int, RefusalDirectionResult]
    best_layer: int
    metadata: dict[str, Any] = field(default_factory=dict)

# model layer discovery
def get_model_layers(model: Any) -> Any:
    """
    Return the ordered sequence of transformer block modules.

    Supports Gemma, Llama, GPT-NeoX, and GPT-2 style architectures.

    Args:
        model: A HuggingFace CausalLM model

    Returns:
        the ``nn.ModuleList`` of transformer blocks.
    
    Raises:
        ValueError: if the architecture is not recognized
    """
    _require_torch()
    # gemma-2 / llama style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # gemma-3 style (language_model wrapper)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers

    # gemma-3 nested (model.language_model.model.layers)
    if (hasattr(model, "language_model")
            and hasattr(model.language_model, "model")
            and hasattr(model.language_model.model, "layers")):
        return model.language_model.model.layers

    # gpt-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # gpt-neox style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    msg = (
        "Cannot find transformer blocks. "
        "Supported: model.model.layers, model.language_model.layers, "
        "model.transformer.h, model.gpt_neox.layers"
    )
    raise ValueError(msg)

# activation gathering
def gather_residual_activations(
    model: Any,
    layer_idx: int,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Capture residual-stream activations at a single layer.

    Uses ``output_hidden_states=True`` instead of forward hooks to avoid
    architecture-specific issues (e.g., Gemma-3 layer-33 RMSNorm mismatch).

    Args:
        model: A huggingface CausalLM model.
        layer_idx: Index of the transformer block (0-indexed).
        input_ids: Tokenized input, shape ``(1, seq_len)``

    Returns:
        Tensor of shape ``(1, seq_len, d_model)`` - the layer's output hidden
        state (detached, on the same device as the model).
    """
    _require_torch()
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
    return out.hidden_states[layer_idx + 1].detach()

def collect_resid_acts_multipos(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    *,
    positions: str | list[int] = "last",
    max_prompts: int | None = None,
) -> dict[int, torch.Tensor]:
    """
    Collect residual activations at multiple token positions.

    Args:
        model: A HuggingFace CausalLM model.
        tokenizer: The model's tokenizer.
        prompts: list of text prompts.
        layers: Layer indices to collect from.
        positions: which token positions to capture:
            - ``'last'``: only the final token
            - ``'last_k'``: last min(5, seq_len) tokens
            - ``'all'``: every token position
            - A list of int indices (negative indexing supported)
        max_prompts: cap the number of prompts processed.
    
    Returns:
        Dict mapping layer index -> tensor of shape
        ``(n_prompts, n_positions, d_model)``.
    """
    _require_torch()
    from .model_loader import tokenize_prompt

    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for i, prompt in enumerate(prompts):
        if i % 20 == 0:
            logger.info("  Collecting activations: %d/%d", i, len(prompts))
        input_ids = tokenize_prompt(prompt, tokenizer)
        seq_len = input_ids.shape[1]

        if positions == "last":
            pos_indices = [-1]
        elif positions == "all":
            pos_indices = list(range(seq_len))
        elif positions == "last_k":
            k = min(5, seq_len)
            pos_indices = list(range(-k, 0))
        elif isinstance(positions, list):
            pos_indices = positions
        else:
            msg = f"Unknown positions mode: {positions!r}"
            raise ValueError(msg)
        
        for l in layers:
            resid = gather_residual_activations(model, l, input_ids)
            pos_acts = [resid[0, p, :].double().cpu() for p in pos_indices]
            acts[l].append(torch.stack(pos_acts))
    
    logger.info("  Done: %d prompts.", len(prompts))
    return {l: torch.stack(acts[l]) for l in layers}

# direction computations
def compute_refusal_directions(
    harmful_acts: dict[int, torch.Tensor],
    harmless_acts: dict[int, torch.Tensor],
    method: str = "mean",
) -> dict[int, RefusalDirectionResult]:
    """
    Compute refusal direction \tilde{r} for each layer.

    Args:
        harmful_acts: Layer -> tensor of shape ``(n_prompts, n_positions, d_model)``.
        harmless_acts: Same shape as ``harmful_acts``.
        method:
            - ``'mean'``: \tilde{r} = normalize(E[harmful] - E[harmless])
            - ``'pca'``: \tilde{r} = top singular vector of per-sample differences
    
    Returns:
        Dict mapping layer index -> :class:`RefusalDirectionResult`.
    """
    _require_torch()
    results: dict[int, RefusalDirectionResult] = {}

    for l in harmful_acts:
        h_mean = harmful_acts[l].double().mean(dim=(0, 1))
        s_mean = harmless_acts[l].double().mean(dim=(0, 1))
        diff = h_mean - s_mean
        separation = diff.norm().item()

        if method == "mean":
            direction = diff / diff.norm().item()
        elif method == "pca":
            # avg over positions, keep per-sample diffs
            h_flat = harmful_acts[l].mean(dim=1) # (n_prompts, d_model)
            s_flat = harmless_acts[l].mean(dim=1)
            diffs = h_flat - s_flat
            diffs = diffs - diffs.mean(0) # center
            _U, _S, Vh = torch.linalg.svd(diffs, full_matrices=False)
            direction = Vh[0]
        else:
            msg = f"Unknown method: {method!r}. Use 'mean' or 'pca'."
            raise ValueError(msg)
        
        results[l] = RefusalDirectionResult(
            direction=direction,
            layer=l,
            method=method,
            separation=separation,
        )
    
    return results

def find_best_layer(
    directions: dict[int, RefusalDirectionResult],
) -> tuple[int, RefusalDirectionResult]:
    """
    Select the layer with the strongest harmful/harmless separation.

    Args:
        directions: dict from :func:`compute_refusal_directions`.
    
    Returns:
        Tuple of ``(best_layer_index, best_result)``.
    
    Raises:
        ValueError: if ``directions`` is empty.
    """
    if not directions:
        msg = "No directions provided."
        raise ValueError(msg)
    best_layer = max(directions, key=lambda l: directions[l].separation)
    return best_layer, directions[best_layer]

def save_directions(
    directions: dict[int, RefusalDirectionResult],
    output_dir: str | Path,
    *,
    best_layer: int | None = None,
) -> Path:
    """
    Save computed refusal directions to disk.

    Creates one ``.pt`` file per layer and a ``summary.json`` with metadata.

    Args:
        directions: dict from :func:`compute_refusal_directions`.
        output_dir: directory to write files into (created if needed).
        best_layer: override for best layer; auto-detectd if None.

    Returns:
        Path to the output directory.
    """
    _require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_layer is None:
        best_layer, _ = find_best_layer(directions)
    
    summary: dict[str, Any] = {
        "best_layer": best_layer,
        "layers": {},
    }

    for layer, result in sorted(directions.items()):
        pt_path = output_dir / f"layer_{layer:02d}.pt"
        torch.save(result.direction, pt_path)
        summary["layers"][str(layer)] = {
            "method": result.method,
            "separation": round(result.separation, 6)
        }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %d directions to %s", len(directions), output_dir)
    return output_dir

def load_directions(
    input_dir: str | Path,
) -> RefusalDirectionSet:
    """
    Load previously saved refusal directions from disk.

    Args:
        input_dir: directory containing ``.pt`` files and ``summary.json``.
    
    Returns:
        A :class:`RefusalDirectionSet` with all loaded directions.
    
    Raises:
        FileNotFoundError: if ``summary.json`` is missing.
    """
    _require_torch()
    input_dir = Path(input_dir)
    summary_path = input_dir / "summary.json"

    if not summary_path.exists():
        msg = f"summary.json not found in {input_dir}"
        raise FileNotFoundError(msg)
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    directions: dict[int, RefusalDirectionResult] = {}
    for layer_str, info in summary["layers"].items():
        layer = int(layer_str)
        pt_path = input_dir / f"layer_{layer:02d}.pt"
        direction = torch.load(pt_path, weights_only=True)
        directions[layer] = RefusalDirectionResult(
            direction=direction,
            layer=layer,
            method=info["method"],
            separation=info["separation"],
        )
    
    return RefusalDirectionSet(
        directions=directions,
        best_layer=summary["best_layer"],
        metadata=summary,
    )