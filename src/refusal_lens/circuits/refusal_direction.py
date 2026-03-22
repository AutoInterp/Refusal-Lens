"""Refusal direction computation using difference-in-means.

This module implements Step 1 of the research plan:
Compute refusal directions r_ℓ = E[x_ℓ | harmful] - E[x_ℓ | harmless]

Following Arditi et al. 2024's approach for computing refusal directions
using the difference in mean activations between harmful and harmless prompts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from ..config import config


@dataclass
class RefusalDirectionResult:
    """Container for refusal direction computation results.

    Attributes:
        layer: The layer index where the direction was computed
        direction: The refusal direction vector (normalized, shape: d_model)
        separation: Separation score showing how well the direction
            separates harmful from harmless activations
        mean_harmful: Mean activation for harmful prompts
        mean_harmless: Mean activation for harmless prompts
        d_model: Hidden dimension size
    """

    layer: int
    direction: torch.Tensor
    separation: float
    mean_harmful: torch.Tensor
    mean_harmless: torch.Tensor
    d_model: int

    def save(self, filepath: str) -> None:
        """Save the result to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "layer": self.layer,
                "direction": self.direction,
                "separation": self.separation,
                "mean_harmful": self.mean_harmful,
                "mean_harmless": self.mean_harmless,
                "d_model": self.d_model,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> RefusalDirectionResult:
        """Load a result from a file."""
        data = torch.load(filepath)
        return cls(
            layer=data["layer"],
            direction=data["direction"],
            separation=data["separation"],
            mean_harmful=data["mean_harmful"],
            mean_harmless=data["mean_harmless"],
            d_model=data["d_model"],
        )


class RefusalDirectionComputer:
    """Compute refusal directions for each layer.

    This class implements the difference-in-means approach for computing
    refusal directions: r_ℓ = E[x_ℓ | harmful] - E[x_ℓ | harmless]

    Where x_ℓ is the residual stream activation at layer ℓ.

    Example:
        >>> from refusal_lens import RefusalDirectionComputer, config
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
        >>> tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        >>> computer = RefusalDirectionComputer(model, tokenizer)
        >>> results = computer.compute_all_layers(harmful_prompts, harmless_prompts)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str | None = None,
    ):
        """Initialize the refusal direction computer.

        Args:
            model: Language model (e.g., from transformers)
            tokenizer: Tokenizer for the model
            device: torch device (cuda or cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or config.DEVICE
        self.model.eval()

        # Get model architecture info
        self._d_model = self._get_hidden_size()
        self._n_layers = self._get_num_layers()

    def _get_hidden_size(self) -> int:
        """Get the hidden dimension size of the model."""
        # Try common attribute names
        for attr in ["hidden_size", "d_model", "embed_dim", "head_dim"]:
            if hasattr(self.model.config, attr):
                value = getattr(self.model.config, attr)
                if isinstance(value, int) and value is not None:
                    return value
        # Fallback: try to infer from embedding layer
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "embed_tokens"):
                return int(self.model.model.embed_tokens.weight.shape[-1])
            # Gemma-3: check language_model
            if hasattr(self.model.model, "language_model"):
                lm = self.model.model.language_model
                if hasattr(lm, "embed_tokens"):
                    return int(lm.embed_tokens.weight.shape[-1])
                if hasattr(lm.config, "hidden_size"):
                    val = lm.config.hidden_size
                    if val is not None:
                        return int(val)
        if hasattr(self.model, "embed_tokens"):
            return int(self.model.embed_tokens.weight.shape[-1])
        msg = "Cannot determine hidden size from model config"
        raise ValueError(msg)

    def _get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        if hasattr(self.model.config, "num_hidden_layers"):
            value = self.model.config.num_hidden_layers
            if isinstance(value, int) and value is not None:
                return value
        # Gemma-3: check language_model
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "language_model"):
                lm = self.model.model.language_model
                if hasattr(lm.config, "num_hidden_layers"):
                    val = lm.config.num_hidden_layers
                    if val is not None:
                        return int(val)
                if hasattr(lm, "layers"):
                    return len(lm.layers)
            if hasattr(self.model.model, "layers"):
                return len(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return len(self.model.layers)
        msg = "Cannot determine number of layers from model config"
        raise ValueError(msg)

    def get_residual_stream(
        self,
        prompt: str,
        layer: int,
        position: str | int = "last_prompt_token",
    ) -> torch.Tensor:
        """Get residual stream activation at specified layer and position.

        Args:
            prompt: Input prompt string
            layer: Layer index
            position: Position to extract ("last_prompt_token", "first_token",
                     or integer index)

        Returns:
            Residual stream tensor of shape (d_model,)
        """
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq_len = tokens.shape[1]

        # Storage for activations
        activations = {}

        def create_hook(name: str) -> Any:
            """Create a hook function that stores activations."""

            def hook(_module: Any, _input: Any, output: Any) -> None:
                # For transformer layers, output is typically a tuple
                # output[0] is the hidden states
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()

            return hook

        # Register hooks on all layers to get intermediate activations
        handles = []
        try:
            # Get the layers module
            if hasattr(self.model, "model"):
                if hasattr(self.model.model, "language_model"):
                    # Gemma-3: use language_model.layers
                    layers_module = self.model.model.language_model.layers
                elif hasattr(self.model.model, "layers"):
                    layers_module = self.model.model.layers
                else:
                    layers_module = self.model.model
            elif hasattr(self.model, "layers"):
                layers_module = self.model.layers
            else:
                layers_module = None

            if layers_module is None or not hasattr(layers_module, "__len__"):
                msg = f"Cannot find layers module for model type {type(self.model).__name__}"
                raise ValueError(msg)

            # Register hook on the specific layer
            handle = layers_module[layer].register_forward_hook(
                create_hook(f"layer_{layer}")
            )
            handles.append(handle)

            # Run model
            with torch.no_grad():
                _ = self.model(tokens)

            # Extract activation
            if f"layer_{layer}" not in activations:
                msg = f"Failed to capture activation at layer {layer}"
                raise ValueError(msg)

            activation = activations[f"layer_{layer}"]  # (batch=1, seq_len, d_model)

            # Get position index
            if position == "last_prompt_token":
                # Find the last token of the prompt (before any generated tokens)
                pos_idx = seq_len - 1
            elif position == "first_token":
                pos_idx = 0
            elif isinstance(position, int):
                pos_idx = position
            else:
                msg = f"Unknown position: {position}"
                raise ValueError(msg)

            # Clamp to valid range
            pos_idx = min(max(0, pos_idx), activation.shape[1] - 1)

            return activation[0, pos_idx, :].cpu()

        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

    def compute_direction(
        self,
        layer: int,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
        position: str = "last_prompt_token",
        show_progress: bool = True,
    ) -> RefusalDirectionResult:
        """Compute refusal direction for a single layer.

        Computes: r_ℓ = E[x_ℓ | harmful] - E[x_ℓ | harmless]

        Args:
            layer: Layer index to compute direction for
            harmful_prompts: List of harmful prompts
            harmless_prompts: List of harmless prompts
            position: Position to extract activations from
            show_progress: Whether to show progress bar

        Returns:
            RefusalDirectionResult containing the computed direction
        """
        if show_progress:
            print(f"\n[Layer {layer}] Computing refusal direction...")

        # Collect activations for harmful prompts
        harmful_acts = []
        harmful_iter = (
            tqdm(harmful_prompts, desc=f"Layer {layer} - Harmful")
            if show_progress
            else harmful_prompts
        )
        for prompt in harmful_iter:
            act = self.get_residual_stream(prompt, layer, position)
            harmful_acts.append(act)

        # Collect activations for harmless prompts
        harmless_acts = []
        harmless_iter = (
            tqdm(harmless_prompts, desc=f"Layer {layer} - Harmless")
            if show_progress
            else harmless_prompts
        )
        for prompt in harmless_iter:
            act = self.get_residual_stream(prompt, layer, position)
            harmless_acts.append(act)

        # Stack into tensors
        harmful_acts_tensor = torch.stack(harmful_acts)  # (n_harmful, d_model)
        harmless_acts_tensor = torch.stack(harmless_acts)  # (n_harmless, d_model)

        # Compute means
        mean_harmful = harmful_acts_tensor.mean(dim=0)
        mean_harmless = harmless_acts_tensor.mean(dim=0)

        # Compute refusal direction
        direction = mean_harmful - mean_harmless

        # Normalize to unit vector
        direction = direction / (direction.norm() + 1e-8)

        # Compute separation score
        # How well does the direction separate the two classes?
        harmful_proj = (harmful_acts_tensor @ direction).mean().item()
        harmless_proj = (harmless_acts_tensor @ direction).mean().item()
        separation = harmful_proj - harmless_proj

        if show_progress:
            print(f"  Layer {layer}: separation = {separation:.4f}")

        return RefusalDirectionResult(
            layer=layer,
            direction=direction,
            separation=separation,
            mean_harmful=mean_harmful,
            mean_harmless=mean_harmless,
            d_model=self._d_model,
        )

    def compute_all_layers(
        self,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
        layers: list[int] | None = None,
        position: str = "last_prompt_token",
    ) -> dict[int, RefusalDirectionResult]:
        """Compute refusal directions for multiple layers.

        Args:
            harmful_prompts: List of harmful prompts
            harmless_prompts: List of harmless prompts
            layers: List of layer indices (defaults to config.REFUSAL_LAYERS)
            position: Position to extract activations from

        Returns:
            Dictionary mapping layer indices to RefusalDirectionResults
        """
        if layers is None:
            layers = config.REFUSAL_LAYERS

        results = {}
        for layer in layers:
            result = self.compute_direction(
                layer, harmful_prompts, harmless_prompts, position
            )
            results[layer] = result

        return results


def save_directions(
    results: dict[int, RefusalDirectionResult], output_dir: str
) -> None:
    """Save computed directions to disk.

    Args:
        results: Dictionary of layer -> RefusalDirectionResult
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save each direction
    for layer, result in results.items():
        filepath = os.path.join(output_dir, f"layer_{layer:02d}.pt")
        result.save(filepath)

    # Save summary as JSON
    summary = {
        str(layer): {
            "separation": result.separation,
            "direction_norm": result.direction.norm().item(),
            "d_model": result.d_model,
        }
        for layer, result in results.items()
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved {len(results)} directions to {output_dir}")


def load_directions(
    input_dir: str,
) -> dict[int, RefusalDirectionResult]:
    """Load computed directions from disk.

    Args:
        input_dir: Directory containing saved directions

    Returns:
        Dictionary mapping layer indices to RefusalDirectionResults
    """
    results = {}

    # Load summary to get layer list
    summary_path = os.path.join(input_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        layer_list = [int(k) for k in summary]
    else:
        # Fallback: scan for layer_XX.pt files
        layer_list = []
        for filename in os.listdir(input_dir):
            if filename.startswith("layer_") and filename.endswith(".pt"):
                layer = int(filename.split("_")[1].split(".")[0])
                layer_list.append(layer)
        layer_list.sort()

    # Load each direction
    for layer in layer_list:
        filepath = os.path.join(input_dir, f"layer_{layer:02d}.pt")
        if os.path.exists(filepath):
            results[layer] = RefusalDirectionResult.load(filepath)

    print(f"✅ Loaded {len(results)} directions from {input_dir}")
    return results


def find_best_layer(
    results: dict[int, RefusalDirectionResult],
) -> tuple[int, RefusalDirectionResult]:
    """Find the layer with the highest separation score.

    Args:
        results: Dictionary of layer -> RefusalDirectionResult

    Returns:
        Tuple of (best_layer, best_result)
    """
    best_layer = max(results.keys(), key=lambda layer: results[layer].separation)
    return best_layer, results[best_layer]
