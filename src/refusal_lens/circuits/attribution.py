"""Attribution circuit analysis for refusal behavior.

This module implements Step 2 of the research plan:
Analyze how refusal is computed by tracing attributions to the refusal direction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..config import config
from .refusal_direction import RefusalDirectionResult


@dataclass
class AttributionResult:
    """Container for attribution analysis results.

    Attributes:
        prompt: The input prompt
        refusal_direction: The target refusal direction
        layer: The layer at which attribution was computed
        token_attributions: Attribution scores for each token
        top_features: List of (feature_idx, score) pairs sorted by importance
        attribution_graph: The constructed attribution graph
    """

    prompt: str
    refusal_direction: torch.Tensor
    layer: int
    token_attributions: list[float] = field(default_factory=list)
    top_features: list[tuple[int, float]] = field(default_factory=list)
    attribution_scores: torch.Tensor | None = None


@dataclass
class AttributionNode:
    """A node in the attribution graph.

    Attributes:
        feature_idx: Index of the feature/neuron
        layer: Layer where the feature is active
        attribution_score: Importance score relative to refusal
        activation: Raw activation value
    """

    feature_idx: int
    layer: int
    attribution_score: float
    activation: float

    def __repr__(self) -> str:
        return f"AttributionNode(feature={self.feature_idx}, layer={self.layer}, score={self.attribution_score:.4f})"


class AttributionGraph:
    """Attribution circuit graph for analyzing refusal computation.

    This class builds a graph of features that contribute to refusal,
    using the refusal direction as the target for attribution.

    Example:
        >>> from refusal_lens import AttributionGraph, RefusalDirectionComputer
        >>> graph = AttributionGraph(model, refusal_direction_result)
        >>> result = graph.analyze(prompt)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        refusal_direction: RefusalDirectionResult,
        layer: int | None = None,
    ):
        """Initialize the attribution graph.

        Args:
            model: Language model
            refusal_direction: Pre-computed refusal direction
            layer: Layer to analyze (defaults to refusal_direction.layer)
        """
        self.model = model
        self.refusal_direction = refusal_direction
        self.layer = layer or refusal_direction.layer
        self.model.eval()

    def get_activations(
        self,
        prompt: str,
        layer: int | None = None,
    ) -> torch.Tensor:
        """Get layer activations for a prompt.

        Args:
            prompt: Input prompt
            layer: Layer to extract (defaults to self.layer)

        Returns:
            Activations tensor of shape (seq_len, d_model)
        """
        layer = layer or self.layer

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(config.DEVICE)

        activations = {}

        def create_hook(name: str) -> Any:
            def hook(_module: Any, _input: Any, output: Any) -> None:
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()

            return hook

        handles = []
        try:
            if hasattr(self.model, "model"):
                if hasattr(self.model.model, "layers"):
                    layers_module = self.model.model.layers
                else:
                    layers_module = self.model.model
            else:
                layers_module = self.model.layers

            handle = layers_module[layer].register_forward_hook(
                create_hook(f"layer_{layer}")
            )
            handles.append(handle)

            with torch.no_grad():
                _ = self.model(tokens)

            if f"layer_{layer}" not in activations:
                msg = f"Failed to capture activation at layer {layer}"
                raise ValueError(msg)

            return activations[f"layer_{layer}"][0]  # (seq_len, d_model)

        finally:
            for handle in handles:
                handle.remove()

    def compute_attributions(
        self,
        prompt: str,
        method: str = "dot_product",
    ) -> AttributionResult:
        """Compute attributions to the refusal direction.

        Args:
            prompt: Input prompt to analyze
            method: Attribution method ("dot_product" or "gradient")

        Returns:
            AttributionResult with computed attributions
        """
        # Get activations
        activations = self.get_activations(prompt)  # (seq_len, d_model)

        # Compute dot product with refusal direction
        refusal_dir = self.refusal_direction.direction.to(config.DEVICE)
        scores = activations @ refusal_dir  # (seq_len,)

        # Convert to list for tokens
        token_attributions = scores.cpu().tolist()

        # Compute feature-level attributions (per-dimension)
        # Sum of absolute attribution per dimension
        feature_attributions = torch.abs(activations.mean(0) @ refusal_dir)
        feature_scores = feature_attributions.cpu()

        # Get top features
        top_k = 50
        top_indices = torch.topk(feature_scores, min(top_k, len(feature_scores)))
        top_features = list(
            zip(top_indices.indices.tolist(), top_indices.values.tolist(), strict=False)
        )

        return AttributionResult(
            prompt=prompt,
            refusal_direction=self.refusal_direction.direction,
            layer=self.layer,
            token_attributions=token_attributions,
            top_features=top_features,
            attribution_scores=scores.cpu(),
        )

    def build_graph(
        self,
        prompt: str,
        threshold: float = 0.1,
    ) -> list[AttributionNode]:
        """Build attribution graph from prompt.

        Args:
            prompt: Input prompt
            threshold: Minimum attribution score to include

        Returns:
            List of AttributionNodes sorted by score
        """
        result = self.compute_attributions(prompt)
        nodes = []

        # Create nodes for tokens with high attribution
        for token_idx, score in enumerate(result.token_attributions):
            if abs(score) > threshold:
                nodes.append(
                    AttributionNode(
                        feature_idx=token_idx,
                        layer=self.layer,
                        attribution_score=score,
                        activation=result.attribution_scores[token_idx].item()
                        if result.attribution_scores is not None
                        else 0.0,
                    )
                )

        # Sort by absolute score
        nodes.sort(key=lambda n: abs(n.attribution_score), reverse=True)

        return nodes

    def analyze_prompt(
        self,
        prompt: str,
        return_graph: bool = True,
    ) -> AttributionResult | tuple[AttributionResult, list[AttributionNode]]:
        """Analyze a single prompt.

        Args:
            prompt: Input prompt
            return_graph: Whether to also return the graph

        Returns:
            AttributionResult, or (AttributionResult, graph) if return_graph=True
        """
        result = self.compute_attributions(prompt)

        if return_graph:
            graph = self.build_graph(prompt)
            return result, graph

        return result


def attribute_to_direction(
    model: torch.nn.Module,
    prompt: str,
    refusal_direction: RefusalDirectionResult,
    layer: int | None = None,
    method: str = "dot_product",
) -> AttributionResult:
    """Attribute a prompt's response to the refusal direction.

    This is the main entry point for Step 2 of the research plan.

    Args:
        model: Language model
        prompt: Input prompt to analyze
        refusal_direction: Pre-computed refusal direction
        layer: Layer to analyze (defaults to refusal_direction.layer)
        method: Attribution method

    Returns:
        AttributionResult with computed attributions
    """
    graph = AttributionGraph(model, refusal_direction, layer)
    return graph.compute_attributions(prompt, method)


def analyze_circuit(
    model: torch.nn.Module,
    refusal_direction: RefusalDirectionResult,
    prompts: list[str],
    layers: list[int] | None = None,
) -> dict[int, dict[str, AttributionResult]]:
    """Analyze attribution circuits across multiple layers.

    Args:
        model: Language model
        refusal_direction: Pre-computed refusal direction
        prompts: List of prompts to analyze
        layers: List of layers to analyze (defaults to [refusal_direction.layer])

    Returns:
        Dictionary mapping layer -> prompt -> AttributionResult
    """
    if layers is None:
        layers = [refusal_direction.layer]

    results = {}

    for layer in layers:
        graph = AttributionGraph(model, refusal_direction, layer)
        layer_results = {}

        for prompt in tqdm(prompts, desc=f"Analyzing layer {layer}"):
            layer_results[prompt] = graph.compute_attributions(prompt)

        results[layer] = layer_results

    return results


def save_attributions(
    results: dict[int, dict[str, AttributionResult]],
    output_dir: str,
) -> None:
    """Save attribution results to disk.

    Args:
        results: Dictionary of layer -> prompt -> AttributionResult
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    for layer, layer_results in results.items():
        layer_dir = os.path.join(output_dir, f"layer_{layer:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for prompt, result in layer_results.items():
            # Create safe filename from prompt
            safe_prompt = prompt[:50].replace("/", "_").replace(" ", "_")
            filepath = os.path.join(layer_dir, f"{safe_prompt}.json")

            data = {
                "prompt": result.prompt,
                "layer": result.layer,
                "token_attributions": result.token_attributions,
                "top_features": result.top_features,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    # Save summary
    summary = {
        str(layer): {
            "n_prompts": len(layer_results),
            "avg_attribution": np.mean(
                [np.mean(np.abs(r.token_attributions)) for r in layer_results.values()]
            ),
        }
        for layer, layer_results in results.items()
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved attribution results to {output_dir}")
