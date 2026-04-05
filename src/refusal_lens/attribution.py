"""
Intermediate-layer attribution to arbitrary directions.

Provides ``attribute_to_direction()`` which combines circuit-tracer's ``attribute()``
with our vendored measurement-layer patch. This is the core function for computing 
A_{s->R} where R = ⟨x^(ℓ*, c*), r̂⟩ at any transformer layer ℓ*.
"""
from __future__ import annotations

from typing import Any
from . import config

HAS_CIRCUIT_TRACER = False
try:
    import torch
    from circuit_tracer import Graph, attribute
    from circuit_tracer.attribution.targets import CustomTarget

    HAS_CIRCUIT_TRACER = True
except ImportError:
    pass

def _require_circuit_tracer() -> None:
    """Raise ImportError if circuit-tracer is not installed"""
    if not HAS_CIRCUIT_TRACER:
        raise ImportError(
            "circuit-tracer is required for attribution. "
            "Install via: pip install git+https://github.com/AutoInterp/circuit-tracer.git@refusal-lens-measurement-patch"
        )

def validate_measurement_point(
    layer: int | None,
    position: int | None,
    *,
    n_layers: int | None = None,
    n_positions: int | None = None,
) -> None:
    """
    Validate that measurement layer/position are sensible.

    Args:
        layer: requested measurement-layer (0-idx)
        position: requested token position (0-idx)
        n_layers: Total number of model layers (for upper bound check).
        n_positions: Total number of token positions (for upper bound check).
    
    Raises:
        ValueError: If layer or position are out of bounds.
    """

    if layer is not None:
        if layer < 0:
            raise ValueError(f"measurement_layer must be >= 0, got {layer}")
        if n_layers is not None and layer >= n_layers:
            raise ValueError(
                f"measurement_layer {layer} >= n_layers {n_layers}. "
                "Use None for last-layer attribution."
            )
    if position is not None:
        if position < 0:
            raise ValueError(f"measurement_position must be >= 0, got {position}")
        if n_positions is not None and position >= n_positions:
            raise ValueError(
                f"measurement_position {position} >= n_positions {n_positions}."
            )
    
def attribute_to_direction(
    prompt: str,
    model: Any,
    direction: torch.Tensor,
    *,
    measurement_layer: int | None = None,
    measurement_position: int | None = None,
    label: str = "refusal_direction",
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> Graph:
    """
    Attribute to an arbitrary direction at an arbitrary measurement point.

    This is the core: compute A_{s->R} where R =  ⟨x^(ℓ*, c*), r̂⟩ with ℓ* at *any* transformer layer.

    When ``measurement_layer`` and ``measurement_position`` are both ``None``,
    this uses standard last-layer attribution.

    Args:
        prompt: Input text to attribute.
        model: A loaded ``ReplacementModel``.
        direction: Direction vector (e.g. refusal direction  r̂),
            shape ``(d_model,)``
        measurement_layer: Transformer layer at which to measure attribution.
            ``None`` means the post-transformer (unembed) layer (default).
        measurement_position: Token position at which to measure.
            ``None`` → last token position.
        label: Label for the target node in the graph.
        batch_size: Batch size for backward passes.
        max_feature_nodes: Max feature nodes (``None`` = all active).
        verbose: Print progress information.

    Returns:
        A ``Graph`` with attribution edges to the direction target.

    Raises:
        ValueError: If ``direction`` is not 1-D or measurement point
            is out of bounds.
        ImportError: If circuit-tracer is not installed.
    """
    _require_circuit_tracer()

    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D (d_model,), got shape {direction.shape}")
    
    validate_measurement_point(measurement_layer, measurement_position)
    target = CustomTarget(token_str=label, prob=1.0, vec=direction)

    # standard last-layer attribution\
    return attribute(
        prompt=prompt,
        model=model,
        attribution_targets=[target],
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        verbose=verbose,
        measurement_layer=measurement_layer,
        measurement_position=measurement_position
    )