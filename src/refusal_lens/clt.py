"""
Circuit-tracer integration for attribution to the refusal direction.

Wraps Anthropic's circuit-tracer library to compute attribution graphs                                                                                                                                                                               
targeting the refusal projection R = ⟨x^(ℓ*,c*), r̂⟩.                                                                                                                                                                                                 
"""                                                                                                                                                                                                                                                  
from __future__ import annotations                                                                                                                                                                                                                   

from typing import Any
from . import config

HAS_CIRCUIT_TRACER = False
try:
    import torch
    from circuit_tracer import Graph, ReplacementModel, attribute
    from circuit_tracer.attribution.targets import CustomTarget
    from circuit_tracer.graph import prune_graph
    from .attribution import attribute_to_direction

    HAS_CIRCUIT_TRACER = True
except ImportError:
    pass


def _require_circuit_tracer() -> None:
    """Raise ImportError if circuit-tracer is not installed."""
    if not HAS_CIRCUIT_TRACER:
        raise ImportError(
            "circuit-tracer is required for attribution. "
            "Install via: pip install git+https://github.com/AutoInterp/circuit-tracer.git@refusal-lens-measurement-patch"
        )


# Default transcoder repo for Gemma 3 4B IT (circuit-tracer compatible)
DEFAULT_TRANSCODER_REPO: str = (
    "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine"
)

DEFAULT_BACKEND: str = "nnsight"

def load_replacement_model(
    model_name: str | None = None,
    transcoder_repo: str | None = None,
    *,
    backend: str | None = None,
    dtype: Any = None,
    lazy_encoder: bool = True,
) -> ReplacementModel:
    """Load a circuit-tracer ReplacementModel with transcoders.

    This wraps ``ReplacementModel.from_pretrained()`` with sensible defaults
    for Gemma 3 4B IT and the Gemma Scope 2 transcoders.

    Args:
        model_name: HuggingFace model ID. Defaults to ``config.MODEL_NAME``.
        transcoder_repo: HuggingFace path to transcoder weights.
            Defaults to ``DEFAULT_TRANSCODER_REPO``.
        backend: ``"nnsight"`` or ``"transformerlens"``.
            Defaults to ``DEFAULT_BACKEND``.
        dtype: Torch dtype. Defaults to ``torch.bfloat16``.
        lazy_encoder: If True, defer encoder loading to save memory.
            Defaults to True.

    Returns:
        A configured ReplacementModel ready for attribution.
    """
    _require_circuit_tracer()
    model_name = model_name or config.MODEL_NAME
    transcoder_repo = transcoder_repo or DEFAULT_TRANSCODER_REPO
    backend = backend or DEFAULT_BACKEND
    if dtype is None:
        dtype = torch.bfloat16

    return ReplacementModel.from_pretrained(
        model_name,
        transcoder_repo,
        backend=backend,
        dtype=dtype,
        lazy_encoder=lazy_encoder,
    )

def make_refusal_target(
    refusal_direction: torch.Tensor,
    *,
    label: str = "refusal_direction",
    weight: float = 1.0,
) -> CustomTarget:
    """Create a CustomTarget for refusal direction attribution.

    This builds the target that tells circuit-tracer to attribute toward
    the refusal direction r̂ instead of output logits.

    The refusal direction should be a unit vector of shape ``(d_model,)``,
    computed via ``compute_refusal_directions()`` from Step 4.

    Args:
        refusal_direction: The refusal direction r̂, shape ``(d_model,)``.
        label: Label for the target node in the attribution graph.
        weight: Probability weight for this target (default 1.0).

    Returns:
        A ``CustomTarget`` ready to pass to ``attribute()``.

    Raises:
        ValueError: If the direction has wrong shape.
    """
    _require_circuit_tracer()
    if refusal_direction.ndim != 1:
        raise ValueError(
            f"refusal_direction must be 1-D (d_model,), got shape {refusal_direction.shape}"
        )
    return CustomTarget(
        token_str=label,
        prob=weight,
        vec=refusal_direction,
    )

def attribute_to_refusal(
    prompt: str,
    model: ReplacementModel,
    refusal_direction: torch.Tensor,
    *,
    layer: int | None = None,
    position: int | None = None,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> Graph:
    """Compute an attribution graph targeting the refusal direction.

    This is the main entry point for the mentor's research vision:
    compute A_{s→R} for all active CLT features, where
    R = ⟨x^(ℓ*, c*), r̂⟩.

    When ``layer`` and ``position`` are both None, attribution targets the
    last layer at the last token position (the default circuit-tracer
    behavior).  Passing explicit values enables measurement at intermediate
    points — see the mentor's Section 3.4 discussion on choosing ℓ*.

    Args:
        prompt: The input text to attribute.
        model: A loaded ReplacementModel (from ``load_replacement_model``).
        refusal_direction: The refusal direction r̂, shape ``(d_model,)``.
        layer: Transformer layer at which to measure the refusal projection.
            ``None`` means the final layer (default circuit-tracer behavior).
        position: Token position at which to measure.
            ``None`` means the last token position.
        batch_size: Batch size for the attribution backward passes.
        max_feature_nodes: Maximum number of feature nodes to include.
            ``None`` means include all active features.
        verbose: Whether to print progress information.

    Returns:
        A ``Graph`` containing the full attribution to the refusal direction.
    """
    _require_circuit_tracer()
    return attribute_to_direction(
        prompt=prompt,
        model=model,
        direction=refusal_direction,
        measurement_layer=layer,
        measurement_position=position,
        label="refusal_direction",
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        verbose=verbose,
    )

def attribute_to_refusal_sweep(
    prompt: str,
    model: ReplacementModel,
    refusal_direction: torch.Tensor,
    *,
    layers: list[int] | None = None,
    position: int | None = None,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> dict[int, Graph]:
    """Run attribution at multiple measurement layers.

    This enables layer-sweep analysis: measure how A_{s→R} changes as the
    measurement point ℓ* moves through the network.  Useful for:

    - Finding the optimal measurement layer (where refusal signal is strongest)
    - Plotting feature attribution vs. layer (like the Arditi et al. heatmap)
    - Understanding at which layer RP features begin suppressing refusal

    Args:
        prompt: The input text to attribute.
        model: A loaded ReplacementModel.
        refusal_direction: The refusal direction r̂, shape ``(d_model,)``.
        layers: List of layer indices to sweep.
            Defaults to ``config.REFUSAL_LAYERS`` (range 8-23).
        position: Token position at which to measure.
            ``None`` means the last token position.
        batch_size: Batch size for attribution.
        max_feature_nodes: Maximum number of feature nodes per graph.
        verbose: Whether to print progress information.

    Returns:
        Dictionary mapping layer index → attribution Graph.
    """
    _require_circuit_tracer()
    if layers is None:
        layers = config.REFUSAL_LAYERS

    results: dict[int, Graph] = {}
    for layer_idx in layers:
        if verbose:
            print(f"Attributing at layer {layer_idx}...")
        results[layer_idx] = attribute_to_refusal(
            prompt=prompt,
            model=model,
            refusal_direction=refusal_direction,
            layer=layer_idx,
            position=position,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            verbose=verbose,
        )
    return results

def prune_refusal_graph(
    graph: Graph,
    *,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> tuple[Graph, Any]:
    """Prune an attribution graph to keep the most important nodes/edges.

    Args:
        graph: The full attribution graph from ``attribute_to_refusal()``.
        node_threshold: Fraction of total node influence to retain (0-1).
        edge_threshold: Fraction of total edge influence to retain (0-1).

    Returns:
        Tuple of ``(graph, prune_result)`` where prune_result contains
        ``node_mask``, ``edge_mask``, and ``cumulative_scores``.
    """
    _require_circuit_tracer()
    result = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    return graph, result

def extract_top_features(
    graph: Graph,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Extract the top-k features by absolute attribution to the refusal direction.

    This reads the adjacency matrix to find which CLT features have the
    largest direct effect on the refusal projection R.

    Args:
        graph: An attribution graph (pruned or unpruned).
        top_k: Number of top features to return.

    Returns:
        List of dicts with keys: ``layer``, ``feature_idx``, ``position``,
        ``attribution`` (the edge weight A_{s→R}), ``activation`` (the
        feature activation value a_s).
    """
    _require_circuit_tracer()
    # active_features has shape (n_active, 3) = [layer, position, feature_idx]
    active = graph.active_features  # (n_features, 3)
    n_features = active.shape[0]

    # The adjacency matrix rows are targets, columns are sources
    # The refusal target is the last node (logit_targets[-1])
    # Source feature nodes are indices 0..n_features-1
    adj = graph.adjacency_matrix
    target_row = adj[-1, :n_features]  # attribution from each feature to refusal target

    # Sort by absolute attribution
    abs_attr = target_row.abs()
    top_indices = abs_attr.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        feat = active[idx]
        results.append({
            "layer": int(feat[0]),
            "position": int(feat[1]),
            "feature_idx": int(feat[2]),
            "attribution": float(target_row[idx]),
            "activation": float(graph.activation_values[idx])
            if idx < len(graph.activation_values)
            else 0.0,
        })
    return results