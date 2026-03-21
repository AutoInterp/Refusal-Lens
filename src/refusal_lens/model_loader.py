"""
Handles model loading, tokenization, and generation utils.

Provides standardized model management for the package.
Requires optional deps: torch, transformers, accelerate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# this is optional, helps refrain from crashing if imported to machine without PyTorch
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch() -> None:
    if not HAS_TORCH:
        msg = (
            "torch and transformers are required for model loading. "
            "Install with: pip install torch transformers accelerate"
        )
        raise ImportError(msg)


@dataclass
class ModelConfig:
    """Config for model loading."""

    model_name: str = "google/gemma-3-4b-it"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    torch_dtype: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype not in dtype_map:
            msg = f"Unsupported dtype: {self.dtype}. Use one of {list(dtype_map)}"
            raise ValueError(msg)
        self.torch_dtype = dtype_map[self.dtype]


def load_model(
    config: ModelConfig | None = None,
) -> tuple[Any, Any]:
    """
    Load a model and tokenizer.

    Args:
        config: Model config dataclass. Uses the default (gemma) if None.

    Returns:
        Tuple of (model, tokenizer)
    """
    _require_torch()
    if config is None:
        config = ModelConfig()

    logger.info("Loading model: %s", config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.torch_dtype,
        device_map=config.device_map,
    )

    num_layers = model.config.text_config.num_hidden_layers
    hidden_dim = model.config.text_config.hidden_size
    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d",
        config.model_name,
        num_layers,
        hidden_dim,
    )

    return model, tokenizer


def get_device_info() -> dict[str, Any]:
    """
    Get info about the available compute device.

    Returns:
        Dict with device_name, type, and memory info.
    """
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info: dict[str, Any] = {"device": device, "type": device}

    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        mem_bytes = torch.cuda.get_device_properties(0).total_memory
        info["memory_gb"] = round(mem_bytes / 1e9, 1)
        info["low_vram"] = info["memory_gb"] < 30

    return info


def tokenize_prompt(
    text: str,
    tokenizer: Any,
    device: str | None = None,
) -> Any:
    """
    Tokenize a prompt using the model's chat template

    Args:
        text: The user message text.
        tokenizer: The tokenizer to use.
        device: Device to put tensors on. Auto-detects if None.

    Returns:
        Input IDs tensor with shape (1, seq_len)
    """

    _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    input_ids = tokenizer.appy_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generate_prompt=True,
    ).to(device)
    return input_ids


def generate_text(
    model: Any,
    input_ids: Any,
    tokenizer: Any,
    max_new_tokens: int = 100,
) -> str:
    """
    Generate text from input IDs using greedy decoding.

    Args:
        model: language model
        input_ids: Tokenized input IDs
        tokenizer: tokenizer for decoding
        max_new_tokens: Maximum tokens to generate

    Returns:
        The generated text (excluding the prompt but this can be changed if needed)
    """
    _require_torch()
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)


def display_topk(
    prompt: str,
    orig_logits: Any,
    new_logits: Any,
    tokenizer: Any,
    k: int = 10,
) -> list[dict[str, Any]]:
    """
    Compare top-k token predictions between original and intervened logits.

    Args:
        prompt: The input prompt (for display).
        orig_logits: Original model logits.
        new_logits: Post-intervention logits.
        tokenizer: Tokenizer for decoding token IDs.
        k: Number of top tokens to compare.

    Returns:
        List of dicts with original and intervened top-k comparisons.
    """
    _require_torch()
    orig_probs = F.softmax(orig_logits[0, -1].float(), dim=-1)
    new_probs = F.softmax(new_logits[0, -1].float(), dim=-1)
    orig_topk = torch.topk(orig_probs, k)
    new_topk = torch.topk(new_probs, k)

    comparisons = []
    for i in range(k):
        comparisons.append(
            {
                "orig_token": tokenizer.decode([orig_topk.indices[i]]),
                "orig_prob": orig_topk.values[i].item(),
                "new_token": tokenizer.decode([new_topk.indices[i]]),
                "new_prob": new_topk.values[i].item(),
            }
        )
    return comparisons
