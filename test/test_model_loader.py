"""
Tests for the model_loader module.

Two test tiers:
  1. Mock-based tests (always run) -- validate logic without real packages.
  2. Real-package tests (skip if torch missing) -- validate with actual tensors.

Note: We use create=True on patch() calls for torch, F, AutoModelForCausalLM,
and AutoTokenizer because these attributes only exist on the module when
torch is installed. create=True lets patch inject them even when absent.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Detect if real torch is available for the real-package test tier
try:
    import torch
    import torch.nn.functional as _F  # noqa: N812

    _HAS_REAL_TORCH = True
except ImportError:
    _HAS_REAL_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_REAL_TORCH, reason="torch not installed"
)

# Shorthand: all patches for torch-dependent attributes need create=True
# because when torch is not installed, these names don't exist on the module.
_ML = "refusal_lens.model_loader"


def _torch_patches(mock_torch, mock_F=None, mock_auto_model=None, mock_auto_tok=None):
    """
    Build a list of patch context managers for torch-dependent attributes.

    Returns a list of context managers to be used with nested `with` or
    via the helper _apply_patches.
    """
    patches = [
        patch(f"{_ML}.HAS_TORCH", True),
        patch(f"{_ML}.torch", mock_torch, create=True),
    ]
    if mock_F is not None:
        patches.append(patch(f"{_ML}.F", mock_F, create=True))
    if mock_auto_model is not None:
        patches.append(
            patch(f"{_ML}.AutoModelForCausalLM", mock_auto_model, create=True)
        )
    if mock_auto_tok is not None:
        patches.append(
            patch(f"{_ML}.AutoTokenizer", mock_auto_tok, create=True)
        )
    return patches


class _PatchContext:
    """Context manager that applies a list of patches."""

    def __init__(self, patches):
        self._patches = patches
        self._active = []

    def __enter__(self):
        for p in self._patches:
            self._active.append(p.__enter__())
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        self._active.clear()
        return False


def apply_patches(*patches):
    """Combine multiple patch context managers into one."""
    return _PatchContext(list(patches))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_torch():
    """Create a mock torch module with realistic tensor behavior."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.bfloat16 = "bfloat16_dtype"
    mock.float16 = "float16_dtype"
    mock.float32 = "float32_dtype"
    mock.no_grad.return_value.__enter__ = MagicMock()
    mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return mock


@pytest.fixture()
def mock_tokenizer():
    """Create a mock tokenizer that mimics HuggingFace behavior."""
    tok = MagicMock()
    template_result = MagicMock()
    template_result.to.return_value = template_result
    template_result.shape = (1, 10)
    tok.apply_chat_template.return_value = template_result
    tok.decode.return_value = "generated response text"
    return tok


@pytest.fixture()
def mock_model():
    """Create a mock model that mimics HuggingFace causal LM behavior."""
    model = MagicMock()
    model.config.text_config.num_hidden_layers = 26
    model.config.text_config.hidden_size = 2304
    output_ids = MagicMock()
    output_ids.__getitem__ = MagicMock(return_value="decoded_slice")
    model.generate.return_value = output_ids
    return model


# ---------------------------------------------------------------------------
# Test: Guarded import pattern (no torch available)
# ---------------------------------------------------------------------------


class TestGuardedImport:
    """Test that the module handles missing torch gracefully."""

    def test_module_imports_without_torch(self):
        """Package should import cleanly even without torch installed."""
        from refusal_lens import model_loader

        assert hasattr(model_loader, "HAS_TORCH")

    def test_has_torch_flag_reflects_availability(self):
        """HAS_TORCH should be False when torch is not installed."""
        from refusal_lens import model_loader

        if "torch" not in sys.modules or sys.modules.get("torch") is None:
            assert model_loader.HAS_TORCH is False

    def test_public_api_accessible(self):
        """All public names should be importable from the package."""
        from refusal_lens import (
            ModelConfig,
            display_topk,
            generate_text,
            get_device_info,
            load_model,
            tokenize_prompt,
        )

        assert callable(load_model)
        assert callable(get_device_info)
        assert callable(tokenize_prompt)
        assert callable(generate_text)
        assert callable(display_topk)
        assert isinstance(ModelConfig, type)


# ---------------------------------------------------------------------------
# Test: _require_torch gate
# ---------------------------------------------------------------------------


class TestRequireTorch:
    """Test that _require_torch raises ImportError when torch is missing."""

    def test_require_torch_raises_without_torch(self):
        from refusal_lens.model_loader import _require_torch

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError, match="torch and transformers"):
                _require_torch()

    def test_require_torch_passes_with_torch(self):
        from refusal_lens.model_loader import _require_torch

        with patch(f"{_ML}.HAS_TORCH", True):
            _require_torch()


# ---------------------------------------------------------------------------
# Test: ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Test ModelConfig dataclass behavior."""

    def test_default_values(self, mock_torch):
        from refusal_lens.model_loader import ModelConfig

        with apply_patches(*_torch_patches(mock_torch)):
            config = ModelConfig()
            assert config.model_name == "google/gemma-3-4b-it"
            assert config.dtype == "bfloat16"
            assert config.device_map == "auto"
            assert config.torch_dtype == "bfloat16_dtype"

    def test_custom_model_name(self, mock_torch):
        from refusal_lens.model_loader import ModelConfig

        with apply_patches(*_torch_patches(mock_torch)):
            config = ModelConfig(model_name="meta-llama/Llama-3-8B")
            assert config.model_name == "meta-llama/Llama-3-8B"

    def test_float16_dtype(self, mock_torch):
        from refusal_lens.model_loader import ModelConfig

        with apply_patches(*_torch_patches(mock_torch)):
            config = ModelConfig(dtype="float16")
            assert config.torch_dtype == "float16_dtype"

    def test_float32_dtype(self, mock_torch):
        from refusal_lens.model_loader import ModelConfig

        with apply_patches(*_torch_patches(mock_torch)):
            config = ModelConfig(dtype="float32")
            assert config.torch_dtype == "float32_dtype"

    def test_invalid_dtype_raises(self, mock_torch):
        from refusal_lens.model_loader import ModelConfig

        with apply_patches(*_torch_patches(mock_torch)):
            with pytest.raises(ValueError, match="Unsupported dtype"):
                ModelConfig(dtype="int8")

    def test_requires_torch(self):
        from refusal_lens.model_loader import ModelConfig

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                ModelConfig()


# ---------------------------------------------------------------------------
# Test: load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Test load_model function with mocked transformers."""

    def test_load_model_calls_from_pretrained(self, mock_torch):
        from refusal_lens.model_loader import ModelConfig, load_model

        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()
        mock_returned_model = MagicMock()
        mock_returned_model.config.text_config.num_hidden_layers = 26
        mock_returned_model.config.text_config.hidden_size = 2304
        mock_auto_model.from_pretrained.return_value = mock_returned_model
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

        with apply_patches(
            *_torch_patches(
                mock_torch,
                mock_auto_model=mock_auto_model,
                mock_auto_tok=mock_auto_tokenizer,
            )
        ):
            config = ModelConfig(model_name="test/model")
            model, tokenizer = load_model(config)

            mock_auto_model.from_pretrained.assert_called_once_with(
                "test/model",
                dtype=config.torch_dtype,
                device_map="auto",
            )
            mock_auto_tokenizer.from_pretrained.assert_called_once_with(
                "test/model"
            )
            assert model is mock_returned_model

    def test_load_model_default_config(self, mock_torch):
        from refusal_lens.model_loader import load_model

        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()
        mock_returned_model = MagicMock()
        mock_returned_model.config.text_config.num_hidden_layers = 26
        mock_returned_model.config.text_config.hidden_size = 2304
        mock_auto_model.from_pretrained.return_value = mock_returned_model

        with apply_patches(
            *_torch_patches(
                mock_torch,
                mock_auto_model=mock_auto_model,
                mock_auto_tok=mock_auto_tokenizer,
            )
        ):
            load_model()
            call_args = mock_auto_model.from_pretrained.call_args
            assert call_args[0][0] == "google/gemma-3-4b-it"

    def test_load_model_returns_tuple(self, mock_torch):
        from refusal_lens.model_loader import load_model

        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()
        mock_returned_model = MagicMock()
        mock_returned_model.config.text_config.num_hidden_layers = 26
        mock_returned_model.config.text_config.hidden_size = 2304
        mock_auto_model.from_pretrained.return_value = mock_returned_model
        mock_returned_tok = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_returned_tok

        with apply_patches(
            *_torch_patches(
                mock_torch,
                mock_auto_model=mock_auto_model,
                mock_auto_tok=mock_auto_tokenizer,
            )
        ):
            result = load_model()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] is mock_returned_model
            assert result[1] is mock_returned_tok

    def test_load_model_requires_torch(self):
        from refusal_lens.model_loader import load_model

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                load_model()


# ---------------------------------------------------------------------------
# Test: get_device_info
# ---------------------------------------------------------------------------


class TestGetDeviceInfo:
    """Test get_device_info with mocked torch."""

    def test_cpu_device(self, mock_torch):
        from refusal_lens.model_loader import get_device_info

        mock_torch.cuda.is_available.return_value = False

        with apply_patches(*_torch_patches(mock_torch)):
            info = get_device_info()
            assert info["device"] == "cpu"
            assert info["type"] == "cpu"
            assert "gpu_name" not in info
            assert "memory_gb" not in info

    def test_cuda_device(self, mock_torch):
        from refusal_lens.model_loader import get_device_info

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_props = SimpleNamespace(total_memory=40_000_000_000)
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with apply_patches(*_torch_patches(mock_torch)):
            info = get_device_info()
            assert info["device"] == "cuda"
            assert info["gpu_name"] == "NVIDIA A100"
            assert info["memory_gb"] == 40.0
            assert info["low_vram"] is False

    def test_low_vram_flag(self, mock_torch):
        from refusal_lens.model_loader import get_device_info

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA T4"
        mock_props = SimpleNamespace(total_memory=16_000_000_000)
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with apply_patches(*_torch_patches(mock_torch)):
            info = get_device_info()
            assert info["low_vram"] is True
            assert info["memory_gb"] == 16.0

    def test_requires_torch(self):
        from refusal_lens.model_loader import get_device_info

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                get_device_info()


# ---------------------------------------------------------------------------
# Test: tokenize_prompt
# ---------------------------------------------------------------------------


class TestTokenizePrompt:
    """Test tokenize_prompt with mocked tokenizer."""

    def test_builds_chat_messages(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import tokenize_prompt

        mock_torch.cuda.is_available.return_value = False

        with apply_patches(*_torch_patches(mock_torch)):
            tokenize_prompt("Hello world", mock_tokenizer)

            mock_tokenizer.apply_chat_template.assert_called_once()
            call_args = mock_tokenizer.apply_chat_template.call_args
            messages = call_args[0][0]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == [
                {"type": "text", "text": "Hello world"}
            ]

    def test_passes_generation_prompt_flag(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import tokenize_prompt

        mock_torch.cuda.is_available.return_value = False

        with apply_patches(*_torch_patches(mock_torch)):
            tokenize_prompt("test", mock_tokenizer)

            call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
            assert call_kwargs["tokenize"] is True
            assert call_kwargs["return_tensors"] == "pt"
            assert call_kwargs["add_generation_prompt"] is True

    def test_moves_to_device(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import tokenize_prompt

        with apply_patches(*_torch_patches(mock_torch)):
            tokenize_prompt("test", mock_tokenizer, device="cpu")

            template_result = mock_tokenizer.apply_chat_template.return_value
            template_result.to.assert_called_once_with("cpu")

    def test_explicit_device_skips_auto_detect(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import tokenize_prompt

        with apply_patches(*_torch_patches(mock_torch)):
            tokenize_prompt("test", mock_tokenizer, device="cuda:1")

            template_result = mock_tokenizer.apply_chat_template.return_value
            template_result.to.assert_called_once_with("cuda:1")
            mock_torch.cuda.is_available.assert_not_called()

    def test_auto_detects_cpu(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import tokenize_prompt

        mock_torch.cuda.is_available.return_value = False

        with apply_patches(*_torch_patches(mock_torch)):
            tokenize_prompt("test", mock_tokenizer)

            template_result = mock_tokenizer.apply_chat_template.return_value
            template_result.to.assert_called_once_with("cpu")

    def test_requires_torch(self, mock_tokenizer):
        from refusal_lens.model_loader import tokenize_prompt

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                tokenize_prompt("test", mock_tokenizer)


# ---------------------------------------------------------------------------
# Test: generate_text
# ---------------------------------------------------------------------------


class TestGenerateText:
    """Test generate_text with mocked model."""

    def test_calls_model_generate(self, mock_torch, mock_model, mock_tokenizer):
        from refusal_lens.model_loader import generate_text

        input_ids = MagicMock()
        input_ids.shape = (1, 5)

        with apply_patches(*_torch_patches(mock_torch)):
            generate_text(mock_model, input_ids, mock_tokenizer)

            mock_model.generate.assert_called_once_with(
                input_ids, max_new_tokens=100, do_sample=False
            )

    def test_custom_max_tokens(self, mock_torch, mock_model, mock_tokenizer):
        from refusal_lens.model_loader import generate_text

        input_ids = MagicMock()
        input_ids.shape = (1, 5)

        with apply_patches(*_torch_patches(mock_torch)):
            generate_text(
                mock_model, input_ids, mock_tokenizer, max_new_tokens=50
            )

            call_kwargs = mock_model.generate.call_args[1]
            assert call_kwargs["max_new_tokens"] == 50

    def test_decodes_only_generated_tokens(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        from refusal_lens.model_loader import generate_text

        input_ids = MagicMock()
        input_ids.shape = (1, 5)

        with apply_patches(*_torch_patches(mock_torch)):
            generate_text(mock_model, input_ids, mock_tokenizer)

            mock_tokenizer.decode.assert_called_once()
            call_kwargs = mock_tokenizer.decode.call_args[1]
            assert call_kwargs["skip_special_tokens"] is True

    def test_returns_string(self, mock_torch, mock_model, mock_tokenizer):
        from refusal_lens.model_loader import generate_text

        input_ids = MagicMock()
        input_ids.shape = (1, 5)
        mock_tokenizer.decode.return_value = "Hello there"

        with apply_patches(*_torch_patches(mock_torch)):
            result = generate_text(mock_model, input_ids, mock_tokenizer)
            assert result == "Hello there"
            assert isinstance(result, str)

    def test_uses_no_grad_context(self, mock_torch, mock_model, mock_tokenizer):
        from refusal_lens.model_loader import generate_text

        input_ids = MagicMock()
        input_ids.shape = (1, 5)

        with apply_patches(*_torch_patches(mock_torch)):
            generate_text(mock_model, input_ids, mock_tokenizer)
            mock_torch.no_grad.assert_called_once()

    def test_requires_torch(self, mock_model, mock_tokenizer):
        from refusal_lens.model_loader import generate_text

        input_ids = MagicMock()
        input_ids.shape = (1, 5)

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                generate_text(mock_model, input_ids, mock_tokenizer)


# ---------------------------------------------------------------------------
# Test: display_topk
# ---------------------------------------------------------------------------


class TestDisplayTopk:
    """Test display_topk with mocked tensors."""

    def test_returns_list_of_dicts(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import display_topk

        mock_F = MagicMock()
        mock_F.softmax.return_value = MagicMock()

        topk_result = SimpleNamespace(
            indices=[0, 1, 2],
            values=[
                SimpleNamespace(item=lambda: 0.5),
                SimpleNamespace(item=lambda: 0.3),
                SimpleNamespace(item=lambda: 0.2),
            ],
        )
        mock_torch.topk.return_value = topk_result

        orig_logits = MagicMock()
        orig_logits.__getitem__ = MagicMock(
            return_value=MagicMock(float=MagicMock(return_value=MagicMock()))
        )
        new_logits = MagicMock()
        new_logits.__getitem__ = MagicMock(
            return_value=MagicMock(float=MagicMock(return_value=MagicMock()))
        )

        mock_tokenizer.decode.return_value = "token"

        with apply_patches(*_torch_patches(mock_torch, mock_F=mock_F)):
            result = display_topk(orig_logits, new_logits, mock_tokenizer, k=3)

            assert isinstance(result, list)
            assert len(result) == 3
            for entry in result:
                assert "orig_token" in entry
                assert "orig_prob" in entry
                assert "new_token" in entry
                assert "new_prob" in entry

    def test_calls_softmax_on_last_token(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import display_topk

        mock_F = MagicMock()
        mock_F.softmax.return_value = MagicMock()

        topk_result = SimpleNamespace(
            indices=[0],
            values=[SimpleNamespace(item=lambda: 0.9)],
        )
        mock_torch.topk.return_value = topk_result

        orig_logits = MagicMock()
        float_result = MagicMock()
        orig_logits.__getitem__ = MagicMock(
            return_value=MagicMock(float=MagicMock(return_value=float_result))
        )
        new_logits = MagicMock()
        new_logits.__getitem__ = MagicMock(
            return_value=MagicMock(float=MagicMock(return_value=MagicMock()))
        )

        with apply_patches(*_torch_patches(mock_torch, mock_F=mock_F)):
            display_topk(orig_logits, new_logits, mock_tokenizer, k=1)

            # Softmax called twice (once for orig, once for new)
            assert mock_F.softmax.call_count == 2
            # First call should use dim=-1
            first_call_kwargs = mock_F.softmax.call_args_list[0][1]
            assert first_call_kwargs["dim"] == -1

    def test_decodes_token_ids(self, mock_torch, mock_tokenizer):
        from refusal_lens.model_loader import display_topk

        mock_F = MagicMock()
        mock_F.softmax.return_value = MagicMock()

        topk_result = SimpleNamespace(
            indices=[42, 7],
            values=[
                SimpleNamespace(item=lambda: 0.6),
                SimpleNamespace(item=lambda: 0.4),
            ],
        )
        mock_torch.topk.return_value = topk_result

        orig_logits = MagicMock()
        orig_logits.__getitem__ = MagicMock(
            return_value=MagicMock(float=MagicMock(return_value=MagicMock()))
        )
        new_logits = MagicMock()
        new_logits.__getitem__ = MagicMock(
            return_value=MagicMock(float=MagicMock(return_value=MagicMock()))
        )

        mock_tokenizer.decode.side_effect = lambda ids: f"tok_{ids[0]}"

        with apply_patches(*_torch_patches(mock_torch, mock_F=mock_F)):
            result = display_topk(orig_logits, new_logits, mock_tokenizer, k=2)

            # Each entry in k=2 decodes orig + new = 4 decode calls total
            assert mock_tokenizer.decode.call_count == 4
            assert result[0]["orig_token"] == "tok_42"
            assert result[0]["new_token"] == "tok_42"

    def test_requires_torch(self, mock_tokenizer):
        from refusal_lens.model_loader import display_topk

        with patch(f"{_ML}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                display_topk(MagicMock(), MagicMock(), mock_tokenizer, k=1)


# ---------------------------------------------------------------------------
# Test: __init__.py exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Test that __init__.py correctly exports model_loader names."""

    def test_all_contains_model_loader_names(self):
        import refusal_lens

        expected = {
            "ModelConfig",
            "load_model",
            "get_device_info",
            "tokenize_prompt",
            "generate_text",
            "display_topk",
        }
        actual = set(refusal_lens.__all__)
        assert expected.issubset(actual), (
            f"Missing from __all__: {expected - actual}"
        )

    def test_model_loader_names_importable(self):
        """All model_loader names should be importable from top-level package."""
        from refusal_lens import (
            ModelConfig,
            display_topk,
            generate_text,
            get_device_info,
            load_model,
            tokenize_prompt,
        )

        assert ModelConfig.__module__ == "refusal_lens.model_loader"
        assert load_model.__module__ == "refusal_lens.model_loader"
        assert get_device_info.__module__ == "refusal_lens.model_loader"
        assert tokenize_prompt.__module__ == "refusal_lens.model_loader"
        assert generate_text.__module__ == "refusal_lens.model_loader"
        assert display_topk.__module__ == "refusal_lens.model_loader"


# ---------------------------------------------------------------------------
# Test: Integration between functions (mocked)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test that model_loader functions compose correctly."""

    def test_tokenize_then_generate_flow(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        """Simulate the tokenize -> generate pipeline from the notebook."""
        from refusal_lens.model_loader import generate_text, tokenize_prompt

        ids_mock = MagicMock()
        ids_mock.to.return_value = ids_mock
        ids_mock.shape = (1, 8)
        mock_tokenizer.apply_chat_template.return_value = ids_mock

        with apply_patches(*_torch_patches(mock_torch)):
            input_ids = tokenize_prompt("How do I code?", mock_tokenizer)
            assert input_ids.shape == (1, 8)

            result = generate_text(mock_model, input_ids, mock_tokenizer)
            assert isinstance(result, str)

    def test_config_flows_to_load(self, mock_torch):
        """ModelConfig values should propagate correctly to load_model."""
        from refusal_lens.model_loader import ModelConfig, load_model

        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()
        returned_model = MagicMock()
        returned_model.config.text_config.num_hidden_layers = 32
        returned_model.config.text_config.hidden_size = 4096
        mock_auto_model.from_pretrained.return_value = returned_model

        with apply_patches(
            *_torch_patches(
                mock_torch,
                mock_auto_model=mock_auto_model,
                mock_auto_tok=mock_auto_tokenizer,
            )
        ):
            config = ModelConfig(
                model_name="custom/model",
                dtype="float16",
                device_map="cpu",
            )
            load_model(config)

            mock_auto_model.from_pretrained.assert_called_once_with(
                "custom/model",
                dtype="float16_dtype",
                device_map="cpu",
            )


# ===========================================================================
# TIER 2: Real-package tests (skipped if torch is not installed)
# ===========================================================================


@requires_torch
class TestModelConfigRealTorch:
    """Test ModelConfig with real torch dtypes."""

    def test_default_dtype_is_bfloat16(self):
        from refusal_lens.model_loader import ModelConfig

        config = ModelConfig()
        assert config.torch_dtype is torch.bfloat16

    def test_float16_dtype(self):
        from refusal_lens.model_loader import ModelConfig

        config = ModelConfig(dtype="float16")
        assert config.torch_dtype is torch.float16

    def test_float32_dtype(self):
        from refusal_lens.model_loader import ModelConfig

        config = ModelConfig(dtype="float32")
        assert config.torch_dtype is torch.float32

    def test_invalid_dtype_raises(self):
        from refusal_lens.model_loader import ModelConfig

        with pytest.raises(ValueError, match="Unsupported dtype"):
            ModelConfig(dtype="int8")

    def test_torch_dtype_not_in_init(self):
        """torch_dtype should be computed, not a constructor arg."""
        from refusal_lens.model_loader import ModelConfig

        with pytest.raises(TypeError):
            ModelConfig(torch_dtype=torch.float32)  # type: ignore[call-arg]


@requires_torch
class TestGetDeviceInfoRealTorch:
    """Test get_device_info with the real torch runtime."""

    def test_returns_dict(self):
        from refusal_lens.model_loader import get_device_info

        info = get_device_info()
        assert isinstance(info, dict)
        assert "device" in info
        assert "type" in info

    def test_device_is_cpu_or_cuda(self):
        from refusal_lens.model_loader import get_device_info

        info = get_device_info()
        assert info["device"] in ("cpu", "cuda")

    def test_cuda_has_gpu_info(self):
        from refusal_lens.model_loader import get_device_info

        info = get_device_info()
        if info["device"] == "cuda":
            assert "gpu_name" in info
            assert isinstance(info["gpu_name"], str)
            assert "memory_gb" in info
            assert isinstance(info["memory_gb"], float)
            assert info["memory_gb"] > 0
            assert "low_vram" in info
            assert isinstance(info["low_vram"], bool)


@requires_torch
class TestHasTorchFlag:
    """Verify HAS_TORCH is True when torch is installed."""

    def test_has_torch_is_true(self):
        from refusal_lens.model_loader import HAS_TORCH

        assert HAS_TORCH is True

    def test_require_torch_does_not_raise(self):
        from refusal_lens.model_loader import _require_torch

        _require_torch()  # should not raise


@requires_torch
class TestDisplayTopkRealTensors:
    """Test display_topk with real torch tensors."""

    def test_basic_comparison(self):
        from refusal_lens.model_loader import display_topk

        vocab_size = 100
        # Create logits: shape (1, seq_len, vocab_size)
        orig_logits = torch.randn(1, 5, vocab_size)
        new_logits = torch.randn(1, 5, vocab_size)

        # Use a mock tokenizer that returns the token ID as a string
        tok = MagicMock()
        tok.decode.side_effect = lambda ids: f"<{ids[0]}>"

        result = display_topk(orig_logits, new_logits, tok, k=5)

        assert isinstance(result, list)
        assert len(result) == 5
        for entry in result:
            assert isinstance(entry["orig_prob"], float)
            assert isinstance(entry["new_prob"], float)
            assert 0.0 <= entry["orig_prob"] <= 1.0
            assert 0.0 <= entry["new_prob"] <= 1.0
            assert isinstance(entry["orig_token"], str)
            assert isinstance(entry["new_token"], str)

    def test_probabilities_sum_roughly_to_one(self):
        """Top-k probs from a small vocab should sum close to 1.0."""
        from refusal_lens.model_loader import display_topk

        vocab_size = 10
        orig_logits = torch.randn(1, 1, vocab_size)
        new_logits = torch.randn(1, 1, vocab_size)

        tok = MagicMock()
        tok.decode.side_effect = lambda ids: str(ids[0])

        # k = vocab_size means we get all probs
        result = display_topk(orig_logits, new_logits, tok, k=vocab_size)

        orig_total = sum(e["orig_prob"] for e in result)
        new_total = sum(e["new_prob"] for e in result)
        assert abs(orig_total - 1.0) < 1e-5
        assert abs(new_total - 1.0) < 1e-5

    def test_results_sorted_by_probability(self):
        """Top-k should return probabilities in descending order."""
        from refusal_lens.model_loader import display_topk

        orig_logits = torch.randn(1, 3, 50)
        new_logits = torch.randn(1, 3, 50)

        tok = MagicMock()
        tok.decode.side_effect = lambda ids: str(ids[0])

        result = display_topk(orig_logits, new_logits, tok, k=10)

        orig_probs = [e["orig_prob"] for e in result]
        new_probs = [e["new_prob"] for e in result]
        assert orig_probs == sorted(orig_probs, reverse=True)
        assert new_probs == sorted(new_probs, reverse=True)

    def test_k_equals_one(self):
        from refusal_lens.model_loader import display_topk

        orig_logits = torch.randn(1, 2, 30)
        new_logits = torch.randn(1, 2, 30)

        tok = MagicMock()
        tok.decode.return_value = "top"

        result = display_topk(orig_logits, new_logits, tok, k=1)
        assert len(result) == 1

    def test_different_logits_produce_different_results(self):
        """Distinct logit distributions should yield different top tokens."""
        from refusal_lens.model_loader import display_topk

        vocab_size = 20
        # Make orig heavily favor token 0, new heavily favor token 5
        orig_logits = torch.full((1, 1, vocab_size), -10.0)
        orig_logits[0, 0, 0] = 10.0

        new_logits = torch.full((1, 1, vocab_size), -10.0)
        new_logits[0, 0, 5] = 10.0

        decoded = {}
        tok = MagicMock()
        tok.decode.side_effect = lambda ids: f"tok{ids[0]}"

        result = display_topk(orig_logits, new_logits, tok, k=3)
        assert result[0]["orig_token"] == "tok0"
        assert result[0]["new_token"] == "tok5"
        assert result[0]["orig_prob"] > 0.99
        assert result[0]["new_prob"] > 0.99


@requires_torch
class TestGenerateTextRealTensors:
    """Test generate_text with real tensors and mocked model."""

    def test_slices_prompt_from_output(self):
        """generate_text should only decode tokens after the prompt."""
        from refusal_lens.model_loader import generate_text

        prompt_len = 5
        gen_len = 3
        input_ids = torch.randint(0, 100, (1, prompt_len))

        # Mock model.generate to return prompt + generated tokens
        output_tensor = torch.randint(0, 100, (1, prompt_len + gen_len))
        mock_model = MagicMock()
        mock_model.generate.return_value = output_tensor

        tok = MagicMock()
        tok.decode.return_value = "hello world"

        result = generate_text(mock_model, input_ids, tok, max_new_tokens=gen_len)

        # decode was called with only the generated slice
        call_args = tok.decode.call_args[0][0]
        assert call_args.shape == (gen_len,)
        assert result == "hello world"

    def test_greedy_decode_kwargs(self):
        from refusal_lens.model_loader import generate_text

        input_ids = torch.randint(0, 100, (1, 3))
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.randint(0, 100, (1, 6))
        tok = MagicMock()
        tok.decode.return_value = ""

        generate_text(mock_model, input_ids, tok, max_new_tokens=200)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert call_kwargs["max_new_tokens"] == 200


@requires_torch
class TestTokenizePromptRealTorch:
    """Test tokenize_prompt's device handling with real torch."""

    def test_returns_tensor_on_cpu(self):
        """With a mock tokenizer returning real tensors, verify device placement."""
        from refusal_lens.model_loader import tokenize_prompt

        fake_ids = torch.tensor([[1, 2, 3, 4]])
        tok = MagicMock()
        tok.apply_chat_template.return_value = fake_ids

        result = tokenize_prompt("hello", tok, device="cpu")

        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.shape == (1, 4)
