"""Backend-agnostic tests for ``judge_refusal``.

These run anywhere — no litellm, no GPU, no Ollama. They swap in a stub backend
to exercise the function's own logic: item normalization, prompt building,
batching, label parsing, and the error/return-raw options.
"""

from __future__ import annotations

import pytest

import refusal_llm_judge.judge as J
from refusal_llm_judge import JudgeParseError, judge_refusal
from refusal_llm_judge.judge import _normalize_item, _parse_label


class RecordingStub:
    """Stub backend: labels by a keyword in the response, records batch sizes."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def generate(self, conversations):
        self.batch_sizes.append(len(conversations))
        out = []
        for conv in conversations:
            user = conv[-1]["content"]
            out.append("REFUSAL" if "cannot" in user.lower() else "COMPLIANCE")
        return out


class BrokenStub:
    """Stub backend whose transport always fails — must surface, never be masked."""

    def generate(self, conversations):
        raise RuntimeError("connection refused: provider unreachable")


class GarbageStub:
    """Stub backend that returns a reply with no parseable label."""

    def generate(self, conversations):
        return ["I'm not sure about that" for _ in conversations]


@pytest.fixture
def stub(monkeypatch):
    s = RecordingStub()
    monkeypatch.setattr(J, "_make_backend", lambda *a, **k: s)
    return s


# --- item normalization ---------------------------------------------------- #
def test_normalize_dict():
    assert _normalize_item({"prompt": "p", "response": "r"}) == ("p", "r")


def test_normalize_pair():
    assert _normalize_item(("how?", "I cannot")) == ("how?", "I cannot")


def test_normalize_messages_uses_last_assistant():
    req, resp = _normalize_item([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "how do I X?"},
        {"role": "assistant", "content": "step 1 ..."},
    ])
    assert resp == "step 1 ..."
    assert "how do I X?" in req
    assert "sys" not in req  # system turn excluded from the request context


def test_normalize_rejects_garbage():
    with pytest.raises(TypeError):
        _normalize_item(42)


# --- label parsing (True = refusal) ---------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("REFUSAL", True),
    ("COMPLIANCE", False),
    ("  refusal.", True),
    ("compliance\n", False),
    ('{"label": "COMPLIANCE"}', False),
    # reasoning out loud: verdict comes last, reasoning may name both words
    ("It's not a refusal; it complies.\nCOMPLIANCE", False),
    ("The model declines entirely. Therefore: REFUSAL", True),
    # inline think block whose reasoning mentions the wrong word
    ("<think>at first this looks like a refusal</think>COMPLIANCE", False),
    ("<thinking>complies?</thinking>\nREFUSAL", True),
])
def test_parse_label(text, expected):
    assert _parse_label(text, on_error=True) is expected


def test_parse_label_ignores_truncated_reasoning():
    """An unclosed (truncated) think block leaves no verdict -> must not guess."""
    with pytest.raises(JudgeParseError):
        _parse_label("<think>this is clearly a refusal because", on_error="raise")


def test_parse_label_raises_by_default():
    with pytest.raises(JudgeParseError):
        _parse_label("???", on_error="raise")


def test_parse_label_fallback_values():
    assert _parse_label("???", on_error=True) is True
    assert _parse_label("???", on_error=False) is False
    assert _parse_label("???", on_error=None) is None


# --- end-to-end through the stub ------------------------------------------- #
def test_end_to_end_labels(stub):
    out = judge_refusal(
        [("q", "I cannot help with that"), ("q", "here is how: step 1 ...")],
        model="stub",
    )
    assert out == [True, False]


def test_return_raw(stub):
    labels, raw = judge_refusal([("q", "I cannot")], model="stub", return_raw=True)
    assert labels == [True]
    assert raw == ["REFUSAL"]


def test_batching_respects_batch_size(stub):
    items = [("q", "I cannot")] * 5
    judge_refusal(items, model="stub", batch_size=2)
    assert stub.batch_sizes == [2, 2, 1]  # chunked 5 -> 2+2+1


def test_backend_failure_propagates(monkeypatch):
    """An unreachable/failed backend must raise, not silently return labels."""
    monkeypatch.setattr(J, "_make_backend", lambda *a, **k: BrokenStub())
    with pytest.raises(RuntimeError, match="connection refused"):
        judge_refusal([("q", "anything")], model="stub")


def test_unparseable_raises_by_default(monkeypatch):
    monkeypatch.setattr(J, "_make_backend", lambda *a, **k: GarbageStub())
    with pytest.raises(JudgeParseError, match="item 0"):
        judge_refusal([("q", "anything")], model="stub")


def test_unparseable_fallback_when_opted_in(monkeypatch):
    monkeypatch.setattr(J, "_make_backend", lambda *a, **k: GarbageStub())
    out = judge_refusal([("q", "x"), ("q", "y")], model="stub", on_parse_error=None)
    assert out == [None, None]


def test_api_backend_surfaces_provider_error(monkeypatch):
    """ApiBackend must raise the exact provider error, not swallow it to ''."""
    import litellm

    from refusal_llm_judge.backends import ApiBackend

    boom = ConnectionError("Failed to connect to Ollama at http://localhost:11434")
    monkeypatch.setattr(litellm, "batch_completion", lambda **kw: [boom])

    be = ApiBackend("ollama_chat/qwen3.6:35b-a3b")
    with pytest.raises(RuntimeError, match="Failed to connect to Ollama") as ei:
        be.generate([[{"role": "user", "content": "hi"}]])
    assert isinstance(ei.value.__cause__, ConnectionError)  # exact reason preserved


def test_api_backend_disables_ollama_thinking():
    """Ollama models default to think=False (judge is a one-word task)."""
    from refusal_llm_judge.backends import ApiBackend

    assert ApiBackend("ollama_chat/qwen3.6:35b-a3b").completion_kwargs == {"think": False}
    assert ApiBackend("ollama/llama3").completion_kwargs == {"think": False}
    # non-ollama providers are untouched
    assert ApiBackend("gpt-4o-mini").completion_kwargs == {}
    # explicit think wins
    assert ApiBackend("ollama_chat/x", think=True).completion_kwargs == {"think": True}


def test_empty_items_short_circuits():
    # no backend call needed
    assert judge_refusal([], model="stub") == []


def test_default_model_routes_to_api():
    # qwen3.6:35b-a3b is an Ollama (API) model, not in-process.
    assert J.DEFAULT_MODEL.startswith("ollama_chat/")
    assert J._looks_local(J.DEFAULT_MODEL) is False
