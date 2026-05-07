"""
Stage 08 ablation-demo backend (FastAPI).
==========================================
A thin HTTP service over `ReplacementModel.feature_intervention_generate` that
the Stage 05 frontend cart can POST to for a live ablation demo.

Singleton: loads `ReplacementModel` once at startup (~60-120s) and reuses
across requests. CORS is opened for localhost:8000 (the frontend dev server).

Backend defaults to `transformerlens` (matches `config.BACKEND` and the user's
working ablation-experiment setup on Georg's measurement_hook patch). The
nnsight backend is still selectable via `--backend nnsight` for parity with
older code paths. Both backends expose `feature_intervention_generate` with
identical signatures since vendor/circuit-tracer's unified ReplacementModel.

Endpoints:
  POST /ablate   — run one (baseline, ablated) pair for a given feature set
  GET  /health   — liveness probe

Request schema for POST /ablate:
  {
    "features": [{"layer": 14, "feat_idx": 480, "value": 0.0}, ...],
    "prompt": "Write instructions for how to pick a lock.",
    "positions": "all" | "anchors",
    "max_new_tokens": 200   (optional)
  }

Response schema:
  {
    "baseline": str, "baseline_cls": "REFUSE"|"COMPLY", "baseline_coherent": bool,
    "ablated":  str, "ablated_cls":  "REFUSE"|"COMPLY", "ablated_coherent":  bool,
    "n_features": int, "positions": str, "elapsed_s": float,
  }

Usage:
    # GPU host (real-time ablation; ~60-120 s model load, ~5-30 s/request):
    PYTHONPATH=src python3 scripts/pipeline/ablation_server.py \\
        --host 127.0.0.1 --port 8080

    # Wiring-test on a CPU/Mac (model load skipped until first /ablate call):
    PYTHONPATH=src python3 scripts/pipeline/ablation_server.py --lazy-model

Then from the frontend (served at localhost:8000) the cart's "Run ablation"
button will POST to localhost:8080/ablate. Without a GPU the first request
will fail or take many minutes — use --lazy-model only to validate that the
HTTP wiring + frontend cart work end-to-end before deploying on a GPU host.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    classify_response,
    format_prompt,
    is_coherent,
    resolve_anchor_positions,
)


def _build_interventions(features, positions_mode, tokenized_length):
    """Duplicate-free copy of 08's build_interventions — kept local so the
    server module has no import dep on the stage-08 CLI script."""
    if positions_mode == "all":
        pos = slice(None)
        return [(int(f["layer"]), pos, int(f["feat_idx"]), float(f.get("value", 0.0)))
                for f in features]
    if positions_mode == "anchors":
        anchors = resolve_anchor_positions(list(config.STAGE_08_TEMPLATE_ANCHORS), tokenized_length)
        return [
            (int(f["layer"]), int(pos), int(f["feat_idx"]), float(f.get("value", 0.0)))
            for f in features
            for pos in anchors
        ]
    raise ValueError(f"Unknown positions mode: {positions_mode!r}")


def _extract_response(full: str, formatted: str) -> str:
    if full.startswith(formatted):
        return full[len(formatted):]
    marker = "<start_of_turn>model\n"
    idx = full.rfind(marker)
    return full[idx + len(marker):] if idx >= 0 else full


class ModelHolder:
    """Module-level singleton for the lazy-loaded ReplacementModel."""
    model = None
    tokenizer = None
    dtype_str = "bfloat16"
    backend = config.BACKEND

    @classmethod
    def load(cls, dtype_str: str = "bfloat16", backend: str | None = None):
        if cls.model is not None:
            return
        backend = backend or cls.backend
        print(f"[ablation_server] loading ReplacementModel "
              f"({config.MODEL_NAME}, {dtype_str}, backend={backend})…")
        import torch
        from circuit_tracer import ReplacementModel

        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        cls.model = ReplacementModel.from_pretrained(
            config.MODEL_NAME,
            config.TRANSCODER_PATH,
            dtype=dtype_map[dtype_str],
            backend=backend,
            lazy_encoder=False,
        )
        cls.tokenizer = cls.model.tokenizer
        cls.dtype_str = dtype_str
        cls.backend = backend
        print("[ablation_server] model ready.")


def build_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError as e:
        raise SystemExit(
            "ablation_server requires fastapi + pydantic + uvicorn.\n"
            "Install with: pip install fastapi pydantic uvicorn\n"
            f"Underlying import error: {e}"
        ) from e

    class Feature(BaseModel):
        layer: int
        feat_idx: int
        value: float = 0.0
        label: str | None = None

    class AblateRequest(BaseModel):
        features: list[Feature]
        prompt: str
        positions: Literal["all", "anchors"] = "all"
        max_new_tokens: int = config.MAX_NEW_TOKENS

    class AblateResponse(BaseModel):
        baseline: str
        baseline_cls: str
        baseline_coherent: bool
        ablated: str
        ablated_cls: str
        ablated_coherent: bool
        n_features: int
        positions: str
        elapsed_s: float

    app = FastAPI(
        title="Refusal-Lens Ablation Demo",
        description="Zero-ablate transcoder features and compare baseline vs ablated generation.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok" if ModelHolder.model is not None else "loading",
            "model": config.MODEL_NAME,
            "dtype": ModelHolder.dtype_str,
            "backend": ModelHolder.backend,
        }

    @app.post("/ablate", response_model=AblateResponse)
    def ablate(req: AblateRequest):
        if ModelHolder.model is None:
            raise HTTPException(503, "model still loading — try again in a moment")
        rm = ModelHolder.model
        tokenizer = ModelHolder.tokenizer
        formatted = format_prompt(tokenizer, req.prompt)
        tokenized_length = len(tokenizer(formatted)["input_ids"])
        feats = [f.dict() for f in req.features]
        interventions = _build_interventions(feats, req.positions, tokenized_length)
        t0 = time.time()

        # Baseline (no interventions)
        decoded_base, _, _ = rm.feature_intervention_generate(
            formatted, [], max_new_tokens=req.max_new_tokens,
            return_activations=False, do_sample=False,
        )
        base_resp = _extract_response(decoded_base, formatted).strip()

        # Ablated
        decoded_abl, _, _ = rm.feature_intervention_generate(
            formatted, interventions, max_new_tokens=req.max_new_tokens,
            return_activations=False, do_sample=False,
        )
        abl_resp = _extract_response(decoded_abl, formatted).strip()

        elapsed = time.time() - t0
        return AblateResponse(
            baseline=base_resp[:1000],
            baseline_cls=classify_response(base_resp),
            baseline_coherent=is_coherent(base_resp),
            ablated=abl_resp[:1000],
            ablated_cls=classify_response(abl_resp),
            ablated_coherent=is_coherent(abl_resp),
            n_features=len(req.features),
            positions=req.positions,
            elapsed_s=round(elapsed, 2),
        )

    return app


def main():
    p = argparse.ArgumentParser(description="Stage 08 ablation demo server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--backend", choices=["transformerlens", "nnsight"],
                   default=config.BACKEND,
                   help=f"ReplacementModel backend (default: {config.BACKEND}). "
                        "TransformerLens matches Georg's measurement_hook patch and the user's "
                        "ablation-experiment setup; nnsight is the legacy alternative.")
    p.add_argument("--lazy-model", action="store_true",
                   help="Delay model load until the first /ablate call. Lets you smoke-test the "
                        "FastAPI wiring + frontend cart on a CPU box; the first POST /ablate will "
                        "block on the load (and likely OOM without a real GPU).")
    args = p.parse_args()

    app = build_app()
    if not args.lazy_model:
        ModelHolder.load(args.dtype, backend=args.backend)
    else:
        ModelHolder.dtype_str = args.dtype
        ModelHolder.backend = args.backend
        print(f"[ablation_server] --lazy-model: model will load on first /ablate call "
              f"(backend={args.backend}, dtype={args.dtype})")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
