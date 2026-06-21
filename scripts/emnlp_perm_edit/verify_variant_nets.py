"""Correctness gate for regenerated variant attribution.

A fresh Stage-02 run targeting a refusal-direction variant must reproduce
Georg's attributed `net` magnitudes (data/.../gemma_var_nets.json). If it does
not, the run used the wrong direction / measurement hook / circuit-tracer build
and the graphs must NOT be trusted. Exit code 0 = pass, 1 = fail (drives the
orchestrator).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_nets(attribution_results: dict) -> list[dict]:
    out = []
    for res in attribution_results.get("results", []):
        idx = res.get("prompt_idx")
        for cond, cdata in (res.get("conditions") or {}).items():
            single = ((cdata or {}).get("graphs") or {}).get("single") or {}
            net = single.get("net")
            if net is not None:
                out.append({"prompt_idx": idx, "condition": cond, "net": float(net)})
    return out


def _bare_mean(records: list[dict]) -> float:
    vals = [r["net"] for r in records if r["condition"] == "bare"]
    return sum(vals) / len(vals) if vals else float("nan")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def compare_nets(new_records: list[dict], ref_records: list[dict], *,
                 corr_min: float = 0.95, bare_rel_tol: float = 0.25) -> dict:
    ref_by_key = {(r["prompt_idx"], r["condition"]): r["net"] for r in ref_records}
    paired = [(r["net"], ref_by_key[(r["prompt_idx"], r["condition"])])
              for r in new_records if (r["prompt_idx"], r["condition"]) in ref_by_key]
    corr = _pearson([a for a, _ in paired], [b for _, b in paired]) if len(paired) >= 2 else float("nan")
    nb, rb = _bare_mean(new_records), _bare_mean(ref_records)
    sign_ok = (nb == 0 and rb == 0) or (nb * rb > 0)
    rel = abs(nb - rb) / max(abs(rb), 1e-9)
    corr_ok = (len(paired) < 2) or (corr >= corr_min)
    ok = bool(corr_ok and sign_ok and rel <= bare_rel_tol)
    return {"ok": ok, "n_paired": len(paired), "corr": corr,
            "bare_mean_new": nb, "bare_mean_ref": rb,
            "sign_ok": sign_ok, "bare_rel_err": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution-results", type=Path, required=True)
    ap.add_argument("--nets-ref", type=Path, required=True)
    ap.add_argument("--variant", required=True, choices=["full", "outlier", "complement"])
    ap.add_argument("--corr-min", type=float, default=0.95)
    ap.add_argument("--bare-rel-tol", type=float, default=0.25)
    a = ap.parse_args()
    new = extract_nets(json.loads(a.attribution_results.read_text()))
    ref = json.loads(a.nets_ref.read_text())["variants"][a.variant]
    r = compare_nets(new, ref, corr_min=a.corr_min, bare_rel_tol=a.bare_rel_tol)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
