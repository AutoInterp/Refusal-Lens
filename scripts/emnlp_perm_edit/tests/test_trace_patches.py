"""Structural checks for the trace frontend patch files (no browser)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]            # repo root via scripts/..
PATCHES = ROOT / "scripts/pipeline/05_frontend_patches"


def test_trace_highlight_js_has_required_hooks():
    js = (PATCHES / "trace-highlight.js").read_text()
    for needle in ["rl_trace_class", "data-rl-trace", "MutationObserver",
                   ".link-graph text.node", "amplification"]:
        assert needle in js, needle


def test_trace_highlight_css_colors_all_classes():
    css = (PATCHES / "trace-highlight.css").read_text()
    for needle in ['data-rl-trace="refusal_centric"', 'data-rl-trace="suppression"',
                   'data-rl-trace="amplification"', 'data-rl-trace="neutral"']:
        assert needle in css, needle


def test_trace_html_structure():
    html = (PATCHES / "trace.html").read_text()
    for needle in ["trace_manifest.json", "frame-bare", "frame-jb",
                   "compact=1", "Refusal-centric", "Suppression", "Amplification",
                   "evidence", "overlap_bucket", "overlap"]:
        assert needle in html, needle
