"""
Generate feature descriptions from feature_labels_cache.json.

This script:
1. Loads feature_labels_cache.json from any run directory
2. Extracts layer/feature indices
3. Fetches descriptions from Neuronpedia/OpenRouter API or generates them from logits
4. Outputs feature_labels_layer_N.json files per layer
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_ID = "gemma-3-4b-it"
SAE_TEMPLATE = "{layer}-gemmascope-2-res-16k"
BASE_URL = "https://www.neuronpedia.org/api/feature"
EXPLANATION_GENERATE_URL = "https://www.neuronpedia.org/api/explanation/generate"
SLEEP_BETWEEN_REQUESTS = 0.5
MAX_RETRIES = 3
TIMEOUT = 60  # explanation/generate can be slow (LLM call server-side)
HEADERS = {"User-Agent": "refusal-lens-script/1.0"}
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Defaults for /api/explanation/generate.
# explanationType is the Neuronpedia "explanation type" key (e.g. "oai_token-act-pair");
# explanationModelName is the LLM Neuronpedia uses to author the explanation.
DEFAULT_EXPLANATION_TYPE = "oai_token-act-pair"
DEFAULT_EXPLANATION_MODEL = "gpt-4o-mini"

# Sentinel returned by generate_explanation_via_api when Neuronpedia responds
# 400 "No activations found for this neuron" -- expected for sparsely-populated
# SAEs and not an error. Callers should treat this as "skip Neuronpedia, fall
# back" while tallying the count.
_NO_ACTIVATIONS = "__NEURONPEDIA_NO_ACTIVATIONS__"


def extract_indices_from_cache(cache: dict) -> list[tuple[int, int]]:
    """Extract (layer, feature_index) pairs from feature_labels_cache.json format."""
    pairs = []
    pattern = re.compile(r"L(\d+):F(\d+)")
    for key in cache.keys():
        match = pattern.match(key)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
    return pairs


def generate_label_from_description(description: str) -> str:
    """Generate a short label (≤7 words) from a description via local truncation.

    The Neuronpedia explanation API only generates explanations for SAE features
    from their activations -- it cannot summarize arbitrary text -- so label
    shortening is done locally.
    """
    if not description:
        return ""

    words = description.strip().split()
    label = " ".join(words[:7]).rstrip(".,;:")
    if len(words) > 7:
        label += "..."
    return label


def generate_explanation_via_api(
    layer: int,
    feature_index: int,
    api_key: str,
    explanation_type: str = DEFAULT_EXPLANATION_TYPE,
    explanation_model: str = DEFAULT_EXPLANATION_MODEL,
) -> str:
    """Generate a new explanation for a feature via Neuronpedia.

    Endpoint: POST /api/explanation/generate
    Docs:     https://www.neuronpedia.org/api-doc#tag/explanations/POST/api/explanation/generate

    Body shape (per docs):
        {
          "modelId":              "<model id>",
          "layer":                "<sae source-set id, e.g. 20-gemmascope-2-res-16k>",
          "index":                <feature index, int>,
          "explanationType":      "<type key>",
          "explanationModelName": "<llm name>"
        }
    """
    sae_id = SAE_TEMPLATE.format(layer=layer)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload = {
        "modelId": MODEL_ID,
        "layer": sae_id,                 # NOTE: SAE source-set id, not bare layer number
        "index": int(feature_index),     # API expects an int
        "explanationType": explanation_type,
        "explanationModelName": explanation_model,
    }

    try:
        resp = requests.post(
            EXPLANATION_GENERATE_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"  [explanation/generate] L{layer}:F{feature_index} -> network error: {e}")
        return ""

    if resp.status_code != 200:
        body = resp.text[:200]
        # Expected case: Neuronpedia hasn't cached activations for this feature,
        # so auto-interp can't run. Bubble the signal up via a sentinel so the
        # caller can tally it instead of printing per-feature noise.
        if resp.status_code == 400 and "No activations found" in body:
            return _NO_ACTIVATIONS
        print(
            f"  [explanation/generate] L{layer}:F{feature_index} -> "
            f"HTTP {resp.status_code}: {body}"
        )
        return ""

    try:
        result = resp.json()
    except ValueError:
        return ""

    # Response is the created explanation object. Field naming has varied across
    # Neuronpedia versions; try the documented `description`, then fallbacks.
    if isinstance(result, dict):
        if isinstance(result.get("description"), str):
            return result["description"]
        if isinstance(result.get("explanation"), str):
            return result["explanation"]
        nested = result.get("explanation")
        if isinstance(nested, dict) and isinstance(nested.get("description"), str):
            return nested["description"]
    return ""


def fetch_neuronpedia_description(
    layer: int,
    feature_index: int,
    api_key: str | None = None,
) -> str:
    """Fetch an existing description; if none exists, generate one."""
    sae_id = SAE_TEMPLATE.format(layer=layer)

    # 1) Try fetching existing explanation
    url = f"{BASE_URL}/{quote(MODEL_ID)}/{quote(sae_id)}/{feature_index}"
    headers = HEADERS.copy()
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            explanations = data.get("explanations") or []
            if explanations:
                desc = explanations[0].get("description", "")
                if desc:
                    return desc
            break  # 200 OK but no explanation cached -> fall through to generate
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2 ** attempt)

    # 2) No cached explanation -> ask Neuronpedia to generate one
    if api_key:
        return generate_explanation_via_api(layer, feature_index, api_key)

    return ""


def generate_description_from_logits(top_logits: list, bottom_logits: list) -> str:
    """Generate a basic description from top/bottom logits when API unavailable."""
    if not top_logits and not bottom_logits:
        return ""

    parts = []
    if top_logits:
        clean_top = [t.strip() for t in top_logits[:5] if t.strip() and not t.startswith("<")]
        if clean_top:
            parts.append(f"activates on: {', '.join(clean_top)}")
    if bottom_logits:
        clean_bottom = [t.strip() for t in bottom_logits[:5] if t.strip() and not t.startswith("<")]
        if clean_bottom:
            parts.append(f"suppresses: {', '.join(clean_bottom)}")

    return "; ".join(parts)


def generate_with_openrouter(
    layer: int,
    feature_index: int,
    top_logits: list,
    bottom_logits: list,
    api_key: str,
) -> str:
    """Generate description using OpenRouter API (fallback when Neuronpedia has nothing)."""
    prompt = f"""Analyze these SAE feature activations for layer {layer}, feature {feature_index}:

Top activating tokens: {top_logits[:10]}
Bottom activating tokens: {bottom_logits[:10]}

Provide a concise 1-2 sentence description of what this feature detects/promotes."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "google/gemini-flash-1.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }

    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=TIMEOUT)
        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
    except requests.RequestException:
        pass

    return ""


def process_cache(
    cache_path: Path,
    output_dir: Path,
    use_api: bool = True,
    api_key: str | None = None,
) -> None:
    """Process cache file and generate descriptions per layer."""
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    pairs = extract_indices_from_cache(cache)

    layers: dict[int, dict[str, dict]] = {}
    for layer, _idx in pairs:
        layers.setdefault(layer, {})

    print(f"Processing {len(pairs)} features from {cache_path.name}")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    stats = {
        "neuronpedia": 0,
        "openrouter": 0,
        "local_logits": 0,
        "no_activations": 0,  # subset: Neuronpedia had no activations cached
        "empty": 0,
    }

    for i, (layer, idx) in enumerate(pairs, 1):
        feature_data = cache.get(f"L{layer}:F{idx}", {})
        source = "empty"

        if use_api:
            desc = fetch_neuronpedia_description(layer, idx, api_key)
            if desc == _NO_ACTIVATIONS:
                stats["no_activations"] += 1
                desc = ""
            elif desc:
                source = "neuronpedia"

            if not desc and openrouter_key:
                desc = generate_with_openrouter(
                    layer, idx,
                    feature_data.get("top_logits", []),
                    feature_data.get("bottom_logits", []),
                    openrouter_key,
                )
                if desc:
                    source = "openrouter"

            if not desc and feature_data:
                desc = generate_description_from_logits(
                    feature_data.get("top_logits", []),
                    feature_data.get("bottom_logits", []),
                )
                if desc:
                    source = "local_logits"
        else:
            desc = generate_description_from_logits(
                feature_data.get("top_logits", []),
                feature_data.get("bottom_logits", []),
            )
            if desc:
                source = "local_logits"

        stats[source] += 1

        layers[layer][str(idx)] = {
            "explanation": {
                "label": generate_label_from_description(desc),
                "description": desc,
            }
        }

        if i % 50 == 0:
            print(f"  Processed {i}/{len(pairs)}")

        if use_api:
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, features in sorted(layers.items()):
        output_file = output_dir / f"feature_labels_layer_{layer}.json"
        output_file.write_text(json.dumps(features, ensure_ascii=False, indent=2))

    print(f"Saved descriptions to {output_dir}")
    print(
        "Sources: "
        f"neuronpedia={stats['neuronpedia']}, "
        f"openrouter={stats['openrouter']}, "
        f"local_logits={stats['local_logits']}, "
        f"empty={stats['empty']} "
        f"(of which {stats['no_activations']} features had no Neuronpedia activations)"
    )


def regenerate_labels(
    input_dir: Path,
    output_dir: Path | None = None,
    neuronpedia_api_key: str | None = None,
) -> None:
    """Regenerate short labels (≤7 words) from existing descriptions.

    Label generation is local (the Neuronpedia explanation API doesn't summarize
    arbitrary text). The api-key arg is kept for forward compatibility but unused.
    """
    _ = neuronpedia_api_key  # kept for backward-compatible call sites

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in sorted(input_dir.glob("feature_labels_layer_*.json")):
        features = json.loads(input_file.read_text(encoding="utf-8"))

        for idx, data in features.items():
            desc = data.get("explanation", {}).get("description", "")
            if desc:
                features[idx]["explanation"]["label"] = generate_label_from_description(desc)

        output_file = output_dir / input_file.name
        output_file.write_text(json.dumps(features, ensure_ascii=False, indent=2))
        print(f"Updated labels in {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate feature descriptions")
    parser.add_argument("cache_file", type=Path, nargs="?", help="Path to feature_labels_cache.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: feature_labels/)")
    parser.add_argument("--no-api", action="store_true", help="Generate descriptions locally without API")
    parser.add_argument("--api-key", type=str, default=None, help="Neuronpedia API key (or set NEURONPEDIA_API_KEY in .env)")
    parser.add_argument("--regenerate-labels", type=Path, metavar="DIR", help="Regenerate labels in existing feature_labels directory")
    args = parser.parse_args()

    if args.regenerate_labels:
        regenerate_labels(args.regenerate_labels)
        return

    if not args.cache_file:
        parser.error("cache_file is required unless using --regenerate-labels")

    api_key = args.api_key or os.getenv("NEURONPEDIA_API_KEY")
    output = args.output or args.cache_file.parent.parent / "feature_labels"
    process_cache(args.cache_file, output, use_api=not args.no_api, api_key=api_key)


if __name__ == "__main__":
    main()
