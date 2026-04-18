"""Shared helpers for Stage 05 circuit visualization."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

OVERLAP_BUCKETS = ("shared_with_bare", "jb_unique", "bare_only", "bare", "non_feature")

# Circuit-tracer browser frontend lives at vendor/circuit-tracer/circuit_tracer/frontend/assets/
# (NOT the parent `frontend/` dir, which contains Python helpers).
VENDOR_FRONTEND = (
    Path(__file__).resolve().parents[2]
    / "vendor" / "circuit-tracer" / "circuit_tracer" / "frontend" / "assets"
)


def feature_key_from_node(node: dict) -> str | None:
    """Extract L{layer}:F{feature_idx} from a frontend-format node dict.
    Returns None for non-feature nodes (embedding/logit/error)."""
    if node.get("feature_type") != "cross layer transcoder":
        return None
    layer = node.get("layer")
    # Node IDs look like "0_4096_5" = "{layer}_{feature}_{ctx_idx}"
    parts = str(node.get("node_id", "")).split("_")
    if len(parts) < 2:
        return None
    try:
        return f"L{int(parts[0])}:F{int(parts[1])}"
    except (ValueError, TypeError):
        return None


def convert_pt_to_frontend_json(
    pt_path: Path, slug: str, out_dir: Path,
    scan: str = "mwhanna/gemma-scope-2-4b-it//transcoder_all/width_16k_l0_small_affine",
    node_threshold: float = 0.8, edge_threshold: float = 0.98,
) -> Path:
    """Wrap circuit-tracer's create_graph_files. Returns path to <slug>.json."""
    from circuit_tracer.utils.create_graph_files import create_graph_files

    out_dir.mkdir(parents=True, exist_ok=True)
    create_graph_files(
        graph_or_path=str(pt_path),
        slug=slug,
        output_path=str(out_dir),
        scan=scan,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    return out_dir / f"{slug}.json"


def annotate_overlap(
    jb_json_path: Path, bare_json_path: Path, jb_class: str, prompt_idx: int,
) -> dict:
    """Load a JB graph JSON, annotate each feature node with overlap_bucket
    relative to the bare graph, persist in-place, and return the mutated dict."""
    with open(jb_json_path) as f:
        jb = json.load(f)
    with open(bare_json_path) as f:
        bare = json.load(f)

    bare_keys = {
        k for k in (feature_key_from_node(n) for n in bare["nodes"]) if k is not None
    }

    for node in jb["nodes"]:
        key = feature_key_from_node(node)
        if key is None:
            node["overlap_bucket"] = "non_feature"
        elif key in bare_keys:
            node["overlap_bucket"] = "shared_with_bare"
        else:
            node["overlap_bucket"] = "jb_unique"

    jb["metadata"]["jb_class"] = jb_class
    jb["metadata"]["prompt_idx"] = prompt_idx

    with open(jb_json_path, "w") as f:
        json.dump(jb, f, indent=2)
    return jb


def annotate_bare(bare_json_path: Path, prompt_idx: int) -> dict:
    """Tag every feature node in a bare graph JSON as 'bare'."""
    with open(bare_json_path) as f:
        bare = json.load(f)
    for node in bare["nodes"]:
        if feature_key_from_node(node) is not None:
            node["overlap_bucket"] = "bare"
        else:
            node["overlap_bucket"] = "non_feature"
    bare["metadata"]["jb_class"] = "bare"
    bare["metadata"]["prompt_idx"] = prompt_idx
    with open(bare_json_path, "w") as f:
        json.dump(bare, f, indent=2)
    return bare


def gzip_json_files(graph_data_dir: Path, keep_plain: bool = False) -> dict:
    """Compress every *.json in graph_data_dir to *.json.gz. Returns size report."""
    results = {"compressed": {}, "total_plain": 0, "total_gz": 0}
    for jp in sorted(graph_data_dir.glob("*.json")):
        gz_path = jp.with_suffix(".json.gz")
        with open(jp, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        plain = jp.stat().st_size
        gz = gz_path.stat().st_size
        results["compressed"][jp.name] = {"plain": plain, "gz": gz, "ratio": round(plain / gz, 2)}
        results["total_plain"] += plain
        results["total_gz"] += gz
        if not keep_plain:
            jp.unlink()
    return results


def stage_frontend(
    graph_data_dir: Path, frontend_out: Path, vendor_frontend: Path = VENDOR_FRONTEND,
) -> None:
    """Copy circuit-tracer's frontend assets + arrange our graphs into the expected layout.

    Final layout:
        <frontend_out>/
            index.html, style.css, util.js, attribution_graph/..., assets/...   (from vendor)
            graph_data/<slug>.json     (moved from graph_data_dir)
            data/graph-metadata.json   (moved from graph_data_dir)
    """
    if not vendor_frontend.exists():
        raise FileNotFoundError(f"vendor frontend not found at {vendor_frontend}")
    frontend_out.mkdir(parents=True, exist_ok=True)

    # Copy frontend scaffolding (don't overwrite existing graph_data/ if already there)
    for entry in vendor_frontend.iterdir():
        if entry.name in {"graph_data", "data"}:
            continue
        dst = frontend_out / entry.name
        if entry.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)

    # Copy our graphs into graph_data/
    graph_data_out = frontend_out / "graph_data"
    graph_data_out.mkdir(exist_ok=True)
    for jp in sorted(graph_data_dir.glob("*.json")):
        if jp.name == "graph-metadata.json":
            continue
        shutil.copy2(jp, graph_data_out / jp.name)
    for gz in sorted(graph_data_dir.glob("*.json.gz")):
        shutil.copy2(gz, graph_data_out / gz.name)
    
    # Inject overlap-coloring patches into index.html
    patches_dir = Path(__file__).resolve().parent / "05_frontend_patches"                                                       
    if patches_dir.exists():                                                                                                    
        # Copy patch files into frontend root                                                                                   
        for patch in patches_dir.iterdir():                                                                                     
            shutil.copy2(patch, frontend_out / patch.name)
        # Inject <link> + <script> into index.html                                                                              
        index_path = frontend_out / "index.html"                                                                                
        html = index_path.read_text()
        injection = (                                                                                                           
            '<link rel="stylesheet" href="./overlap-colors.css">\n'
            '<script src="./overlap-annotate.js" defer></script>\n'                                                             
        )       
        # Insert before the closing of the first <link> block (just after last CSS link)                                        
        marker = "<script src='./util.js'></script>"                                                                            
        if marker in html and injection not in html:
            html = html.replace(marker, injection + marker)                                                                     
            index_path.write_text(html)

    # Move graph-metadata.json to data/
    data_out = frontend_out / "data"
    data_out.mkdir(exist_ok=True)
    metadata_src = graph_data_dir / "graph-metadata.json"
    if metadata_src.exists():
        shutil.copy2(metadata_src, data_out / "graph-metadata.json")