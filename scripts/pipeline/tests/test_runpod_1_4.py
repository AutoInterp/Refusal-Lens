"""                                                                                                                                                       
Pipeline Integration Test — Stages 01 through 04                                                                                                          
=================================================                                                                                                         
Runs the full pipeline end-to-end on RunPod.                                                                                                              
Validates outputs at each stage before proceeding to the next.                                                                                            
                                                                                                                                                        
Usage:                                                                                                                                                    
python scripts/pipeline/run_pipeline_test.py                      # default: 150 prompts                                                                
python scripts/pipeline/run_pipeline_test.py --n-prompts 50       # smaller test
python scripts/pipeline/run_pipeline_test.py --skip-stage 01 02   # skip heavy stages                                                                   
python scripts/pipeline/run_pipeline_test.py --run-dir <path>     # resume into existing run                                                            
                                                                                                                                                        
Expected runtime (150 prompts, A40/RTX 6000 Ada):                                                                                                         
Stage 01: ~15 min                                                                                                                                       
Stage 02: ~5-6 hours
Stage 02b: ~30 sec                                                                                                                                      
Stage 03: ~15 min                    
Stage 04: ~2 min                                                                                                                                        
"""
from __future__ import annotations                                                                                                                        
                                        
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path                                                                                                                                  

PIPELINE_DIR = Path(__file__).resolve().parent                                                                                                            
REPO_ROOT = PIPELINE_DIR.parent.parent 
                                                                                                                                                        

def parse_args():                                                                                                                                         
    parser = argparse.ArgumentParser(description="Pipeline integration test (Stages 01-04)")
    parser.add_argument("--n-prompts", type=int, default=150)
    parser.add_argument("--n-samples", type=int, default=64)                                                                                              
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-stage", nargs="+", default=[], help="Stages to skip (e.g. 01 02)")                                                        
    parser.add_argument("--resume", action="store_true", help="Resume Stage 02 from checkpoint")                                                          
    return parser.parse_args()                                                                                                                            
                                                                                                                                                        
                                        
def setup_environment():                                                                                                                                  
    """Configure RunPod environment if applicable."""
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
        os.environ.setdefault("TMPDIR", "/workspace/tmp")                                                                                                 
        Path("/workspace/tmp").mkdir(exist_ok=True)
                                                                                                                                                        
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists() and "HF_TOKEN" not in os.environ:                                                                                          
            os.environ["HF_TOKEN"] = token_path.read_text().strip()                                                                                       
            print("  HF_TOKEN loaded from cache")
                                                                                                                                                        
    # Get GPU info                     
    try:                                                                                                                                                  
        import torch                   
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9                                                                                 
            return f"{gpu_name} ({gpu_mem:.0f}GB)"
    except Exception:                                                                                                                                     
        pass                           
    return "unknown"                                                                                                                                      
                                        
                                                                                                                                                        
def run_stage(name: str, stage_id: str, cmd: list[str], skip_stages: list[str],
            log_file: Path) -> tuple[bool, float]:                                                                                                      
    """Run a pipeline stage as a subprocess. Returns (success, elapsed_seconds)."""                                                                       
    if stage_id in skip_stages:                                                                                                                           
        print(f"\n[SKIP] Stage {stage_id}: {name}")                                                                                                       
        return True, 0.0                                                                                                                                  
                                        
    print(f"\n{'='*60}")                                                                                                                                  
    print(f"[START] Stage {stage_id}: {name} ({datetime.now().strftime('%H:%M:%S')})")
    print(f"{'='*60}")                                                                                                                                    
                                                                                                                                                        
    t0 = time.time()
                                                                                                                                                        
    with open(log_file, "a") as lf:    
        lf.write(f"\n{'='*60}\n")
        lf.write(f"Stage {stage_id}: {name} — {datetime.now().isoformat()}\n")                                                                            
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write(f"{'='*60}\n")                                                                                                                           
                                        
    # Run with unbuffered output                                                                                                                          
    result = subprocess.run(           
        [sys.executable, "-u"] + cmd,
        cwd=str(REPO_ROOT),                                                                                                                               
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,                                                                                                                         
    )                                  

    elapsed = time.time() - t0                                                                                                                            
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)                                                                                                                           
                                        
    if result.returncode == 0:
        print(f"\n[DONE] Stage {stage_id}: {minutes}m {seconds}s")
        with open(log_file, "a") as lf:                                                                                                                   
            lf.write(f"[DONE] {minutes}m {seconds}s\n")
        return True, elapsed                                                                                                                              
    else:                              
        print(f"\n[FAIL] Stage {stage_id}: {name} (exit code {result.returncode})")                                                                       
        with open(log_file, "a") as lf:                                                                                                                   
            lf.write(f"[FAIL] exit code {result.returncode}\n")
        return False, elapsed                                                                                                                             
                                        
                                                                                                                                                        
def validate_stage(stage_id: str, run_dir: Path) -> list[str]:
    """Validate stage outputs exist and look correct. Returns list of issues."""                                                                          
    issues = []                                                                                                                                           

    if stage_id == "01":                                                                                                                                  
        d = run_dir / "01_direction"   
        for f in ["refusal_direction.pt", "unnormalized_r.pt", "direction_metadata.json"]:                                                                
            if not (d / f).exists():                                                                                                                      
                issues.append(f"Missing {f}")
        if (d / "direction_metadata.json").exists():                                                                                                      
            meta = json.loads((d / "direction_metadata.json").read_text())                                                                                
            if meta.get("best_separation_layer") != 32:
                issues.append(f"Unexpected best layer: {meta.get('best_separation_layer')}")                                                              
                                                                                                                                                        
    elif stage_id == "02":                                                                                                                                
        d = run_dir / "02_attribution"                                                                                                                    
        if not (d / "attribution_results.json").exists():
            issues.append("Missing attribution_results.json")                                                                                             
        else:
            raw = json.loads((d / "attribution_results.json").read_text())                                                                                
            results = raw if isinstance(raw, list) else raw.get("results", [])
            if len(results) == 0:                                                                                                                         
                issues.append("Empty attribution results")
            else:                                                                                                                                         
                # Check first result has expected structure
                r = results[0]                                                                                                                            
                if "conditions" not in r:
                    issues.append("Missing 'conditions' key in results")                                                                                  
                elif "bare" not in r.get("conditions", {}):
                    issues.append("Missing 'bare' condition")                                                                                             
                elif "net" not in r["conditions"]["bare"]:
                    issues.append("Missing 'net' in bare condition")                                                                                      
                                        
    elif stage_id == "02b":                                                                                                                               
        d = run_dir / "02b_stats"      
        for f in ["statistical_analysis.json", "class_comparison.png",                                                                                    
                    "per_prompt_deltas.png", "effect_sizes.png", "EXPERIMENT_SUMMARY.md"]:
            if not (d / f).exists():                                                                                                                      
                issues.append(f"Missing {f}")
        if (d / "statistical_analysis.json").exists():                                                                                                    
            stats = json.loads((d / "statistical_analysis.json").read_text())
            for cls in ["roleplay", "fiction", "analytical", "completion", "cognitive_reframe"]:                                                          
                if cls not in stats:                                                                                                                      
                    issues.append(f"Missing class: {cls}")                                                                                                
                elif stats[cls].get("wilcoxon_pval") is None:                                                                                             
                    issues.append(f"No Wilcoxon p-value for {cls}")
                                                                                                                                                        
    elif stage_id == "03":             
        d = run_dir / "03_verification"                                                                                                                   
        if not (d / "verification_results.json").exists():
            issues.append("Missing verification_results.json")                                                                                            
        else:
            vr = json.loads((d / "verification_results.json").read_text())                                                                                
            summary = vr.get("summary", {})
            mlp_pct = summary.get("mlp_pct_mean", 0)                                                                                                      
            if not (0.01 < mlp_pct < 5.0):                                                                                                                
                issues.append(f"MLP ratio out of expected range: {mlp_pct}%")                                                                             
                                                                                                                                                        
    elif stage_id == "04":             
        d = run_dir / "04_labels"                                                                                                                         
        for f in ["feature_labels.json", "feature_comparison_labeled.json",                                                                               
                    "label_coverage.json", "top_features_report.md"]:                                                                                      
            if not (d / f).exists():                                                                                                                      
                issues.append(f"Missing {f}")                                                                                                             
        if (d / "label_coverage.json").exists():
            cov = json.loads((d / "label_coverage.json").read_text())
            if cov.get("coverage_pct", 0) < 50:                                                                                                           
                issues.append(f"Low label coverage: {cov.get('coverage_pct')}%")
                                                                                                                                                        
    return issues                      
                                                                                                                                                        
                                        
def main():
    args = parse_args()

    # Setup
    gpu_info = setup_environment()
                                                                                                                                                        
    # Create run directory
    if args.run_dir is None:                                                                                                                              
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "data" / "results" / "pipeline_runs" / f"run_{stamp}"
    else:                                                                                                                                                 
        run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)                                                                                                            
                                        
    log_file = run_dir / "pipeline.log"                                                                                                                   
                                        
    print("=" * 60)
    print("REFUSAL-LENS PIPELINE — INTEGRATION TEST")
    print("=" * 60)                                                                                                                                       
    print(f"  Repo:       {REPO_ROOT}")
    print(f"  Run dir:    {run_dir}")                                                                                                                     
    print(f"  GPU:        {gpu_info}")                                                                                                                    
    print(f"  N prompts:  {args.n_prompts}")
    print(f"  N samples:  {args.n_samples}")                                                                                                              
    print(f"  Skip:       {args.skip_stage or 'none'}")
    print(f"  Resume:     {args.resume}")                                                                                                                 
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                                                                                                                                        
    # Save run config                  
    run_config = {                                                                                                                                        
        "n_prompts": args.n_prompts,   
        "n_samples": args.n_samples,
        "skip_stages": args.skip_stage,                                                                                                                   
        "resume": args.resume,
        "started": datetime.now().isoformat(),                                                                                                            
        "gpu": gpu_info,               
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)                                                                                                                

    # Define stages                                                                                                                                       
    p = str(PIPELINE_DIR)              
    resume_flag = ["--resume"] if args.resume else []                                                                                                     

    stages = [                                                                                                                                            
        ("Compute Directions", "01", [ 
            f"{p}/01_compute_direction.py",                                                                                                               
            "--run-dir", str(run_dir),
            "--n-samples", str(args.n_samples),                                                                                                           
        ]),                            
        ("Run Attribution", "02", [
            f"{p}/02_run_attribution.py",                                                                                                                 
            "--run-dir", str(run_dir),
            "--n-prompts", str(args.n_prompts),                                                                                                           
        ] + resume_flag),              
        ("Statistical Analysis", "02b", [                                                                                                                 
            f"{p}/02b_statistical_analysis.py",
            "--run-dir", str(run_dir),                                                                                                                    
        ]),                            
        ("Verify Attribution", "03", [
            f"{p}/03_verify_attribution.py",
            "--run-dir", str(run_dir),                                                                                                                    
            "--n-decompose", "10",
        ]),                                                                                                                                               
        ("Feature Labeling", "04", [   
            f"{p}/04_label_features.py",
            "--run-dir", str(run_dir),                                                                                                                    
        ]),
    ]                                                                                                                                                     
                                        
    # Run stages
    timings = {}
    for name, stage_id, cmd in stages:
        success, elapsed = run_stage(name, stage_id, cmd, args.skip_stage, log_file)

        if not success:
            print(f"\nPipeline stopped at Stage {stage_id}.")
            print(f"Check output at: {run_dir}")                                                                                                          
            sys.exit(1)
                                                                                                                                                        
        timings[stage_id] = elapsed    
                                                                                                                                                        
        # Validate after each stage    
        if stage_id not in args.skip_stage:
            issues = validate_stage(stage_id, run_dir)
            if issues:                                                                                                                                    
                print(f"\n[WARN] Stage {stage_id} validation issues:")
                for issue in issues:                                                                                                                      
                    print(f"  - {issue}")
            else:                                                                                                                                         
                print(f"[OK] Stage {stage_id} validation passed")
                                                                                                                                                        
    # Summary                          
    total = sum(timings.values())
    print(f"\n{'='*60}")                                                                                                                                  
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")                                                                                                                                    
    print(f"  Run dir:  {run_dir}")    
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")                                                                                  
    print(f"\n  Stage timings:")
    for stage_id, elapsed in timings.items():                                                                                                             
        if elapsed > 0:                
            print(f"    Stage {stage_id}: {int(elapsed//60)}m {int(elapsed%60)}s")                                                                        
    print(f"    Total:    {int(total//60)}m {int(total%60)}s")                                                                                            
                                                                                                                                                        
    print(f"\n  Review results:")                                                                                                                         
    print(f"    cat {run_dir}/02b_stats/EXPERIMENT_SUMMARY.md")                                                                                           
    print(f"    cat {run_dir}/04_labels/top_features_report.md")
    print(f"    cat {run_dir}/04_labels/label_coverage.json")                                                                                             

                                                                                                                                                        
if __name__ == "__main__":             
    main()