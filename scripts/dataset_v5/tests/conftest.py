import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]        # scripts/dataset_v5
REPO = Path(__file__).resolve().parents[3]        # repo root (for refusal_llm_judge)
for p in (str(PKG), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
