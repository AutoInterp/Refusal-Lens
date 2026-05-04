"""Two-panel figure for §5.5: per-condition projection + cosine alignment."""
import json
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/c/Users/Mahmoud Shabana/Documents/algoverse/Refusal-Lens")
RUN = REPO / "data/results/pipeline_runs/run_20260430_023247"

d = json.loads((RUN / "02b_stats/direction_alignment.json").read_text())
r_norm = d["metadata"]["r_hat_norm"]

CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: per-condition projection at pos=-2, in units of |r̂|
ax = axes[0]
proj = d["per_condition_proj"]
bare_proj = proj["bare"]["per_position"]["-2"]

# Order: bare, then for each class: jb, ctrl
xpos = []
labels = []
proj_vals = []
colors = []

xpos.append(0); labels.append("bare"); proj_vals.append(bare_proj/r_norm); colors.append("tab:gray")
for i, cls in enumerate(CLASSES):
    base_x = 1.5 + i * 2.5
    for j, (kind, color) in enumerate([("ctrl", "tab:blue"), ("jb", "tab:red")]):
        cond = f"{kind}_{cls}"
        xpos.append(base_x + j * 0.9)
        labels.append(f"{kind}_{cls[:5]}")
        proj_vals.append(proj[cond]["per_position"]["-2"] / r_norm)
        colors.append(color)

ax.bar(xpos, proj_vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(bare_proj/r_norm, color="black", linestyle="--", alpha=0.5, lw=1, label=f"bare = {bare_proj/r_norm:.2f}·|r̂|")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Refusal-direction projection / |r̂|  (at pos=−2)", fontsize=11)
ax.set_title("(a) JB conditions push the residual along r̂\nrelative to bare and ctrl", fontsize=11)
ax.grid(axis="y", alpha=0.3)
# Annotate the key finding
ax.text(0.98, 0.97,
        f"|r̂| = {r_norm:.0f}\n"
        f"Stage 06 +1·|r̂| flips 89/89 jb→refuse\n"
        f"Stage 06 −1·|r̂| flips 49/50 bare→comply",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color="tab:gray", label="bare"),
    plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="ctrl_*"),
    plt.Rectangle((0, 0), 1, 1, color="tab:red", label="jb_*"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9)

# Panel B: cosine similarity of derived JB directions with r̂, at pos=-2
ax = axes[1]
cos_jb = [d["per_class"][cls]["pos_minus_2"]["cos_r_hat_r_jb"] for cls in CLASSES]
cos_sem = [d["per_class"][cls]["pos_minus_2"]["cos_r_hat_r_jb_sem"] for cls in CLASSES]
mag_jb = [d["per_class"][cls]["pos_minus_2"]["mag_ratio_r_jb"] for cls in CLASSES]
mag_sem = [d["per_class"][cls]["pos_minus_2"]["mag_ratio_r_jb_sem"] for cls in CLASSES]

x = np.arange(len(CLASSES))
w = 0.36
b1 = ax.bar(x - w/2, cos_jb, w, label="r_jb = bare − jb (with prefix effect)",
            color="tab:purple", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, cos_sem, w, label="r_jb_sem = ctrl − jb (semantic only)",
            color="tab:orange", edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", lw=0.5)
ax.axhline(1.0, color="green", linestyle=":", lw=1, alpha=0.6, label="cos = 1 (parallel to r̂)")
ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15, fontsize=9)
ax.set_ylabel("cos(r_jb, r̂)  at pos=−2", fontsize=11)
ax.set_ylim(-1.0, 1.1)
ax.set_title("(b) Empirical JB directions are aligned with r̂ at pos=−2\n(cos +0.72 to +0.94 for r_jb; mixed for semantic-only)",
             fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left", fontsize=8)

# Annotate magnitudes above the bars
for i, (cj, mj) in enumerate(zip(cos_jb, mag_jb)):
    ax.text(i - w/2, cj + 0.04, f"|r_jb|/|r̂|\n={mj:.2f}",
            ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out = REPO / "figures" / "F7_refusal_direction_alignment.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")

# Also save into the run dir for self-contained reproduction
out2 = RUN / "02b_stats" / "direction_alignment.png"
plt.savefig(out2, dpi=180, bbox_inches="tight")
print(f"wrote {out2}")
