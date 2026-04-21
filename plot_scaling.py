"""
Reproduce scaling_law_replication_lines.png.
Reads "ours" results from results_overall CSVs; paper Gemma-3 values are
taken directly from original_code/visualization/models/datasets/no_qwen_absolute.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ---------------------------------------------------------------------------
# Model parameter sizes (billions) — nominal sizes from model names, not
# total parameter counts (which include embeddings and differ from the name).
# Using nominal sizes matches the convention in the original paper's figures.
# ---------------------------------------------------------------------------
PARAM_SIZES = {
    "gemma-3-270m-it": 0.27,
    "gemma-3-1b-it":   1.0,
    "gemma-3-4b-it":   4.0,
    "gemma-3-12b-it":  12.0,
    "gemma-3-27b-it":  27.0,
    "Qwen3.5-0.8B":    0.8,
    "Qwen3.5-2B":      2.0,
    "Qwen3.5-4B":      4.0,
    "Qwen3.5-9B":      9.0,
    "Qwen3.5-27B":     27.0,
    "gemma-4-E2B-it":  2.0,
    "gemma-4-E4B-it":  4.0,
    "gemma-4-31B-it":  31.0,
}

# ---------------------------------------------------------------------------
# Paper Gemma-3 values (from original_code visualization dataset)
# ---------------------------------------------------------------------------
PAPER_GEMMA3_CSV = "original_code/visualization/models/datasets/no_qwen_absolute.csv"
paper_df = pd.read_csv(PAPER_GEMMA3_CSV, header=None,
                       names=["model", "size", "auroc", "layers", "best_layer", "dataset"])
paper_df = paper_df[paper_df["model"].str.contains("gemma-3", case=False, na=False)].copy()
paper_df["model"] = paper_df["model"].str.strip()
paper_df["size_b"] = paper_df["model"].str.strip().map(
    lambda m: PARAM_SIZES.get(m.lower().replace("b-it", "b-it"), None)
)
# map using lowercase-normalised model names
paper_gemma3 = {}
for _, row in paper_df.iterrows():
    key = row["model"].lower().replace("b-it", "b-it")
    matched = next((v for k, v in PARAM_SIZES.items() if k.lower() == key), None)
    if matched:
        paper_gemma3[matched] = float(row["auroc"])

# Manual fallback in case name normalisation mismatches
if not paper_gemma3:
    paper_gemma3 = {0.27: 0.1503, 1.0: 0.1798, 4.0: 0.2459, 12.0: 0.2940, 27.0: 0.1985}

# ---------------------------------------------------------------------------
# "Ours" results — prefer results/results.csv (current run); fall back to
# results_overall if it doesn't exist yet. Deduplicate by keeping the last
# entry per model (run.py appends, so later rows are more recent).
# ---------------------------------------------------------------------------
results_csv = "results/results.csv"
if os.path.exists(results_csv):
    ours_df = pd.read_csv(results_csv)
    ours_df = ours_df.drop_duplicates(subset="model", keep="last")
else:
    dfs = []
    for f in ["results_overall/results_a40.csv", "results_overall/results_a100.csv"]:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    ours_df = pd.concat(dfs, ignore_index=True)
ours_df["params_b"] = ours_df["model"].map(PARAM_SIZES)

our_gemma3  = ours_df[ours_df["model"].str.startswith("gemma-3")].dropna(subset=["params_b"])
our_qwen35  = ours_df[ours_df["model"].str.startswith("Qwen3.5")].dropna(subset=["params_b"])
our_gemma4  = ours_df[ours_df["model"].str.startswith("gemma-4")].dropna(subset=["params_b"])

our_gemma3  = our_gemma3.sort_values("params_b")
our_qwen35  = our_qwen35.sort_values("params_b")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7.5))
fig.patch.set_facecolor("#f5f0e5")
ax.set_facecolor("#f5f0e5")

GOLD  = "#c8960c"
BLUE  = "#2060c8"
RED   = "#c82020"
GREEN = "#2ca02c"
LW    = 2.5
MS    = 10

# Gemma-3 (paper)
px = sorted(paper_gemma3.keys())
py = [paper_gemma3[x] for x in px]
ax.plot(px, py, color=GOLD, linewidth=LW, zorder=2)
ax.scatter(px, py, color=GOLD, marker="P", s=MS**2, zorder=3, label="Gemma-3 (paper)")

# Gemma-3 (ours)
ax.plot(our_gemma3["params_b"], our_gemma3["AUROC"], color=BLUE, linewidth=LW, zorder=2)
ax.scatter(our_gemma3["params_b"], our_gemma3["AUROC"],
           color=BLUE, marker="o", s=MS**2, zorder=3, label="Gemma-3 (ours)")

# Qwen-3.5 (ours)
ax.plot(our_qwen35["params_b"], our_qwen35["AUROC"], color=RED, linewidth=LW, zorder=2)
ax.scatter(our_qwen35["params_b"], our_qwen35["AUROC"],
           color=RED, marker="s", s=MS**2, zorder=3, label="Qwen-3.5 (ours)")

# Gemma-4 (ours)
our_gemma4 = our_gemma4.sort_values("params_b")
ax.plot(our_gemma4["params_b"], our_gemma4["AUROC"], color=GREEN, linewidth=LW, zorder=2)
ax.scatter(our_gemma4["params_b"], our_gemma4["AUROC"],
           color=GREEN, marker="^", s=MS**2, zorder=3, label="Gemma-4 (ours)")

ax.set_xscale("log")
ax.set_xlim(0.18, 40)
ax.set_ylim(0, 0.43)

ax.xaxis.set_major_locator(ticker.LogLocator(base=2))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}"
))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.set_xlabel("Model Parameters (Billions)", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("AUROC Absolute Distance from 0.5", fontsize=15, fontweight="bold", labelpad=10)

ax.tick_params(axis="both", labelsize=12)
ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="#cccccc")
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#dddddd")

legend = ax.legend(fontsize=12, framealpha=0.85, loc="upper left")

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color("#888888")

plt.tight_layout()
out = "scaling_law_replication_lines.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out}")
