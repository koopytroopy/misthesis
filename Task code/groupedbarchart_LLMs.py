import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# uOttawa colors
GARNET = "#8A1538"
LIGHT_GREY = "#B1B1B1"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

# Load CSV
df = pd.read_csv("Final_f1_results.csv")


models = df["model"].unique()
domains = ["Health", "Politics", "War"]

fig, axes = plt.subplots(1, 3, figsize=(18, 10)) # change subplots 1st value to 2 for LLMS
axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]
    subset = df[df["model"] == model]

    ugc_vals = subset[subset["source"] == "UGC"].set_index("domain").loc[domains]["f1"].values
    ngc_vals = subset[subset["source"] == "NGC"].set_index("domain").loc[domains]["f1"].values

    x = np.arange(len(domains))
    width = 0.35

    bars_ugc = ax.bar(x - width/2, ugc_vals, width, color=GARNET, label="UGC")
    bars_ngc = ax.bar(x + width/2, ngc_vals, width, color=LIGHT_GREY, label="NGC")

    # --- NEW: Add F1 value labels above bars ---
    for bar in bars_ugc:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    for bar in bars_ngc:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_title(model, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylim(0, 1)

# Shared labels
fig.text(0.5, 0.03, "Domain", ha="center", fontsize=16)
fig.text(0.03, 0.5, "F1 Score", va="center", rotation="vertical", fontsize=16)

# Global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Source", fontsize=14, title_fontsize=14, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))

plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
plt.show()