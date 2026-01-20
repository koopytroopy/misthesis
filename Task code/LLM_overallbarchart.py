# This script creates a bar chart to visually present overall UGC vs NGC performance across llms
# define what csv to pull from at line 9
# define title at line 93

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "2ANOVA_CxD_FS.csv"  
df = pd.read_csv(csv_path)
df["domain"] = df["domain"].astype(str).str.strip().str.upper()
df["source"] = df["source"].astype(str).str.strip().str.upper()
df["model"] = df["model"].astype(str).str.strip()
subset = df[
    (df["domain"] == "OVERALL") &
    (df["source"].isin(["UGC", "NGC"]))
].copy()

model_order = [
    "Llama4b",
    "Llama70b",
    "Gemma4b",
    "Gemma27b",
    "Mistral7b",
    "Mixtral7b"
]
subset["model"] = pd.Categorical(
    subset["model"],
    categories=model_order,
    ordered=True
)

metric = "f1"
ugc_means = subset[subset["source"] == "UGC"][metric].values
ngc_means = subset[subset["source"] == "NGC"][metric].values
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

ugc_color = "#7A1438"
ngc_color = "#B3B3B3"
x = np.arange(len(model_order))
bar_width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
bars_ugc = ax.bar(
    x - bar_width / 2,
    ugc_means,
    bar_width,
    color=ugc_color,
    label="UGC"
)

bars_ngc = ax.bar(
    x + bar_width / 2,
    ngc_means,
    bar_width,
    color=ngc_color,
    label="NGC"
)

ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("F1 Score")
ax.set_xlabel("Model")

for bars in [bars_ugc, bars_ngc]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )
ax.legend(
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)
fig.suptitle(
    "Few-shot overall LLM performance by content type (F1 score)",
    fontsize=14
)
fig.subplots_adjust(top=0.82, bottom=0.18)
plt.savefig(
    "Figure_Overall_UGC_NGC_by_LLM.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
