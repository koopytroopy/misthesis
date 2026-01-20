# This script creates a grouped barchart to visually present the findings of a 2x3 ANOVA
# Model performance on UGC vs NGC across domains
# Line 10 is where you define what CSV file to call
# Line 102 is where you define the figure title

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_path = "2ANOVA_CxD_FS.csv"
df = pd.read_csv(csv_path)
df = df[df["domain"] != "OVERALL"]
model_order = [
    "Llama4b", "Gemma4b", "Mistral7b",
    "Llama70b", "Gemma27b", "Mixtral7b"
]
domains = ["Health", "Politics", "War"]
sources = ["UGC", "NGC"]  

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
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()
bar_width = 0.35
x = np.arange(len(domains))

for ax, model in zip(axes, model_order):
    model_df = df[df["model"] == model]

    ugc_vals = (
        model_df[model_df["source"] == "UGC"]
        .set_index("domain")
        .loc[domains]["f1"]
        .values
    )

    ngc_vals = (
        model_df[model_df["source"] == "NGC"]
        .set_index("domain")
        .loc[domains]["f1"]
        .values
    )

    bars_ugc = ax.bar(
        x - bar_width / 2,
        ugc_vals,
        bar_width,
        color=ugc_color,
        label="UGC"
    )

    bars_ngc = ax.bar(
        x + bar_width / 2,
        ngc_vals,
        bar_width,
        color=ngc_color,
        label="NGC"
    )
    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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

fig.text(0.04, 0.5, "F1 Score", va="center", rotation="vertical", fontsize=12)
fig.text(0.5, 0.06, "Domain", ha="center", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.95)
)
fig.subplots_adjust(top=0.86)
fig.suptitle(
    "Few-shot: LLM performance across content type and domain (F1 Score)",
    fontsize=14
)

plt.savefig("Fewshot_Figure_Interaction_ContentType_Domain.png", dpi=300)
plt.show()
