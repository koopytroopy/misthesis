import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("2ANOVA_CxD_FS.csv")

GARNET = "#8A1538"
LIGHT_GREY = "#B1B1B1"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

domains = ["Health", "Politics", "War"]
models = df["model"].unique()

fig, axes = plt.subplots(
    2, 3,
    figsize=(23, 12),
    sharey=True
)

plt.subplots_adjust(
    wspace=0.35,
    hspace=0.40,
    top=0.85
)

axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]
    sub = df[(df["model"] == model) & (df["domain"].isin(domains))]

    ugc = sub[sub["source"] == "UGC"].set_index("domain").loc[domains]["f1"]
    ngc = sub[sub["source"] == "NGC"].set_index("domain").loc[domains]["f1"]

    x = np.arange(len(domains))

    ax.plot(x, ugc, "-o", color=GARNET, linewidth=3, markersize=10, label="UGC")
    ax.plot(x, ngc, "-o", color=LIGHT_GREY, linewidth=3, markersize=10, label="NGC")

    ax.set_title(model, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylim(0.0, 0.9)

# Shared big title
fig.suptitle(
    "Few-shot LLM-Level Interaction Plots: Content Type × Domain",
    fontsize=26,
    fontweight="bold"
)

# Shared y-axis label
fig.text(
    0.06, 0.5, "F1 Score",
    ha="center", va="center",
    fontsize=20,
    rotation="vertical"
)
# Shared X-axis label
fig.text(
    0.50, 0.04, "Domain",
    ha="center", va="center",
    fontsize=20
)

# Legend OUTSIDE plot + OUT OF TITLE
fig.legend(
    ["UGC", "NGC"],
    loc="upper right",
    bbox_to_anchor=(0.98, 0.97),
    fontsize=18,
    frameon=False
)

plt.show()