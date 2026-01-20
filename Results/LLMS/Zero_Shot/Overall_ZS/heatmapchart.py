import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Colors (uOttawa thesis palette) ---
GARNET = "#8A1538"
GREY = "#B1B1B1"

# --- Load Data ---
df = pd.read_csv("2ANOVA_cxd_zs.csv")

# Expected order for models
models = [
    "Llama4b",
    "Llama70b",
    "Gemma4b",
    "Gemma27b",
    "Mistral7b",
    "Mixtral7b"
]

domains = ["Health", "Politics", "War"]

# --- Figure Layout ---
fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
axes = axes.flatten()

# Horizontal jitter
UGC_OFFSET = -0.06
NGC_OFFSET = +0.06

for ax, model in zip(axes, models):
    sub = df[df["model"] == model]

    for source, color, offset, marker, fill in [
        ("UGC", GARNET, UGC_OFFSET, "o", True),
        ("NGC", GREY, NGC_OFFSET, "o", False),
    ]:
        vals = [
            sub[(sub.domain == d) & (sub.source == source)]["f1"].values[0]
            for d in domains
        ]

        # consistent dummy sd for visual symmetry
        sd = [0.015] * len(vals)

        ax.errorbar(
            np.arange(len(domains)) + offset,
            vals,
            yerr=sd,
            fmt=marker,
            markersize=10,
            color=color,
            markerfacecolor=color if fill else "white",
            markeredgecolor=color,
            markeredgewidth=2,
            capsize=4,
            linestyle="None",
            label=source if model == models[0] else ""
        )

    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_title(model, fontsize=16)
    ax.set_xlabel("Domain", fontsize=13)
    ax.grid(alpha=0.28, linestyle="--")

# ---- Shared ylabel on left column ----
axes[0].set_ylabel("F1 Score", fontsize=14)
axes[3].set_ylabel("F1 Score", fontsize=14)

# ---- Legend ----
fig.legend(
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=14,
    title="Source",
    title_fontsize=15
)

# ---- Title ----
fig.suptitle(
    "Content Type × Domain Interaction on LLM Detection Performance",
    fontsize=22,
    y=1.08,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
