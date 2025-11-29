import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# uOttawa thesis colors
GARNET = "#8A1538"      # UGC
LIGHT_GREY = "#B1B1B1"  # NGC

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})

# Load your results CSV
df = pd.read_csv("2ANOVA_CxD_ZS.csv")

# We only compare UGC vs NGC using OVERALL domain
subset = df[(df["domain"] == "OVERALL") &
            (df["source"].isin(["UGC", "NGC"]))].copy()

# Identify model order automatically


model_order = sorted(subset["model"].unique())
subset["model"] = pd.Categorical(subset["model"],
                                 categories=model_order,
                                 ordered=True)

# ------------------------------
# Extract means
# ------------------------------
ugc_means = subset[subset["source"] == "UGC"]["f1"].values
ngc_means = subset[subset["source"] == "NGC"]["f1"].values

# ------------------------------
# Generate synthetic SDs (optional)
# If you want real SDs, you must compute from repeated runs
# ------------------------------
ugc_sd = np.array([0.02] * len(model_order))
ngc_sd = np.array([0.03] * len(model_order))

# ------------------------------
# Build plot
# ------------------------------
x = np.arange(len(model_order))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 7))

# UGC bars
ax.bar(
    x - width/2, ugc_means, width,
    yerr=ugc_sd, capsize=6, color=GARNET,
    label="UGC"
)

# NGC bars
ax.bar(
    x + width/2, ngc_means, width,
    yerr=ngc_sd, capsize=6, color=LIGHT_GREY,
    label="NGC"
)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=20, ha="right")
ax.set_ylabel("F1 Score")
ax.set_xlabel("Model")
ax.set_title("Zero-Shot LLM Misinformation Detection: UGC vs NGC", fontsize=22)

# Legend
ax.legend(title="Source", fontsize=14, title_fontsize=14)

plt.tight_layout()
plt.show()
