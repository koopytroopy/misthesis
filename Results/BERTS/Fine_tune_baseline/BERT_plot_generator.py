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
df = pd.read_csv("Final_F1_results.csv")

# Filter for OVERALL domain for UGC vs NGC comparison
subset = df[(df["domain"] == "OVERALL") & 
            (df["source"].isin(["UGC", "NGC"]))].copy()

# Order models
model_order = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
subset["model"] = pd.Categorical(subset["model"], categories=model_order, ordered=True)

# Prepare bar positions
x = np.arange(len(model_order))
width = 0.35

# Extract values
ugc_means = subset[subset["source"] == "UGC"]["f1"].values
ngc_means = subset[subset["source"] == "NGC"]["f1"].values

# Synthetic standard deviations based on your screenshot
ugc_sd = np.array([0.015, 0.035, 0.010])
ngc_sd = np.array([0.040, 0.050, 0.045])

x = np.arange(len(model_order))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

# Plot with synthetic error bars
ax.bar(x - width/2, ugc_means, width,
       yerr=ugc_sd, capsize=6, color=GARNET, label="UGC")

ax.bar(x + width/2, ngc_means, width,
       yerr=ngc_sd, capsize=6, color=LIGHT_GREY, label="NGC")

# Labels & styling
ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=15)
ax.set_ylabel("F1 Score")
ax.set_xlabel("BERT Model")
ax.set_title("BERT-based model detection performance by content source type", fontsize=22)
ax.legend(title="Source", fontsize=14, title_fontsize=14)

plt.tight_layout()
plt.show()