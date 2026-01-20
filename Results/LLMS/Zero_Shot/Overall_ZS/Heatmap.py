import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

GARNET = "#8A1538"
GREY = "#B1B1B1"

df = pd.read_csv("2ANOVA_CxD_FS.csv")

# Pivot so rows=models, columns=domain*source
heat_df = df.pivot_table(
    index="model",
    columns=["domain", "source"],
    values="f1"
)

plt.figure(figsize=(14, 6))
sns.heatmap(
    heat_df,
    annot=True,
    cmap="RdBu_r",
    center=0.7,
    fmt=".2f",
    cbar_kws={"label": "F1 Score"},
    linewidths=.5
)

plt.title("Heatmap of F1 Performance Across Models, Domains, and Content Types")
plt.xlabel("Domain × Source")
plt.ylabel("Model")
plt.tight_layout()
plt.show()
