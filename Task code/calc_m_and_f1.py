import pandas as pd

# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------
df = pd.read_csv("llama70bZS_thesis_comprehensive_results_20251124_132912.csv")

# Normalize UGC/NGC labels just in case
df["dataset"] = df["dataset"].str.upper()

# ------------------------------------------------------------
# COMPUTE MEAN + SD FOR ACCURACY & F1
# ------------------------------------------------------------
metrics = (
    df.groupby("dataset")[["accuracy", "f1_score"]]
      .agg(["mean", "std"])
      .reset_index()
)

# ------------------------------------------------------------
# CLEAN PRINTOUT
# ------------------------------------------------------------
print("\n===== Mean + SD for UGC and NGC =====\n")
print(metrics.to_string(index=False))
