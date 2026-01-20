import pandas as pd

# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------
df = pd.read_csv("Final_F1_results.csv")

# Normalize UGC/NGC labels just in case
df["source"] = df["source"].str.upper()

# ------------------------------------------------------------
# COMPUTE MEAN + SD FOR ACCURACY & F1
# ------------------------------------------------------------
metrics = (
    df.groupby("source")[["accuracy", "f1"]]
      .agg(["mean", "std"])
      .reset_index()
)

# ------------------------------------------------------------
# CLEAN PRINTOUT
# ------------------------------------------------------------
print("\n===== Mean + SD for UGC and NGC =====\n")
print(metrics.to_string(index=False))
