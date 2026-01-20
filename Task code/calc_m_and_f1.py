import pandas as pd

df = pd.read_csv("Final_F1_results.csv")

df["source"] = df["source"].str.upper()

metrics = (
    df.groupby("source")[["accuracy", "f1"]]
      .agg(["mean", "std"])
      .reset_index()
)

print("\n Mean + SD for UGC and NGC \n")
print(metrics.to_string(index=False))
