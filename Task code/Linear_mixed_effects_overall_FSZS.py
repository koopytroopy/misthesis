import pandas as pd
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv("zerohot_NGC:UGC_linear_mixed_effects.csv")

# Normalize dataset labels just in case
df["dataset"] = df["dataset"].str.upper()

# Encode UGC = 0, NGC = 1
df["dataset_binary"] = (df["dataset"] == "NGC").astype(int)

# ------------------------------------------------------------
# Mixed-Effects Model
# ------------------------------------------------------------
# f1_score ~ dataset_binary + (1 | Model)
model = smf.mixedlm(
    "f1_score ~ dataset_binary",
    df,
    groups=df["Model"]     # random intercept per model
).fit()

print("\n===== MIXED EFFECTS MODEL RESULTS =====\n")
print(model.summary())
