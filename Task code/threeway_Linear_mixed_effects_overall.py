import pandas as pd
import statsmodels.formula.api as smf

# Load your combined dataset
df = pd.read_csv("combined_results_twowayLME.csv")

# Normalize labels
df["dataset"] = df["dataset"].str.upper()   # UGC / NGC
df["condition"] = df["condition"].str.upper()  # ZS / FS

# Encode binary variables
df["dataset_binary"] = (df["dataset"] == "NGC").astype(int)   # UGC=0, NGC=1
df["shot_binary"] = (df["condition"] == "FS").astype(int)     # ZS=0, FS=1

# Mixed-effects model with interaction
model = smf.mixedlm(
    "f1_score ~ dataset_binary * shot_binary",
    df,
    groups=df["Model"]  # random intercept per model
).fit()

print(model.summary())
