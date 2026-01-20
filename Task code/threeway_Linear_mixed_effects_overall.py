import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("combined_results_twowayLME.csv")

df["dataset"] = df["dataset"].str.upper()   
df["condition"] = df["condition"].str.upper()  
df["dataset_binary"] = (df["dataset"] == "NGC").astype(int)   
df["shot_binary"] = (df["condition"] == "FS").astype(int)     

model = smf.mixedlm(
    "f1_score ~ dataset_binary * shot_binary",
    df,
    groups=df["Model"]  
).fit()

print(model.summary())
