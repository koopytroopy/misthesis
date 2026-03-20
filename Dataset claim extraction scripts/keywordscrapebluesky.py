import pandas as pd
import re
import os

# File path 
bluesky_path = "/Users/macbook/Desktop/Thesis_Work/claims_with_verifiable_label.csv"

if not os.path.exists(bluesky_path):
    raise FileNotFoundError(f"⚠️ File not found: {bluesky_path}")
print("✅ File found, loading Bluesky data...")

bluesky_df = pd.read_csv(bluesky_path)
print(f"Bluesky shape: {bluesky_df.shape}")
print(f"Columns: {list(bluesky_df.columns)}")

# Define keyword bundles
bluesky_keywords = {
    "COVID-19": [
        "covid", "coronavirus", "pandemic", "long covid", "covid deaths",
        "covid vaccine", "covid treatment", "covid cases", "covid test",
        "covid spread", "covid variant", "omicron", "delta variant",
        "covid lockdown", "covid restrictions", "mask mandate", "social distancing"
    ],
    "Vaccines": [
        "vaccine", "vaccination", "vaccinated", "mrna", "pfizer", "moderna",
        "booster", "immunization", "vaccine safety", "vaccine efficacy",
        "vaccine side effects", "vaccine mandate", "vaccine injury",
        "anti-vax", "herd immunity", "vaccine hesitancy", "jab"
    ],
    "Cancer_Medical": [
        "cancer", "tumor", "chemotherapy", "chemo", "radiation therapy",
        "cancer treatment", "cancer cure", "oncology", "carcinogen",
        "cancer research", "cancer drug", "remission", "metastatic",
        "biopsy", "cancer screening", "medical treatment"
    ],
    "Drug_Crisis": [
        "opioid", "fentanyl", "overdose", "drug addiction", "narcan",
        "naloxone", "opioid crisis", "prescription drugs", "drug epidemic",
        "harm reduction", "sackler", "purdue pharma", "drug policy",
        "addiction treatment", "substance abuse", "drug deaths"
    ],
    "Sexual_Gender_Health": [
        "abortion", "reproductive rights", "roe v wade", "roe wade",
        "gender affirming", "trans healthcare", "trans rights", "transgender",
        "puberty blockers", "hormone therapy", "gender identity",
        "lgbtq", "sexual health", "contraception", "birth control",
        "reproductive health", "pregnancy", "maternal health"
    ],
    "Elections": [
        "election", "voting", "ballot", "voter", "electoral",
        "midterm", "primary", "poll", "vote count", "voter fraud",
        "election fraud", "election security", "voter suppression",
        "gerrymandering", "voting rights", "mail-in ballot",
        "january 6", "jan 6", "capitol riot", "insurrection"
    ],
    "Immigration": [
        "immigration", "immigrant", "migrant", "refugee", "asylum",
        "border", "deportation", "ice", "daca", "dreamer",
        "undocumented", "illegal immigration", "border security",
        "immigration reform", "family separation", "border policy",
        "immigration enforcement", "sanctuary city"
    ],
    "Political_Figures": [
        "trump", "biden", "desantis", "harris", "pelosi", "mcconnell",
        "obama", "clinton", "sanders", "aoc", "pence", "schumer",
        "mccarthy", "president", "senator", "governor", "congress",
        "white house", "administration", "classified documents"
    ],
    "Ukraine_Russia": [
        "ukraine", "russia", "putin", "zelensky", "kyiv", "kiev",
        "war in ukraine", "russian invasion", "nato", "kremlin",
        "ukrainian", "russian troops", "war crimes", "sanctions russia",
        "ukraine aid", "donbas", "crimea", "russian military"
    ],
    "Israel_Palestine": [
        "israel", "palestine", "gaza", "hamas", "west bank",
        "israeli", "palestinian", "netanyahu", "idf", "tel aviv",
        "jerusalem", "ceasefire", "gaza strip", "two-state solution",
        "occupation", "settlements", "iron dome", "hostages"
    ]
}

# Helper function
def filter_claims(df, text_col, keywords):
    pattern = "|".join([re.escape(k) for k in keywords])
    mask = df[text_col].astype(str).str.contains(pattern, case=False, na=False)
    return df[mask].copy()

# Use the 'text' column
text_col = "text"
print(f"Using text column: {text_col}")

# Filter dataset
filtered_bluesky = {}

for domain, words in bluesky_keywords.items():
    subset = filter_claims(bluesky_df, text_col, words)
    subset = subset.copy()
    subset["Topic_Category"] = domain  
    print(f"{domain}: {len(subset)} claims found")
    filtered_bluesky[domain] = subset

# Combine and export
combined = pd.concat(filtered_bluesky.values(), ignore_index=True)
output_path = "/Users/macbook/Desktop/Thesis_Work/filtered_blusky_claims.xlsx"
combined.to_excel(output_path, index=False)

print(f" Done! Filtered and labeled results saved to:\n{output_path}")
