import pandas as pd

# Load your CSV file
df = pd.read_csv("/Users/macbook/Desktop/Thesis_Work/sm_dataset/potential_claims_english_with_IDandTrace.csv")

# Define domain keywords
domains = {
    "Health": ["vaccine", "covid", "virus", "health", "disease", "cancer", "doctor", "mask"],
    "Politics": ["election", "government", "president", "biden", "trump", "vote", "policy", "minister"],
    "War": ["war", "ukraine", "russia", "israel", "hamas", "army", "soldier", "conflict", "attack"]
}

# Match tweets to domains
def detect_domain(text):
    text_lower = text.lower()
    for domain, keywords in domains.items():
        if any(k in text_lower for k in keywords):
            return domain
    return None

df["domain"] = df["text"].apply(detect_domain)

# Keep only tweets in your three domains
domain_tweets = df.dropna(subset=["domain"])

print(f"✅ Found {len(domain_tweets)} tweets related to Health, Politics, or War.")
print(domain_tweets["domain"].value_counts())

# Save filtered results
domain_tweets.to_csv("/Users/macbook/Desktop/Thesis_Work/sm_dataset/claims_by_domain_filtering.csv", index=False)
