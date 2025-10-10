import pandas as pd
import openai
import os
from tqdm import tqdm
import time

# Set key like:
# export OPENAI_API_KEY="your_key_here"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to input CSV
INPUT_PATH = "/Users/macbook/Desktop/Thesis_Work/sm_dataset/claims_by_domain_filtering.csv"
OUTPUT_PATH = "/Users/macbook/Desktop/Thesis_Work/sm_dataset/claims_with_verifiable_label.csv"

# Load CSV
df = pd.read_csv(INPUT_PATH)


texts = df["text"].astype(str).tolist()

# Define the classification function
def classify_verifiable(text, retries=3):
    prompt = f"""
    Determine if the following tweet contains a factual, verifiable claim
    (something that could be proven True or False by checking evidence)
    or if it is non-verifiable (e.g., opinion, vague, emotional, rhetorical).

    Respond with only one word: "Verifiable" or "Not Verifiable".

    Tweet: "{text}"
    """

    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            label = response.choices[0].message.content.strip()
            return label
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)
    return "Error"

# Process all tweets with progress bar
labels = []
for text in tqdm(texts, desc="Classifying tweets"):
    label = classify_verifiable(text)
    labels.append(label)

# Add new column to dataframe
df["verifiable_label"] = labels

# Save new CSV
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved labeled dataset to:\n{OUTPUT_PATH}")
