import os
import json
import re

# ---- 1. Set up the folder path ----
folder_path = "/Users/macbook/Desktop/Thesis_Work/sm_dataset"  # adjust if needed

# ---- 2. Collect English tweet texts from .jsonl files ----
all_tweets = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tweet = json.loads(line.strip())
                    # Check that 'text' exists
                    text = tweet.get("text")
                    langs = tweet.get("langs", [])
                    # Ensure langs is a list and contains 'eng'
                    if text and isinstance(langs, list) and "eng" in langs:
                        all_tweets.append(text)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid line in {filename}")
                except Exception as e:
                    print(f"⚠️ Error processing line in {filename}: {e}")

print(f"✅ Loaded {len(all_tweets)} English tweets from all files.")

# ---- 3. Define a simple rule for 'claim-like' statements ----
def is_claim_like(text):
    text_lower = text.lower()
    if any(x in text_lower for x in [
        "i think", "maybe", "probably", "in my opinion", "imo", "idk", "guess"
    ]):
        return False
    if re.search(r"\b(is|are|was|were|has|have|had|will|can|should|must)\b", text_lower):
        return True
    return False

# ---- 4. Filter only those tweets that look like factual claims ----
claims = [t for t in all_tweets if is_claim_like(t)]
print(f"💬 Found {len(claims)} potential factual claims (English only).")

# ---- 5. Save them into a text file ----
output_file = os.path.join(folder_path, "potential_claims_english.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for claim in claims:
        f.write(claim + "\n")

print(f"📝 Saved English claim-like tweets to: {output_file}")
