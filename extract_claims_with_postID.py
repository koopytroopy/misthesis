# adds function to extract postID
import os
import json
import re
import csv

# ---- 1. Set up the folder path ----
folder_path = "/Users/macbook/Desktop/Thesis_Work/sm_dataset"

# ---- 2. Collect English tweets (filename + post_id + text) ----
all_tweets = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tweet = json.loads(line.strip())
                    text = tweet.get("text")
                    langs = tweet.get("langs", [])
                    post_id = tweet.get("post_id", None)

                    # Only include English tweets with text
                    if text and isinstance(langs, list) and "eng" in langs:
                        all_tweets.append({
                            "filename": filename,
                            "post_id": post_id,
                            "text": text
                        })

                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid line in {filename}")
                except Exception as e:
                    print(f"⚠️ Error processing line in {filename}: {e}")

print(f"✅ Loaded {len(all_tweets)} English tweets from all files.")

# ---- 3. Simple rule for 'claim-like' statements ----
def is_claim_like(text):
    text_lower = text.lower()
    if any(x in text_lower for x in [
        "i think", "maybe", "probably", "in my opinion", "imo", "idk", "guess"
    ]):
        return False
    if re.search(r"\b(is|are|was|were|has|have|had|will|can|should|must)\b", text_lower):
        return True
    return False

# ---- 4. Filter only English tweets that sound factual ----
claims = [
    tweet for tweet in all_tweets
    if is_claim_like(tweet["text"])
]

print(f"💬 Found {len(claims)} potential factual claims (English only).")

# ---- 5. Save all claim-like tweets to CSV (filename + post_id + text) ----
output_file = os.path.join(folder_path, "potential_claims_english.csv")

with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "post_id", "text"])
    writer.writeheader()
    writer.writerows(claims)

print(f"📝 Saved English claim-like tweets (with file info) to: {output_file}")
