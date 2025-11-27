# BERT CROSS-CONTENT EXPERIMENT
# # Train on UGC → Test on NGC  AND  Train on NGC → Test on UGC

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from scipy.stats import ttest_rel
import pingouin as pg

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# CONFIG 

DATA_UGC = "UGC_Master_Ex.csv"
DATA_NGC = "NGC_Master_Ex.csv"

MODELS_TO_RUN = [
    "distilbert-base-uncased",
    "distilbert-base-cased",
    "roberta-base",
    "bert-base-uncased",
    "bert-base-cased",
    "albert-base-v2",
    "google/electra-base-discriminator",
]

DOMAIN_MAP = {1: "Health", 2: "Politics", 3: "War"}
MAX_LENGTH = 128
BATCH_SIZE = 2
LR = 2e-5
EPOCHS = 3


# DATASET CLASS


class MisinformationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["claim"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }



# METRICS


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# TRAINING FUNCTION


def train_model(model_name, train_df, test_df):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = MisinformationDataset(train_df, tokenizer)
    test_ds = MisinformationDataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=f"./cross_content_models/{model_name}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    return preds



# EVALUATION WITH DOMAIN BREAKDOWN


def evaluate_model(model_name, df, preds):

    rows = []

    # ------- OVERALL -------
    rows.append({
        "model": model_name,
        "source": "OVERALL",
        "domain": "OVERALL",
        "accuracy": accuracy_score(df["label"], preds),
        "precision": precision_score(df["label"], preds),
        "recall": recall_score(df["label"], preds),
        "f1": f1_score(df["label"], preds),
    })

    # ------- BY CONTENT TYPE (UGC / NGC) -------
    for src in ["UGC", "NGC"]:
        df_s = df[df["source"] == src]
        if len(df_s) == 0:
            continue
        preds_s = preds[df["source"] == src]

        rows.append({
            "model": model_name,
            "source": src,
            "domain": "OVERALL",
            "accuracy": accuracy_score(df_s["label"], preds_s),
            "precision": precision_score(df_s["label"], preds_s),
            "recall": recall_score(df_s["label"], preds_s),
            "f1": f1_score(df_s["label"], preds_s)
        })

        # ------- DOMAIN WITHIN CONTENT TYPE -------
        for d in sorted(df_s["domain"].unique()):
            df_sd = df_s[df_s["domain"] == d]
            preds_sd = preds_s[df_s["domain"] == d]

            rows.append({
                "model": model_name,
                "source": src,
                "domain": DOMAIN_MAP[d],
                "accuracy": accuracy_score(df_sd["label"], preds_sd),
                "precision": precision_score(df_sd["label"], preds_sd),
                "recall": recall_score(df_sd["label"], preds_sd),
                "f1": f1_score(df_sd["label"], preds_sd)
            })

    return rows


# PLOT FUNCTIONS


def generate_plots(df):
    sns.set(style="whitegrid")

    
    # OVERALL BAR PLOT
    
    overall = df[(df["domain"] == "OVERALL") & (df["source"] == "OVERALL")]

    plt.figure(figsize=(8,5))
    sns.barplot(
        data=overall,
        x="model", y="f1",
        hue="train_source"
    )
    plt.xticks(rotation=45)
    plt.title("Cross-Content Overall F1 Performance")
    plt.tight_layout()
    plt.savefig("CrossContent_F1_Overall.png", dpi=300)
    plt.close()




# STATISTICS


def run_statistics(df):

    print("\n===== STATISTICAL TESTS =====\n")

   
    # Paired t-test on OVERALL F1
    
    ugc_to_ngc = df[
        (df["train_source"]=="UGC") &
        (df["test_source"]=="NGC") &
        (df["domain"]=="OVERALL") &
        (df["source"]=="OVERALL")
    ].sort_values("model")["f1"].values

    ngc_to_ugc = df[
        (df["train_source"]=="NGC") &
        (df["test_source"]=="UGC") &
        (df["domain"]=="OVERALL") &
        (df["source"]=="OVERALL")
    ].sort_values("model")["f1"].values

    t, p = ttest_rel(ugc_to_ngc, ngc_to_ugc)

    print(f"Paired t-test (UGC→NGC vs NGC→UGC): t={t:.3f}, p={p:.4f}")

    # ------------------------------
    # RM ANOVA — full domain test
    # ------------------------------

    domain_df = df[df["domain"].isin(["Health","Politics","War"])].copy()

    try:
        aov = pg.rm_anova(
            dv="f1",
            within=["train_source","domain"],
            subject="model",
            data=domain_df,
            detailed=True
        )
        print("\nRepeated-Measures ANOVA:\n", aov)
    except Exception as e:
        print("\nANOVA failed:", e)


# MAIN


def main():

    print("\nLoading data...")

    all_rows = []

    ugc = pd.read_csv(DATA_UGC)
    ngc = pd.read_csv(DATA_NGC)

    ugc["source"] = "UGC"
    ngc["source"] = "NGC"

    # ---- Clean labels/domains (safe version) ----
    for df in [ugc, ngc]:

        # Convert label/domain to strings and normalize junk
        df["label"] = (
            df["label"].astype(str).str.strip()
            .replace(["", "nan", "NaN", "None"], np.nan)
        )
        df["domain"] = (
            df["domain"].astype(str).str.strip()
            .replace(["", "nan", "NaN", "None"], np.nan)
        )

        # Drop rows with missing label or domain
        df.dropna(subset=["label", "domain"], inplace=True)

        # Now safe to convert to numeric
        df["label"] = df["label"].astype(float).astype(int)
        df["domain"] = df["domain"].astype(float).astype(int)

        # Reset indices after cleaning
    ugc_only = ugc.reset_index(drop=True)
    ngc_only = ngc.reset_index(drop=True)


    # TRAIN ON UGC → TEST ON NGC
    
    print("\n=== TRAIN ON UGC → TEST ON NGC ===")

    for model in MODELS_TO_RUN:
        preds = train_model(model, ugc_only, ngc_only)
        rows = evaluate_model(model, ngc_only, preds)
        for r in rows:
            r["train_source"] = "UGC"
            r["test_source"] = "NGC"
        all_rows.extend(rows)

    
    # TRAIN ON NGC → TEST ON UGC
    
    print("\n=== TRAIN ON NGC → TEST ON UGC ===")

    for model in MODELS_TO_RUN:
        preds = train_model(model, ngc_only, ugc_only)
        rows = evaluate_model(model, ugc_only, preds)
        for r in rows:
            r["train_source"] = "NGC"
            r["test_source"] = "UGC"
        all_rows.extend(rows)

    
    # SAVE RESULTS
    
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("8_BERT_cross_content_results.csv", index=False)

    print("\nSaved 8_BERT_cross_content_results.csv\n")

   
    # PLOTS + STATISTICS
    
    generate_plots(results_df)
    run_statistics(results_df)


if __name__ == "__main__":
    main()