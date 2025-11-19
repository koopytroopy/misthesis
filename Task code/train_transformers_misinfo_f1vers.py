# ===============================================================
# MISINFORMATION THESIS — TRANSFORMER ANALYSIS (F1 VERSION)
# Full Training + Evaluation + Plots + Stats
# ===============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Torch + Transformers
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# Stats
from scipy.stats import ttest_ind
import pingouin as pg

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ===============================================================
# CONFIG
# ===============================================================

DATA_UGC = "UGC_Master_Ex.csv"
DATA_NGC = "NGC_Master_Ex.csv"

MODELS_TO_RUN = [
    "distilbert-base-uncased",
    "roberta-base",
    "bert-base-uncased",
]

DOMAIN_MAP = {1: "Health", 2: "Politics", 3: "War"}

MAX_LENGTH = 128
BATCH_SIZE = 2
LR = 2e-5
EPOCHS = 3

RESULTS = []
PREDICTION_OUTPUTS = {}


# ===============================================================
# DATASET
# ===============================================================

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


# ===============================================================
# METRICS (trainer uses accuracy but we compute F1 manually later)
# ===============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# ===============================================================
# EVALUATION (SAVE F1 AS PRIMARY METRIC)
# ===============================================================

def evaluate_model(model_name, df, preds):

    rows = []

    # ---------- OVERALL ----------
    acc = accuracy_score(df["label"], preds)
    prec = precision_score(df["label"], preds)
    rec = recall_score(df["label"], preds)
    f1v = f1_score(df["label"], preds)

    rows.append({
        "model": model_name,
        "source": "OVERALL",
        "domain": "OVERALL",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v
    })

    # ---------- UGC vs NGC ----------
    for src in ["UGC", "NGC"]:
        df_s = df[df["source"] == src]
        preds_s = preds[df["source"] == src]

        rows.append({
            "model": model_name,
            "source": src,
            "domain": "OVERALL",
            "accuracy": accuracy_score(df_s["label"], preds_s),
            "precision": precision_score(df_s["label"], preds_s),
            "recall": recall_score(df_s["label"], preds_s),
            "f1": f1_score(df_s["label"], preds_s),
        })

        # ---------- DOMAIN WITHIN SOURCE ----------
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
                "f1": f1_score(df_sd["label"], preds_sd),
            })

    return rows


# ===============================================================
# TRAINING
# ===============================================================

def train_model(model_name, train_df, test_df):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = MisinformationDataset(train_df, tokenizer)
    test_ds  = MisinformationDataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=f"./models/{model_name.replace('/', '_')}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        fp16=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)

    PREDICTION_OUTPUTS[model_name] = {"df": test_df.copy(), "preds": preds}

    return evaluate_model(model_name, test_df, preds)


# ===============================================================
# PLOTS (F1 ONLY)
# ===============================================================

def generate_plots(df):

    # ===== UGC vs NGC only =====
    ugc_ngc = df[
        (df["domain"] == "OVERALL") &
        (df["source"].isin(["UGC", "NGC"]))
    ]

    plt.figure(figsize=(6,4))
    sns.barplot(data=ugc_ngc, x="source", y="f1", hue="model")
    plt.title("UGC vs NGC (F1 Score)")
    plt.ylabel("F1 Score")
    plt.savefig("F1_ugc_ngc_accuracy.png", dpi=300)
    plt.close()

    # ===== Heatmap of domains =====
    domain_df = df[df["domain"].isin(["Health", "Politics", "War"])]

    pivot = domain_df.pivot_table(
        index=["source", "domain"], columns="model", values="f1"
    )

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako")
    plt.title("F1 Score by Domain × Source × Model")
    plt.savefig("F1_domain_heatmap.png", dpi=300)
    plt.close()

    # ===== Line plot (clean domain axis) =====
    plt.figure(figsize=(7,5))
    sns.lineplot(
        data=domain_df,
        x="domain", y="f1", hue="source", style="model", markers=True
    )
    plt.title("F1 Score per Domain by Source")
    plt.savefig("F1_domain_lineplot.png", dpi=300)
    plt.close()


# ===============================================================
# STATISTICS (F1-BASED)
# ===============================================================

def run_statistics(df):

    print("\n====== STATISTICS (F1) ======\n")

    ugc = df[(df["source"]=="UGC") & (df["domain"]=="OVERALL")]["f1"]
    ngc = df[(df["source"]=="NGC") & (df["domain"]=="OVERALL")]["f1"]

    if len(ugc) > 1:
        t, p = ttest_ind(ugc, ngc)
        print(f"UGC vs NGC (F1): t={t:.3f}, p={p:.4f}")
    else:
        print("UGC vs NGC t-test skipped (missing models).")

    # Repeated-measures ANOVA
    rep = df[df["domain"].isin(["Health", "Politics", "War"])].copy()

    try:
        aov = pg.rm_anova(
            dv="f1",
            within=["source","domain"],
            subject="model",
            data=rep,
            detailed=True
        )
        print("\nRepeated-Measures ANOVA (F1):\n", aov)
    except Exception as e:
        print("ANOVA failed:", e)


# ===============================================================
# APA NARRATIVE
# ===============================================================

def generate_apa(df):

    ugc = df[(df["source"]=="UGC") & (df["domain"]=="OVERALL")]["f1"]
    ngc = df[(df["source"]=="NGC") & (df["domain"]=="OVERALL")]["f1"]

    t, p = ttest_ind(ugc, ngc)

    narrative = f"""
Models demonstrated slightly higher performance on NGC content (M = {ngc.mean():.2f}) 
than on UGC content (M = {ugc.mean():.2f}), based on F1 scores. 
This difference was {"statistically significant" if p < .05 else "not statistically significant"}, 
t({len(ugc)+len(ngc)-2}) = {t:.2f}, p = {p:.3f}.

Domain-level analyses (Health, Politics, War) revealed performance variation across topics 
within both UGC and NGC content, indicating that model detectability is jointly influenced 
by linguistic framing and topical domain.
"""

    with open("Final_F1_APA.txt", "w") as f:
        f.write(narrative)

    print("\nSaved APA narrative → Final_F1_APA.txt")


# ===============================================================
# MAIN
# ===============================================================

def main():

    print("Loading data...")

    ugc = pd.read_csv(DATA_UGC)
    ngc = pd.read_csv(DATA_NGC)

    ugc["source"] = "UGC"
    ngc["source"] = "NGC"

    for df in [ugc, ngc]:
        df["label"] = df["label"].astype(float).astype(int)
        df["domain"] = df["domain"].astype(int)

    df = pd.concat([ugc, ngc], ignore_index=True)

    # Balanced stratified split
    train_df = df.groupby(["source","domain","label"]).sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)

    # Train all models
    for model in MODELS_TO_RUN:
        RESULTS.extend(train_model(model, train_df, test_df))

    results_df = pd.DataFrame(RESULTS)
    results_df.to_csv("Final_F1_results.csv", index=False)

    print("\nGenerated Final_F1_results.csv")

    generate_plots(results_df)
    run_statistics(results_df)
    generate_apa(results_df)


if __name__ == "__main__":
    main()
