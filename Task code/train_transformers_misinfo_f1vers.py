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

    print(f"\n============================\n TRAINING {model_name}\n============================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = MisinformationDataset(train_df, tokenizer)
    test_ds = MisinformationDataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=f"./models/{model_name.replace('/', '_')}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",               # ← prevents checkpoint saving
        logging_steps=20,
        gradient_accumulation_steps=4,
        load_best_model_at_end=False      # ← required if save_strategy="no"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
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

    # Save predictions
    PREDICTION_OUTPUTS[model_name] = {
        "df": test_df.copy(),
        "preds": preds
    }

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

def generate_apa(results_df):

    print("\nGenerating APA narrative (F1-based RM-ANOVA)...")

    # Filter out the OVERALL rows
    df = results_df[results_df["domain"] != "OVERALL"]

    # Reformat for Pingouin
    aov_df = df.pivot_table(
        index="model",
        columns=["source", "domain"],
        values="f1"
    )

    # Melt back into long format
    long_df = df[["model", "source", "domain", "f1"]].copy()

    # Run repeated-measures ANOVA
    rm = pg.rm_anova(
        dv="f1",
        within=["source", "domain"],
        subject="model",
        data=long_df,
        detailed=True
    )

    # Extract values
    source_row = rm.iloc[0]
    domain_row = rm.iloc[1]
    interaction_row = rm.iloc[2]

    # Helper for APA-style p-values
    def p_format(p):
        return "< .001" if p < 0.001 else f"= {p:.3f}"

    # Construct APA narrative
    narrative = []

    narrative.append("A 2 (Source: UGC, NGC) × 3 (Domain: Health, Politics, War) "
                     "repeated-measures ANOVA was conducted on F1 scores to examine "
                     "differences in model performance across content types and topical domains.\n")

    # --- SOURCE EFFECT ---
    narrative.append(
        f"There was a significant main effect of source, "
        f"F({int(source_row['ddof1'])}, {int(source_row['ddof2'])}) "
        f"= {source_row['F']:.2f}, p {p_format(source_row['p-unc'])}, "
        f"η²₍G₎ = {source_row['ng2']:.3f}. "
        "Models performed significantly differently on UGC versus NGC content.\n"
    )

    # --- DOMAIN EFFECT ---
    if domain_row['p-unc'] < 0.05:
        narrative.append(
            f"There was also a significant main effect of domain, "
            f"F({int(domain_row['ddof1'])}, {int(domain_row['ddof2'])}) "
            f"= {domain_row['F']:.2f}, p {p_format(domain_row['p-unc'])}, "
            f"η²₍G₎ = {domain_row['ng2']:.3f}, "
            "indicating that performance varied across Health, Politics, and War claims.\n"
        )
    else:
        narrative.append(
            f"The main effect of domain was not statistically significant, "
            f"F({int(domain_row['ddof1'])}, {int(domain_row['ddof2'])}) "
            f"= {domain_row['F']:.2f}, p {p_format(domain_row['p-unc'])}, "
            f"η²₍G₎ = {domain_row['ng2']:.3f}.\n"
        )

    # --- INTERACTION EFFECT ---
    if interaction_row['p-unc'] < 0.05:
        narrative.append(
            f"There was a significant Source × Domain interaction, "
            f"F({int(interaction_row['ddof1'])}, {int(interaction_row['ddof2'])}) "
            f"= {interaction_row['F']:.2f}, p {p_format(interaction_row['p-unc'])}, "
            f"η²₍G₎ = {interaction_row['ng2']:.3f}, "
            "suggesting that source differences (UGC vs. NGC) varied across the three domains.\n"
        )
    else:
        narrative.append(
            f"The Source × Domain interaction was not statistically significant, "
            f"F({int(interaction_row['ddof1'])}, {int(interaction_row['ddof2'])}) "
            f"= {interaction_row['F']:.2f}, p {p_format(interaction_row['p-unc'])}, "
            f"η²₍G₎ = {interaction_row['ng2']:.3f}.\n"
        )

    # Write to file
    apa_text = "\n".join(narrative)

    with open("Final_APA_F1_Narrative.txt", "w") as f:
        f.write(apa_text)

    print("Saved APA narrative to Final_APA_F1_Narrative.txt")
    print("\n" + apa_text + "\n")

# ===============================================================
# MAIN
# ===============================================================

def main():

    print("Loading data...")

    ugc = pd.read_csv(DATA_UGC)
    ngc = pd.read_csv(DATA_NGC)

    # Tag source
    ugc["source"] = "UGC"
    ngc["source"] = "NGC"

    # -------------------------------
    # FULL CLEANING FOR LABEL & DOMAIN
    # -------------------------------
    for df in [ugc, ngc]:

        # Convert label to clean string
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .replace(["nan", "NaN", "None", ""], np.nan)  # convert junk labels to NaN
        )

        # Drop invalid label rows
        df.dropna(subset=["label"], inplace=True)

        # Convert label → float → int
        df["label"] = df["label"].astype(float).astype(int)

        # Domain cleaning
        df["domain"] = (
            df["domain"]
            .astype(str)
            .str.strip()
            .replace(["nan", "NaN", "None", ""], np.nan)
        )
        df.dropna(subset=["domain"], inplace=True)
        df["domain"] = df["domain"].astype(float).astype(int)

    # -------------------------------
    # MERGE CLEANED DATASETS
    # -------------------------------
    df = pd.concat([ugc, ngc], ignore_index=True)

    print("\nDataset after cleaning:")
    print(df["label"].value_counts())
    print(df["domain"].value_counts())
    print(df["source"].value_counts())

    # -------------------------------
    # STRATIFIED TRAIN/TEST SPLIT
    # -------------------------------
    train_df = df.groupby(["source","domain","label"]).sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)

    # -------------------------------
    # TRAIN ALL MODELS
    # -------------------------------
    for model in MODELS_TO_RUN:
        RESULTS.extend(train_model(model, train_df, test_df))

    # -------------------------------
    # SAVE RESULTS
    # -------------------------------
    results_df = pd.DataFrame(RESULTS)
    results_df.to_csv("Final_F1_results.csv", index=False)
    print("\nGenerated Final_F1_results.csv")

    # -------------------------------
    # ANALYSIS & PLOTS
    # -------------------------------
    generate_plots(results_df)
    run_statistics(results_df)
    generate_apa(results_df)


if __name__ == "__main__":
    main()
