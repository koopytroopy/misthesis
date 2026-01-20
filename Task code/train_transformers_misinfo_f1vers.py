# This is the script for the finetuned bert baselines

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from scipy.stats import ttest_ind
import pingouin as pg
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def evaluate_model(model_name, df, preds):
    rows = []
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

def train_model(model_name, train_df, test_df):
    print(f"\n\n TRAINING {model_name}\n")
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
    PREDICTION_OUTPUTS[model_name] = {
        "df": test_df.copy(),
        "preds": preds
    }
    return evaluate_model(model_name, test_df, preds)


def generate_plots(df):
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

    domain_df = df[df["domain"].isin(["Health", "Politics", "War"])]
    pivot = domain_df.pivot_table(
        index=["source", "domain"], columns="model", values="f1"
    )

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako")
    plt.title("F1 Score by Domain × Source × Model")
    plt.savefig("F1_domain_heatmap.png", dpi=300)
    plt.close()
    plt.figure(figsize=(7,5))
    sns.lineplot(
        data=domain_df,
        x="domain", y="f1", hue="source", style="model", markers=True
    )
    plt.title("F1 Score per Domain by Source")
    plt.savefig("F1_domain_lineplot.png", dpi=300)
    plt.close()

def run_statistics(df):
    print("\n STATISTICS (F1) \n")
    ugc = df[(df["source"]=="UGC") & (df["domain"]=="OVERALL")]["f1"]
    ngc = df[(df["source"]=="NGC") & (df["domain"]=="OVERALL")]["f1"]
    if len(ugc) > 1:
        t, p = ttest_ind(ugc, ngc)
        print(f"UGC vs NGC (F1): t={t:.3f}, p={p:.4f}")
    else:
        print("UGC vs NGC t-test skipped (missing models).")
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

def generate_apa(results_df):
    print("\nGenerating APA narrative (F1-based RM-ANOVA)...")
    df = results_df[results_df["domain"] != "OVERALL"]
    aov_df = df.pivot_table(
        index="model",
        columns=["source", "domain"],
        values="f1"
    )
    long_df = df[["model", "source", "domain", "f1"]].copy()
    rm = pg.rm_anova(
        dv="f1",
        within=["source", "domain"],
        subject="model",
        data=long_df,
        detailed=True
    )

    source_row = rm.iloc[0]
    domain_row = rm.iloc[1]
    interaction_row = rm.iloc[2]
    def p_format(p):
        return "< .001" if p < 0.001 else f"= {p:.3f}"
    narrative = []
    narrative.append("A 2 (Source: UGC, NGC) × 3 (Domain: Health, Politics, War) "
                     "repeated-measures ANOVA was conducted on F1 scores to examine "
                     "differences in model performance across content types and topical domains.\n")
    narrative.append(
        f"There was a significant main effect of source, "
        f"F({int(source_row['ddof1'])}, {int(source_row['ddof2'])}) "
        f"= {source_row['F']:.2f}, p {p_format(source_row['p-unc'])}, "
        f"η²₍G₎ = {source_row['ng2']:.3f}. "
        "Models performed significantly differently on UGC versus NGC content.\n"
    )

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

    apa_text = "\n".join(narrative)
    with open("Final_APA_F1_Narrative.txt", "w") as f:
        f.write(apa_text)
    print("Saved APA narrative to Final_APA_F1_Narrative.txt")
    print("\n" + apa_text + "\n")

def main():

    print("Loading data...")

    ugc = pd.read_csv(DATA_UGC)
    ngc = pd.read_csv(DATA_NGC)
    ugc["source"] = "UGC"
    ngc["source"] = "NGC"

    for df in [ugc, ngc]:
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .replace(["nan", "NaN", "None", ""], np.nan)  
        )
        df.dropna(subset=["label"], inplace=True)
        df["label"] = df["label"].astype(float).astype(int)
        df["domain"] = (
            df["domain"]
            .astype(str)
            .str.strip()
            .replace(["nan", "NaN", "None", ""], np.nan)
        )
        df.dropna(subset=["domain"], inplace=True)
        df["domain"] = df["domain"].astype(float).astype(int)
    df = pd.concat([ugc, ngc], ignore_index=True)

    print("\nDataset after cleaning:")
    print(df["label"].value_counts())
    print(df["domain"].value_counts())
    print(df["source"].value_counts())

    train_df = df.groupby(["source","domain","label"]).sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)
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
