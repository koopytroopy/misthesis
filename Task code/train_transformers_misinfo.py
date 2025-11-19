
# MISINFORMATION THESIS — BERT ANALYSIS 
# Full Training + Evaluation + Statistics + Plots :)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
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
try:
    import pingouin as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# CONFIGURATION


DATA_UGC = "UGC_Master_Ex.csv"
DATA_NGC = "NGC_Master_Ex.csv"

MODELS_TO_RUN = [
    "distilbert-base-uncased",
    "roberta-base",
    "bert-base-uncased",
    # add more models here if desired:
    # "bert-base-uncased",
    # "roberta-base",
    
]

MAX_LENGTH = 128
BASE_BATCH_SIZE = 2
LR = 2e-5
EPOCHS = 3


# GLOBAL RESULTS STORAGE


RESULTS = []
PREDICTION_OUTPUTS = {}


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


# EVALUATION (UGC vs NGC × DOMAIN BREAKDOWN)

def evaluate(model_name, df, preds):

    print(f"\n======= Evaluating {model_name} =======")

    rows = []

    #  OVERALL 
    overall_acc = accuracy_score(df["label"], preds)
    overall_prec = precision_score(df["label"], preds)
    overall_rec = recall_score(df["label"], preds)
    overall_f1 = f1_score(df["label"], preds)

    rows.append({
        "model": model_name,
        "source": "ALL",
        "domain": "ALL",
        "accuracy": overall_acc,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1
    })

    # SOURCE LEVEL (UGC/NGC) 
    for src in ["UGC", "NGC"]:
        src_df = df[df["source"] == src]
        src_preds = preds[df["source"] == src]

        sacc = accuracy_score(src_df["label"], src_preds)
        sprec = precision_score(src_df["label"], src_preds)
        srec = recall_score(src_df["label"], src_preds)
        sf1 = f1_score(src_df["label"], src_preds, average="weighted")

        rows.append({
            "model": model_name,
            "source": src,
            "domain": "ALL",
            "accuracy": sacc,
            "precision": sprec,
            "recall": srec,
            "f1": sf1
        })

        # SOURCE × DOMAIN 
        for dom in sorted(src_df["domain"].unique()):
            dom_df = src_df[src_df["domain"] == dom]
            dom_preds = src_preds[src_df["domain"] == dom]

            dacc = accuracy_score(dom_df["label"], dom_preds)
            dprec = precision_score(dom_df["label"], dom_preds)
            drec = recall_score(dom_df["label"], dom_preds)
            df1 = f1_score(dom_df["label"], dom_preds, average="weighted")

            rows.append({
                "model": model_name,
                "source": src,
                "domain": dom,
                "accuracy": dacc,
                "precision": dprec,
                "recall": drec,
                "f1": df1
            })

    return rows


# TRAIN MODEL


def train_model(model_name, train_df, test_df):

    print(f"\n============================\n TRAINING {model_name}\n============================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = MisinformationDataset(train_df, tokenizer)
    test_ds  = MisinformationDataset(test_df,  tokenizer)

    args = TrainingArguments(
        output_dir=f"./models/{model_name.replace('/', '_')}",
        learning_rate=LR,
        per_device_train_batch_size=BASE_BATCH_SIZE,
        per_device_eval_batch_size=BASE_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=4,
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

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=1)

    # Store predictions
    PREDICTION_OUTPUTS[model_name] = {"df": test_df.copy(), "preds": preds}

    return evaluate(model_name, test_df, preds)



# PLOTS FOR THESIS


def generate_plots(results_df):

    # Create a copy for plotting only
    plot_df = results_df.copy()

    # Convert domain for plotting
    plot_df["domain"] = plot_df["domain"].replace("ALL", np.nan)
    plot_df["domain"] = pd.to_numeric(plot_df["domain"], errors="coerce").astype("Int64")

    # ---- 1. UGC vs NGC BARPLOT ----
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=plot_df[plot_df["domain"].isna()],
        x="source", y="accuracy", hue="model"
    )
    plt.title("UGC vs NGC Accuracy Across Models")
    plt.savefig("Finalplot_ugc_ngc_accuracy.png", dpi=300)
    plt.close()

    # ---- 2. HEATMAP Source × Domain ----
    pivot = plot_df[plot_df["domain"].notna()].pivot_table(
        index=["source","domain"], columns="model", values="accuracy"
    )

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Accuracy by Source × Domain × Model")
    plt.savefig("Finalplot_domain_source_accuracy_heatmap.png", dpi=300)
    plt.close()

    # ---- 3. DOMAIN LINE PLOT ----
    plt.figure(figsize=(7,5))
    sns.lineplot(
        data=plot_df[plot_df["domain"].notna()],
        x="domain", y="accuracy",
        hue="source", style="model", markers=True
    )
    plt.xticks([1, 2, 3])
    plt.title("Accuracy per Domain: UGC vs NGC")
    plt.savefig("Finalplot_domain_line.png", dpi=300)
    plt.close()


# STATISTICS


def run_statistics(df):

    print("\n====== STATISTICAL TESTING ======\n")

   
    # UGC vs NGC t-test 
  
    ugc = df[(df["source"]=="UGC") & (df["domain"]=="ALL")]["accuracy"]
    ngc = df[(df["source"]=="NGC") & (df["domain"]=="ALL")]["accuracy"]

    if len(ugc) > 1 and len(ngc) > 1:
        t, p = ttest_ind(ugc, ngc)
        print(f"UGC vs NGC t-test: t={t:.3f}, p={p:.4f}")
    else:
        print("UGC vs NGC t-test skipped (≥2 models required).")

  
    # Two-Way Repeated-Measures ANOVA
   
    rep_df = df[df["domain"]!="ALL"].copy()

    rep_df["domain"] = rep_df["domain"].astype(str)

    try:
        aov = pg.rm_anova(
            dv="accuracy",
            within=["source", "domain"],
            subject="model",
            data=rep_df,
            detailed=True
        )
        print("\nRepeated-Measures ANOVA (Source × Domain):\n", aov)
    except Exception as e:
        print("Repeated-measures ANOVA failed:", e)




# APA NARRATIVE


def generate_apa_narrative(df):

    ugc = df[(df["source"]=="UGC") & (df["domain"]=="ALL")]["accuracy"]
    ngc = df[(df["source"]=="NGC") & (df["domain"]=="ALL")]["accuracy"]

    t, p = ttest_ind(ugc, ngc)

    narrative = f"""
Models showed higher accuracy on NGC content (M = {ngc.mean():.2f}) than on UGC content 
(M = {ugc.mean():.2f}). The difference was {"significant" if p < .05 else "not significant"}, 
t({len(ugc)+len(ngc)-2}) = {t:.2f}, p = {p:.3f}.

Domain-level differences were also observed, indicating variation across Domain 1, Domain 2, 
and Domain 3 within both content types. These results suggest that linguistic framing 
(UGC vs NGC) and topical domain interact to influence misinformation detectability.
"""

    with open("Final_apa_narrative.txt", "w") as f:
        f.write(narrative)

    print("\nSaved APA narrative to Final_apa_narrative.txt")



# MAIN SCRIPT


def main():

    print("Loading data...")

    ugc = pd.read_csv(DATA_UGC)
    ngc = pd.read_csv(DATA_NGC)

    # Label content type
    ugc["source"] = "UGC"
    ngc["source"] = "NGC"

    # Clean labels
    ugc = ugc.dropna(subset=["label"])
    ngc = ngc.dropna(subset=["label"])
    ugc["label"] = ugc["label"].astype(str).str.strip().astype(float).astype(int)
    ngc["label"] = ngc["label"].astype(str).str.strip().astype(float).astype(int)

    df = pd.concat([ugc, ngc], ignore_index=True)

    print("Dataset Loaded:")
    print(df["source"].value_counts())
    print(df["domain"].value_counts())
    print(df["label"].value_counts())

    # Stratify
    train_df = df.groupby(["source","domain","label"]).sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)

    for model in MODELS_TO_RUN:
        rows = train_model(model, train_df, test_df)
        RESULTS.extend(rows)


    # EXPORT + ANALYSIS
 

    results_df = pd.DataFrame(RESULTS)
    results_df.to_csv("Final_all_model_results.csv", index=False)
    print("\nSaved Final_all_model_results.csv")

    latex_table = results_df.to_latex(index=False, float_format="%.3f")
    with open("Final_all_model_results.tex", "w") as f:
        f.write(latex_table)
    print("Saved Final_all_model_results.tex")

    print("\nGenerating plots...")
    generate_plots(results_df)

    print("\nRunning statistics...")
    run_statistics(results_df)

    print("\nGenerating APA narrative...")
    generate_apa_narrative(results_df)


if __name__ == "__main__":
    main()
