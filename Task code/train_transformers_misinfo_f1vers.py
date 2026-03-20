
# This is the script for few-shot BERT evaluation
# Modified from original to save item level predictions in a single CSV for GLMM
# Call on model at line 28


import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

DESCRIPTIVE_RESULTS = []

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


def train_and_predict(model_name, train_df, test_df):

    print(f"\n===== TRAINING {model_name} =====\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = MisinformationDataset(train_df, tokenizer)
    test_ds = MisinformationDataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir=f"./models/{model_name.replace('/', '_')}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_strategy="no",
        logging_steps=20,
        gradient_accumulation_steps=4,
        load_best_model_at_end=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )

    trainer.train()


    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=1)

    item_df = test_df.copy().reset_index(drop=True)
    item_df["prediction"] = preds
    item_df["correct"] = (item_df["prediction"] == item_df["label"]).astype(int)
    item_df["model"] = model_name

    os.makedirs("item_level_predictions", exist_ok=True)

    save_path = f"item_level_predictions/{model_name.replace('/', '_')}_item_predictions.csv"
    item_df.to_csv(save_path, index=False)

    print(f"Saved item-level predictions → {save_path}")

    compute_descriptives(model_name, item_df)



def compute_descriptives(model_name, df):

    # Overall
    DESCRIPTIVE_RESULTS.append({
        "model": model_name,
        "source": "OVERALL",
        "domain": "OVERALL",
        "accuracy": accuracy_score(df["label"], df["prediction"]),
        "precision": precision_score(df["label"], df["prediction"]),
        "recall": recall_score(df["label"], df["prediction"]),
        "f1": f1_score(df["label"], df["prediction"])
    })

    # By source
    for src in ["UGC", "NGC"]:
        df_s = df[df["source"] == src]

        DESCRIPTIVE_RESULTS.append({
            "model": model_name,
            "source": src,
            "domain": "OVERALL",
            "accuracy": accuracy_score(df_s["label"], df_s["prediction"]),
            "precision": precision_score(df_s["label"], df_s["prediction"]),
            "recall": recall_score(df_s["label"], df_s["prediction"]),
            "f1": f1_score(df_s["label"], df_s["prediction"])
        })

        # By domain
        for d in sorted(df_s["domain"].unique()):
            df_sd = df_s[df_s["domain"] == d]

            DESCRIPTIVE_RESULTS.append({
                "model": model_name,
                "source": src,
                "domain": DOMAIN_MAP[d],
                "accuracy": accuracy_score(df_sd["label"], df_sd["prediction"]),
                "precision": precision_score(df_sd["label"], df_sd["prediction"]),
                "recall": recall_score(df_sd["label"], df_sd["prediction"]),
                "f1": f1_score(df_sd["label"], df_sd["prediction"])
            })


def main():

    print("Loading and cleaning data...")

    ugc = pd.read_csv(DATA_UGC)
    ngc = pd.read_csv(DATA_NGC)

    ugc["source"] = "UGC"
    ngc["source"] = "NGC"

    df = pd.concat([ugc, ngc], ignore_index=True)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    df["domain"] = pd.to_numeric(df["domain"], errors="coerce")
    df = df[df["domain"].isin([1, 2, 3])]
    df["domain"] = df["domain"].astype(int)

    df["item_id"] = df.index

    print("\nDataset summary:")
    print("Source counts:\n", df["source"].value_counts())
    print("Domain counts:\n", df["domain"].value_counts())
    print("Label counts:\n", df["label"].value_counts())


    df["strat_key"] = (
        df["source"].astype(str) + "_" +
        df["domain"].astype(str) + "_" +
        df["label"].astype(str)
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["strat_key"]
    )

    train_df = train_df.drop(columns=["strat_key"])
    test_df = test_df.drop(columns=["strat_key"])

    print("\nTrain/Test sizes:", len(train_df), len(test_df))


    for model in MODELS_TO_RUN:
        train_and_predict(model, train_df, test_df)



    print("\nCombining item-level files...")

    files = glob.glob("item_level_predictions/*.csv")
    master = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    master.to_csv("BERT_item_level_master.csv", index=False)
    print("Saved → BERT_item_level_master.csv")


    desc_df = pd.DataFrame(DESCRIPTIVE_RESULTS)
    desc_df.to_csv("BERT_descriptive_metrics.csv", index=False)
    print("Saved → BERT_descriptive_metrics.csv")

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
