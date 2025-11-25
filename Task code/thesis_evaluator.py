import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    classification_report
)
from scipy import stats
import numpy as np
from time import sleep
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

print("Thesis Misinformation Testing")
print(f"Experiment started: {datetime.now()}")


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

CONFIG = {
    "ugc_file": "/Users/vanessabelanger/Desktop/misthesis/task code/UGC_Master_Ex.csv",
    "ngc_file": "/Users/vanessabelanger/Desktop/misthesis/task code/NGC_Master_Ex.csv",
    "model_name": "mistral-7b-instruct-v0.2",
    "api_url": "http://127.0.0.1:1234/v1/completions",
    "n_runs": 5,
    "random_seeds": [42, 123, 456, 789, 999]
}


# ------------------------------------------------------------
# MAIN EVALUATOR CLASS
# ------------------------------------------------------------

class ThesisEvaluator:
    def __init__(self, config):
        self.config = config
        self.all_results = []
        self.run_data_storage = {}

    # --------------------------------------------------------
    # PROMPT BUILDER
    # --------------------------------------------------------

    def generate_prompt(self, claim, domain):
        return (
            f"Classify the following {domain} claim as 'true' or 'false'. "
            f"Respond with only one word (true/false).\n\n"
            f"Claim: \"{claim}\"\nAnswer:"
        )

    # --------------------------------------------------------
    # ZERO-SHOT CLASSIFIER (LLM CALL)
    # --------------------------------------------------------

    def classify_claim(self, claim, domain, run_id=0):
        if pd.isna(claim) or not isinstance(claim, str):
            return None

        prompt = self.generate_prompt(claim, domain)

        try:
            response = requests.post(
                self.config["api_url"],
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.config["model_name"],
                    "prompt": prompt,
                    "max_tokens": 5,
                    "temperature": 0.1,
                    "stop": ["\n"],
                    "seed": self.config["random_seeds"][run_id]
                }
            )

            if response.status_code != 200:
                return None

            raw = response.json()["choices"][0]["text"]
            if raw is None:
                return None

            output = raw.strip().lower()

            cleaned = (
                output.replace(".", "")
                      .replace(",", "")
                      .replace(":", "")
                      .replace(";", "")
                      .strip()
            )
            tokens = cleaned.split()

            if tokens:
                if tokens[0] == "true":
                    return 1
                if tokens[0] == "false":
                    return 0

            window = tokens[:5]
            if "true" in window:
                return 1
            if "false" in window:
                return 0

            import re
            if re.search(r"\btrue\b", cleaned):
                return 1
            if re.search(r"\bfalse\b", cleaned):
                return 0

            return None

        except:
            return None

    # --------------------------------------------------------
    # DATA SPLITTING (STRATIFIED BY DOMAIN + LABEL)
    # --------------------------------------------------------

    def create_stratified_split(self, df, test_size=0.3, random_state=42):
        from sklearn.model_selection import train_test_split

        df = df.copy()
        df["stratify_col"] = df["domain"].astype(str) + "_" + df["label"].astype(str)

        print("\nDataset Distribution Before Split:")
        print(df.groupby(["domain", "label"]).size().unstack(fill_value=0))
        print(f"\nTotal samples: {len(df)}")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["stratify_col"]
        )

        train_df = train_df.drop(columns=["stratify_col"])
        test_df = test_df.drop(columns=["stratify_col"])

        print(f"\nTrain Set ({len(train_df)} samples):")
        print(train_df.groupby(["domain", "label"]).size().unstack(fill_value=0))

        print(f"\nTest Set ({len(test_df)} samples):")
        print(test_df.groupby(["domain", "label"]).size().unstack(fill_value=0))

        return train_df, test_df

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    def generate_baseline_predictions(self, train_df, test_df, method="random"):
        np.random.seed(42)

        if method == "random":
            return np.random.choice([0, 1], size=len(test_df))

        elif method == "majority":
            majority = train_df["label"].mode()[0]
            return np.full(len(test_df), majority)

        elif method == "tfidf_lr":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            try:
                valid_train = train_df.dropna(subset=["claim", "label"])
                valid_test = test_df.dropna(subset=["claim"])

                X_train = valid_train["claim"].astype(str)
                y_train = valid_train["label"]
                X_test = valid_test["claim"].astype(str)

                vectorizer = TfidfVectorizer(
                    max_features=1500, ngram_range=(1, 2)
                )
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_vec, y_train)

                preds = clf.predict(X_test_vec)

                out = np.full(len(test_df), np.nan)
                out[valid_test.index] = preds
                return out

            except Exception as e:
                print(f"TF-IDF baseline failed: {e}")
                majority = train_df["label"].mode()[0]
                return np.full(len(test_df), majority)

        else:
            raise ValueError(f"Unknown baseline: {method}")

    # --------------------------------------------------------
    # METRICS PRINTING
    # --------------------------------------------------------

    def print_overall_results(self, df_clean, label):
        acc = accuracy_score(df_clean["label"], df_clean["prediction"])
        prec = precision_score(df_clean["label"], df_clean["prediction"], average="macro")
        rec = recall_score(df_clean["label"], df_clean["prediction"], average="macro")
        f1 = f1_score(df_clean["label"], df_clean["prediction"], average="macro")

        print(f"\nOverall Results for {label}:")
        print(f"Accuracy: {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall: {rec:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(df_clean["label"], df_clean["prediction"]))
        print(f"Cohen’s Kappa: {cohen_kappa_score(df_clean['label'], df_clean['prediction']):.2f}")
        print(f"MCC: {matthews_corrcoef(df_clean['label'], df_clean['prediction']):.2f}")

    # --------------------------------------------------------
    # DOMAIN-SPECIFIC METRICS
    # --------------------------------------------------------

    def print_domain_results(self, df_clean, label):
        print(f"\nDomain-Specific Results for {label}:")
        domains = domains = sorted(df_clean["domain"].unique())

        for domain in domains:
            print(f"\nDomain {domain}:")
            subset = df_clean[df_clean["domain"] == domain]

            if len(subset) == 0:
                print("No data for this domain.")
                continue

            acc = accuracy_score(subset["label"], subset["prediction"])
            prec = precision_score(subset["label"], subset["prediction"], average="macro")
            rec = recall_score(subset["label"], subset["prediction"], average="macro")
            f1 = f1_score(subset["label"], subset["prediction"], average="macro")

            print(f"Accuracy: {acc:.2f}")
            print(f"Precision: {prec:.2f}")
            print(f"Recall: {rec:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print(confusion_matrix(subset["label"], subset["prediction"]))

    # --------------------------------------------------------
    # FULL METRICS OBJECT (FOR CSV)
    # --------------------------------------------------------

    def calculate_comprehensive_metrics(self, y_true, y_pred, dataset_name, condition, run_id):
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

        if len(y_true) == 0:
            return None

        report = classification_report(
            y_true, y_pred, output_dict=True,
            target_names=["False", "True"],
            zero_division=0
        )

        return {
            "dataset": dataset_name,
            "condition": condition,
            "run_id": run_id,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1_score": f1_score(y_true, y_pred, average="macro"),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "precision_false": report["False"]["precision"],
            "recall_false": report["False"]["recall"],
            "f1_false": report["False"]["f1-score"],
            "precision_true": report["True"]["precision"],
            "recall_true": report["True"]["recall"],
            "f1_true": report["True"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "n_samples": len(y_true),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    # --------------------------------------------------------
    # RUN ONE EXPERIMENT (UGC or NGC for 1 RUN)
    # --------------------------------------------------------

    def run_single_experiment(self, df, dataset_name, run_id):
        print("\n" + "="*60)
        print(f"{dataset_name} Dataset ‒ Run {run_id+1}/{self.config['n_runs']}")
        print("="*60)

        seed = self.config["random_seeds"][run_id]

        train_df, test_df = self.create_stratified_split(
            df, test_size=0.3, random_state=seed
        )

        print(f"\nProcessing {len(test_df)} test claims...")

        predictions = []
        for i, (_, row) in enumerate(test_df.iterrows(), 1):
            if i % 25 == 0:
                print(f"Processing {i}/{len(test_df)}")
            pred = self.classify_claim(row["claim"], row["domain"], run_id)
            predictions.append(pred)
            sleep(0.1)

        # Baselines
        random_baseline = self.generate_baseline_predictions(train_df, test_df, "random")
        majority_baseline = self.generate_baseline_predictions(train_df, test_df, "majority")
        tfidf_baseline = self.generate_baseline_predictions(train_df, test_df, "tfidf_lr")

        results_df = test_df.copy()
        results_df["prediction"] = predictions
        results_df["random_baseline"] = random_baseline
        results_df["majority_baseline"] = majority_baseline
        results_df["tfidf_baseline"] = tfidf_baseline
        results_df["run_id"] = run_id

        df_clean = results_df.dropna(subset=["prediction"])

        print(f"Successfully classified {len(df_clean)}/{len(results_df)} claims")

        self.print_overall_results(df_clean, f"{dataset_name} Run {run_id+1}")
        self.print_domain_results(df_clean, f"{dataset_name} Run {run_id+1}")

        # Store metrics
        model_metrics = self.calculate_comprehensive_metrics(
            df_clean["label"], df_clean["prediction"], dataset_name, "Model", run_id
        )
        if model_metrics:
            self.all_results.append(model_metrics)

        # Store baselines only for run 0 (clean comparison)
        if run_id == 0:
            for base_name, base_preds in [
                ("Random", random_baseline),
                ("Majority", majority_baseline),
                ("TFIDF", tfidf_baseline)
            ]:
                base_df = results_df.copy()
                base_df["prediction"] = base_preds
                base_clean = base_df.dropna(subset=["prediction"])

                base_metrics = self.calculate_comprehensive_metrics(
                    base_clean["label"], base_clean["prediction"], dataset_name, base_name, 0
                )
                if base_metrics:
                    self.all_results.append(base_metrics)

        # Save run results
        self.run_data_storage[f"{dataset_name}_run_{run_id}"] = df_clean

        outfile = f"{dataset_name.lower()}_run_{run_id+1}_results.csv"
        df_clean.to_csv(outfile, index=False)
        print(f"Saved results to {outfile}")

        return df_clean

    # --------------------------------------------------------
    # STATISTICAL ANALYSIS ACROSS RUNS
    # --------------------------------------------------------

    def perform_statistical_analysis(self, dataset_name):
        print("\n" + "="*60)
        print(f"STATISTICAL ANALYSIS: {dataset_name}")
        print("="*60)

        model_results = [
            r for r in self.all_results
            if r["dataset"] == dataset_name and r["condition"] == "Model"
        ]

        if not model_results:
            print("No model results found.")
            return

        accuracies = [r["accuracy"] for r in model_results]
        f1s = [r["f1_score"] for r in model_results]

        print(f"\nMean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Mean F1-Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # --------------------------------------------------------
    # AGGREGATED DOMAIN ANALYSIS
    # --------------------------------------------------------

    def generate_aggregated_domain_analysis(self, dataset_name):
        print("\n" + "="*60)
        print(f"AGGREGATED DOMAIN ANALYSIS: {dataset_name}")
        print("="*60)

        frames = [
            df for key, df in self.run_data_storage.items()
            if key.startswith(dataset_name)
        ]

        all_df = pd.concat(frames, ignore_index=True)

        domains = sorted(all_df["domain"].unique())
        summary = []

        for dom in domains:
            subset = all_df[all_df["domain"] == dom]

            accuracy = accuracy_score(subset["label"], subset["prediction"])
            precision = precision_score(subset["label"], subset["prediction"], average="macro")
            recall = recall_score(subset["label"], subset["prediction"], average="macro")
            f1 = f1_score(subset["label"], subset["prediction"], average="macro")

            summary.append({
                "domain": dom,
                "n_samples": len(subset),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        df_summary = pd.DataFrame(summary)
        outfile = f"{dataset_name.lower()}_aggregated_domain_analysis_mistral-7b-instruct-v0.2.csv"
        df_summary.to_csv(outfile, index=False)
        print(f"Saved aggregated domain analysis to {outfile}")

        return summary

    # --------------------------------------------------------
    # FINAL REPORT (MAIN CSV OUTPUT)
    # --------------------------------------------------------

    def generate_final_report(self):
        print("\n" + "="*60)
        print("FINAL THESIS RESULTS SUMMARY")
        print("="*60)

        df = pd.DataFrame(self.all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = f"mistral-7b-instruct-v0.2_thesis_comprehensive_results_{timestamp}.csv"
        df.to_csv(outfile, index=False)

        print(f"Saved final comprehensive results to {outfile}")
        return df

    # --------------------------------------------------------
    # CLEAN DATASETS
    # --------------------------------------------------------

    def clean_datasets(self):
        print("\nCleaning datasets...")

        def load_and_clean(path):
            df = pd.read_csv(path)
            df = df.dropna(subset=["claim", "label", "domain"])
            df = df[df["claim"].astype(str).str.strip() != ""]
            return df

        ugc = load_and_clean(self.config["ugc_file"])
        ngc = load_and_clean(self.config["ngc_file"])

        print(f"UGC rows: {len(ugc)}")
        print(f"NGC rows: {len(ngc)}")

        return ugc, ngc

    # --------------------------------------------------------
    # MAIN RUN CONTROLLER
    # --------------------------------------------------------

    def run_complete_evaluation(self):
        try:
            ugc_df, ngc_df = self.clean_datasets()

            print(f"\nFinal datasets loaded: UGC={len(ugc_df)}, NGC={len(ngc_df)}")

            for run_id in range(self.config["n_runs"]):
                print("\n" + "#"*60)
                print(f"# EXPERIMENTAL RUN {run_id+1}/{self.config['n_runs']}")
                print("#"*60)

                self.run_single_experiment(ugc_df, "UGC", run_id)
                self.run_single_experiment(ngc_df, "NGC", run_id)

            print("\n" + "#"*60)
            print("# STATISTICAL ANALYSIS ACROSS RUNS")
            print("#"*60)
            self.perform_statistical_analysis("UGC")
            self.perform_statistical_analysis("NGC")

            print("\n" + "#"*60)
            print("# AGGREGATED DOMAIN ANALYSIS")
            print("#"*60)
            self.generate_aggregated_domain_analysis("UGC")
            self.generate_aggregated_domain_analysis("NGC")

            self.generate_final_report()

            print("\nEVALUATION COMPLETED SUCCESSFULLY!")
            print(f"Completed at: {datetime.now()}")

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------

if __name__ == "__main__":
    evaluator = ThesisEvaluator(CONFIG)
    evaluator.run_complete_evaluation()
