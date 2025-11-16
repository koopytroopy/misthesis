import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, classification_report
from scipy import stats
import numpy as np
from time import sleep
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Thesis Misinformation Testing")
print(f"Experiment started: {datetime.now()}")


# CONFIGURATION

CONFIG = {
    "ugc_file": "/Users/koopytroopy/Desktop/ugc_master_Ex.csv",
    "ngc_file": "/Users/koopytroopy/Desktop/ngc_master_Ex.csv",
    "model_name": "gemma-3-4b",
    "api_url": "http://127.0.0.1:1234/v1/completions",
    "n_runs": 5,
    "random_seeds": [42, 123, 456, 789, 999]
}
# EVALUATOR CLASS

class ThesisEvaluator:
    def __init__(self, config):
        self.config = config
        self.all_results = []
        self.run_data_storage = {}

# PROMPT BUILDER

    def generate_prompt(self, claim, domain):
        return (
            f"Classify the following {domain} claim strictly as 'true' or 'false'. "
            f"Respond with only one word (true/false).\n\n"
            f"Claim: \"{claim}\"\nAnswer:"
        )
    
    # CLASSIFIER 

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

            # Normalize punctuation
            cleaned = (
                output.replace(".", "")
                .replace(",", "")
                .replace(":", "")
                .replace(";", "")
                .strip()
            )

            tokens = cleaned.split()

            # 1. First token direct match
            if tokens:
                if tokens[0] == "true":
                    return 1
                if tokens[0] == "false":
                    return 0

            # 2. Look in first 5 tokens
            window = tokens[:5]
            if "true" in window:
                return 1
            if "false" in window:
                return 0

            # 3. Regex fallback
            import re
            if re.search(r"\btrue\b", cleaned):
                return 1
            if re.search(r"\bfalse\b", cleaned):
                return 0

            # No match
            return None

        except:
            return None
    
    # DATA SPLITTING 

    def create_stratified_split(self, df, test_size=0.3, random_state=42):
        from sklearn.model_selection import train_test_split

        df = df.copy()
        df["stratify_col"] = df["domain"] + "_" + df["label"].astype(str)

        print("\nDataset Distribution Before Split:")
        print(df.groupby(["domain", "label"]).size().unstack(fill_value=0))
        print(f"\nTotal samples: {len(df)}")

        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df["stratify_col"]
            )

            train_df.drop(columns=["stratify_col"], inplace=True)
            test_df.drop(columns=["stratify_col"], inplace=True)

            print(f"\nTrain Set ({len(train_df)} samples):")
            print(train_df.groupby(["domain", "label"]).size().unstack(fill_value=0))

            print(f"\nTest Set ({len(test_df)} samples):")
            print(test_df.groupby(["domain", "label"]).size().unstack(fill_value=0))

            return train_df, test_df

        except ValueError:
            print("\n⚠️ Stratified split failed. Falling back to label-only stratification.")
            train_df, test_df = train_test_split(
                df.drop(columns=["stratify_col"]),
                test_size=test_size,
                stratify=df["label"],
                random_state=random_state
            )
            return train_df, test_df

   
  
    # BASELINES

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
                X_train = train_df["claim"].astype(str)
                y_train = train_df["label"]

                X_test = test_df["claim"].astype(str)

                vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_vec, y_train)

                return clf.predict(X_test_vec)

            except Exception as e:
                print(f"TF-IDF baseline failed: {e}")
                majority = train_df["label"].mode()[0]
                return np.full(len(test_df), majority)

        else:
            raise ValueError(f"Unknown baseline method '{method}'")

 
    # PRINTING METRICS
   
    def print_overall_results(self, df_clean, label):
        accuracy = accuracy_score(df_clean["label"], df_clean["prediction"])
        precision = precision_score(df_clean["label"], df_clean["prediction"], average="macro", zero_division=0)
        recall = recall_score(df_clean["label"], df_clean["prediction"], average="macro", zero_division=0)
        f1 = f1_score(df_clean["label"], df_clean["prediction"], average="macro", zero_division=0)

        print(f"\nOverall Results for {label}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(df_clean["label"], df_clean["prediction"]))

        print(f"Cohen's Kappa: {cohen_kappa_score(df_clean['label'], df_clean['prediction']):.2f}")
        print(f"MCC: {matthews_corrcoef(df_clean['label'], df_clean['prediction']):.2f}")

    
    # FIXED DOMAIN-SAFE METRICS
    
    def print_domain_results(self, df_clean, label):
        print(f"\nDomain-Specific Results for {label}:")

        domains = sorted(df_clean["domain"].astype(str).unique())

        for domain in domains:
            domain_str = str(domain).strip()
            print(f"\n{domain_str.capitalize()}:")

            subset = df_clean[df_clean["domain"] == domain]
            if len(subset) == 0:
                print("No data for this domain.")
                continue

            accuracy = accuracy_score(subset["label"], subset["prediction"])
            precision = precision_score(subset["label"], subset["prediction"], average="macro", zero_division=0)
            recall = recall_score(subset["label"], subset["prediction"], average="macro", zero_division=0)
            f1 = f1_score(subset["label"], subset["prediction"], average="macro", zero_division=0)

            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print(confusion_matrix(subset["label"], subset["prediction"]))

    
    # FULL METRICS OBJECT
   
    def calculate_comprehensive_metrics(self, y_true, y_pred, dataset_name, condition, run_id):
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

        if len(y_true) == 0:
            return None

        return {
            "dataset": dataset_name,
            "condition": condition,
            "run_id": run_id,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "n_samples": len(y_true),
        }

    
    # RUN ONE EXPERIMENT
    
    def run_single_experiment(self, df, dataset_name, run_id):
        print("\n" + "="*60)
        print(f"{dataset_name} Dataset - Run {run_id+1}/{self.config['n_runs']}")
        print("="*60)

        seed = self.config["random_seeds"][run_id]
        train_df, test_df = self.create_stratified_split(df, 0.3, random_state=seed)

        print(f"\nProcessing {len(test_df)} claims...")

        predictions = []
        for i, row in enumerate(test_df.iterrows(), 1):
            if i % 25 == 0:
                print(f"Processing claim {i}/{len(test_df)}")
            pred = self.classify_claim(row[1]["claim"], row[1]["domain"], run_id)
            predictions.append(pred)
            sleep(0.1)

        # Baselines
        random_baseline = self.generate_baseline_predictions(train_df, test_df, "random")
        majority_baseline = self.generate_baseline_predictions(train_df, test_df, "majority")
        tfidf_baseline = self.generate_baseline_predictions(train_df, test_df, "tfidf_lr")

        test_df = test_df.copy()
        test_df["prediction"] = predictions
        test_df["random_baseline"] = random_baseline
        test_df["majority_baseline"] = majority_baseline
        test_df["tfidf_baseline"] = tfidf_baseline

        df_clean = test_df.dropna(subset=["prediction", "label"])

        print(f"Successfully classified {len(df_clean)}/{len(test_df)} claims")

        self.print_overall_results(df_clean, f"{dataset_name} Run {run_id+1}")
        self.print_domain_results(df_clean, f"{dataset_name} Run {run_id+1}")

        metrics = self.calculate_comprehensive_metrics(
            df_clean["label"], df_clean["prediction"], dataset_name, "Model", run_id
        )
        if metrics:
            self.all_results.append(metrics)

        # Save cleaned data
        self.run_data_storage[f"{dataset_name}_run_{run_id}"] = df_clean

        outfile = f"{dataset_name.lower()}_run_{run_id+1}_results.csv"
        df_clean.to_csv(outfile, index=False)
        print(f"Saved results to {outfile}")

    
    # DOMAIN ANALYSIS
    
    def generate_aggregated_domain_analysis(self, dataset_name):
        print("\n" + "="*60)
        print(f"AGGREGATED DOMAIN ANALYSIS: {dataset_name}")
        print("="*60)

        frames = [
            df for key, df in self.run_data_storage.items()
            if key.startswith(dataset_name)
        ]

        all_df = pd.concat(frames, ignore_index=True)
        all_df["domain"] = all_df["domain"].astype(str)

        domains = sorted(all_df["domain"].unique())

        summary = []

        for dom in domains:
            subset = all_df[all_df["domain"] == dom]
            accuracy = accuracy_score(subset["label"], subset["prediction"])
            precision = precision_score(subset["label"], subset["prediction"], average="macro", zero_division=0)
            recall = recall_score(subset["label"], subset["prediction"], average="macro", zero_division=0)
            f1 = f1_score(subset["label"], subset["prediction"], average="macro", zero_division=0)

            summary.append({
                "domain": dom,
                "n_samples": len(subset),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        df_summary = pd.DataFrame(summary)
        outfile = f"{dataset_name.lower()}_aggregated_domain_analysis.csv"
        df_summary.to_csv(outfile, index=False)
        print(f"Saved domain analysis to {outfile}")

        return summary

    # ============================================================
    # CLEAN DATASETS
    # ============================================================
    def clean_datasets(self):
        print("Cleaning datasets...")

        def load_and_clean(path):
            df = pd.read_csv(path)
            df = df.dropna(subset=["claim", "label", "domain"])
            df = df[df["claim"].astype(str).str.strip() != ""]
            df["domain"] = df["domain"].astype(str).str.strip()
            return df

        ugc_df = load_and_clean(self.config["ugc_file"])
        ngc_df = load_and_clean(self.config["ngc_file"])

        print(f"UGC rows: {len(ugc_df)}")
        print(f"NGC rows: {len(ngc_df)}")

        return ugc_df, ngc_df

    
    # RUN EVERYTHING
    
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
            print("# AGGREGATED DOMAIN ANALYSIS")
            print("#"*60)

            self.generate_aggregated_domain_analysis("UGC")
            self.generate_aggregated_domain_analysis("NGC")

            print("\n\nEVALUATION COMPLETED SUCCESSFULLY!")
            print(f"Completed at: {datetime.now()}")

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()


# MAIN

if __name__ == "__main__":
    evaluator = ThesisEvaluator(CONFIG)
    evaluator.run_complete_evaluation() 
    # --------------------------------------------------------
    def generate_baseline_predictions(self, train_df, test_df, method='random'):
        np.random.seed(42)

        if method == 'random':
            return np.random.choice([0, 1], size=len(test_df))

        elif method == 'majority':
            majority_class = train_df['label'].mode()[0]
            return np.full(len(test_df), majority_class)

        elif method == 'tfidf_lr':
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            try:
                valid_train = train_df.dropna(subset=['claim', 'label'])
                valid_test = test_df.dropna(subset=['claim'])

                X_train = valid_train['claim'].astype(str)
                y_train = valid_train['label']
                X_test = valid_test['claim'].astype(str)

                vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_vec, y_train)

                preds = clf.predict(X_test_vec)
                return preds
            
            except Exception as e:
                print(f"⚠️ TF-IDF baseline failed: {e}. Using majority class.")
                majority_class = train_df['label'].mode()[0]
                return np.full(len(test_df), majority_class)

        else:
            raise ValueError(f"Unknown baseline method: {method}")

    # --------------------------------------------------------

    def print_overall_results(self, df_clean, label):
        accuracy = accuracy_score(df_clean["label"], df_clean["prediction"])
        precision = precision_score(df_clean["label"], df_clean["prediction"], average='macro', zero_division=0)
        recall = recall_score(df_clean["label"], df_clean["prediction"], average='macro', zero_division=0)
        f1 = f1_score(df_clean["label"], df_clean["prediction"], average='macro', zero_division=0)
        
        print(f"\nOverall Results for {label}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(df_clean["label"], df_clean["prediction"]))
        
        kappa = cohen_kappa_score(df_clean["label"], df_clean["prediction"])
        mcc = matthews_corrcoef(df_clean["label"], df_clean["prediction"])
        print(f"Cohen's Kappa: {kappa:.2f}")
        print(f"Matthews Correlation Coefficient: {mcc:.2f}")

    def print_domain_results(self, df_clean, label):
        print(f"\nDomain-Specific Results for {label}:")
        domains = sorted([d for d in df_clean["domain"].unique() if pd.notna(d)])
        
        for domain in domains:
            print(f"\n{domain.capitalize()}:")
            subset = df_clean[df_clean["domain"] == domain]
            if len(subset) == 0:
                print("No data for this domain")
                continue
            
            accuracy = accuracy_score(subset["label"], subset["prediction"])
            precision = precision_score(subset["label"], subset["prediction"], average='macro', zero_division=0)
            recall = recall_score(subset["label"], subset["prediction"], average='macro', zero_division=0)
            f1 = f1_score(subset["label"], subset["prediction"], average='macro', zero_division=0)
            
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print("Confusion Matrix:")
            print(confusion_matrix(subset["label"], subset["prediction"]))

    def calculate_comprehensive_metrics(self, y_true, y_pred, dataset_name, condition_name, run_id):
        valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = np.array(y_true)[valid_mask]
        y_pred = np.array(y_pred)[valid_mask]
        
        if len(y_true) == 0:
            return None
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True, target_names=["False", "True"], zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            return {
                'dataset': dataset_name,
                'condition': condition_name,
                'run_id': run_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cohen_kappa': kappa,
                'mcc': mcc,
                'precision_false': report['False']['precision'],
                'recall_false': report['False']['recall'],
                'f1_false': report['False']['f1-score'],
                'precision_true': report['True']['precision'],
                'recall_true': report['True']['recall'],
                'f1_true': report['True']['f1-score'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'confusion_matrix': cm.tolist(),
                'n_samples': len(y_true)
            }
        except:
            return None

    def run_single_experiment(self, df, dataset_name, run_id):
        print(f"\n{'='*60}")
        print(f"{dataset_name} Dataset - Run {run_id + 1}/{self.config['n_runs']}")
        print(f"{'='*60}")
        
        seed = self.config['random_seeds'][run_id]
        train_df, test_df = self.create_stratified_split(df, test_size=0.3, random_state=seed)
        
        print(f"\nProcessing {len(test_df)} test claims...")
        
        np.random.seed(seed)
        
        predictions = []
        for i, (_, row) in enumerate(test_df.iterrows(), start=1):
            if i % 25 == 0:
                print(f"Processing test claim {i}/{len(test_df)}")
            pred = self.classify_claim(row["claim"], row["domain"], run_id)
            predictions.append(pred)
            sleep(0.1)
        
        print("\nGenerating baseline predictions...")
        random_baseline = self.generate_baseline_predictions(train_df, test_df, 'random')
        majority_baseline = self.generate_baseline_predictions(train_df, test_df, 'majority')
        tfidf_baseline = self.generate_baseline_predictions(train_df, test_df, 'tfidf_lr')
        
        results_df = test_df.copy()
        results_df['prediction'] = predictions
        results_df['random_baseline'] = random_baseline
        results_df['majority_baseline'] = majority_baseline
        results_df['tfidf_baseline'] = tfidf_baseline
        results_df['run_id'] = run_id
        results_df['split'] = 'test'
        
        df_clean = results_df.dropna(subset=['label', 'prediction'])
        
        print(f"Successfully classified {len(df_clean)}/{len(results_df)} test claims")
        
        self.print_overall_results(df_clean, f"{dataset_name} Run {run_id + 1}")
        self.print_domain_results(df_clean, f"{dataset_name} Run {run_id + 1}")
        
        model_metrics = self.calculate_comprehensive_metrics(
            df_clean['label'], df_clean['prediction'], dataset_name, 'Model', run_id
        )
        if model_metrics:
            self.all_results.append(model_metrics)
        
        if run_id == 0:
            for baseline_name, baseline_preds in [
                ('Random', random_baseline),
                ('Majority', majority_baseline),
                ('TF-IDF+LR', tfidf_baseline)
            ]:
                baseline_df = results_df.copy()
                baseline_df['prediction'] = baseline_preds
                baseline_clean = baseline_df.dropna(subset=['prediction'])
                
                baseline_metrics = self.calculate_comprehensive_metrics(
                    baseline_clean['label'], baseline_clean['prediction'],
                    dataset_name, baseline_name, 0
                )
                if baseline_metrics:
                    self.all_results.append(baseline_metrics)
        
        storage_key = f"{dataset_name}_run_{run_id}"
        self.run_data_storage[storage_key] = {
            'dataframe': df_clean,
            'dataset': dataset_name,
            'run_id': run_id
        }
        
        output_file = f"{dataset_name.lower()}_run_{run_id + 1}_results.csv"
        df_clean.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        
        return df_clean

    def perform_statistical_analysis(self, dataset_name):
        print(f"\n{'='*60}")
        print(f"STATISTICAL ANALYSIS: {dataset_name}")
        print(f"{'='*60}")
        
        model_results = [r for r in self.all_results
                         if r['dataset'] == dataset_name and r['condition'] == 'Model']
        
        if len(model_results) == 0:
            print("No valid results for statistical analysis")
            return
        
        accuracies = [r['accuracy'] for r in model_results]
        precisions = [r['precision'] for r in model_results]
        recalls = [r['recall'] for r in model_results]
        f1_scores = [r['f1_score'] for r in model_results]
        
        print(f"\nModel Performance across {len(model_results)} runs:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"  Mean Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        print(f"  Mean Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        print(f"  Mean F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        if len(accuracies) > 1:
            acc_ci = stats.t.interval(0.95, len(accuracies)-1,
                                      loc=np.mean(accuracies),
                                      scale=stats.sem(accuracies))
            f1_ci = stats.t.interval(0.95, len(f1_scores)-1,
                                     loc=np.mean(f1_scores),
                                     scale=stats.sem(f1_scores))
            
            print("\n95% Confidence Intervals:")
            print(f"  Accuracy: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
            print(f"  F1-Score: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
        
        baseline_results = [r for r in self.all_results
                            if r['dataset'] == dataset_name and r['condition'] != 'Model']
        
        if len(baseline_results) > 0:
            print(f"\nBaseline Comparisons:")
            for baseline_result in baseline_results:
                baseline_acc = baseline_result['accuracy']
                baseline_f1 = baseline_result['f1_score']
                acc_improvement = np.mean(accuracies) - baseline_acc
                f1_improvement = np.mean(f1_scores) - baseline_f1
                
                print(f"\n  vs {baseline_result['condition']} Baseline:")
                print(f"    Baseline Accuracy: {baseline_acc:.3f} (Model: +{acc_improvement:+.3f})")
                print(f"    Baseline F1-Score: {baseline_f1:.3f} (Model: +{f1_improvement:+.3f})")
                
                if len(accuracies) > 1:
                    t_stat, p_value = stats.ttest_1samp(accuracies, baseline_acc)
                    significance = ("***" if p_value < 0.001 else
                                    "**" if p_value < 0.01 else
                                    "*" if p_value < 0.05 else "ns")
                    print(f"    Statistical Significance: p={p_value:.4f} {significance}")

    def generate_aggregated_domain_analysis(self, dataset_name):
        print(f"\n{'='*60}")
        print(f"AGGREGATED DOMAIN ANALYSIS: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_runs = []
        for key, stored_data in self.run_data_storage.items():
            if stored_data['dataset'] == dataset_name:
                dataset_runs.append(stored_data['dataframe'])
        
        if len(dataset_runs) == 0:
            print("No domain analysis data available.")
            return
        
        all_data = pd.concat(dataset_runs, ignore_index=True)
        
        domains = sorted([d for d in all_data['domain'].unique() if pd.notna(d)])
        
        print(f"\nDomains found: {domains}")
        print(f"Total runs aggregated: {len(dataset_runs)}")
        
        summary = []
        
        for domain in domains:
            domain_data = all_data[all_data['domain'] == domain]
            if len(domain_data) == 0:
                continue
            
            accuracy = accuracy_score(domain_data['label'], domain_data['prediction'])
            precision = precision_score(domain_data['label'], domain_data['prediction'], average='macro', zero_division=0)
            recall = recall_score(domain_data['label'], domain_data['prediction'], average='macro', zero_division=0)
            f1 = f1_score(domain_data['label'], domain_data['prediction'], average='macro', zero_division=0)
            
            summary.append({
                'domain': domain,
                'n_samples': len(domain_data),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        summary = sorted(summary, key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\nAggregated Domain Performance:")
        print("-" * 80)
        print(f"{'Domain':<15} {'N':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        
        for s in summary:
            print(f"{s['domain']:<15} {s['n_samples']:<8} {s['accuracy']:<10.2f} {s['precision']:<10.2f} "
                  f"{s['recall']:<10.2f} {s['f1_score']:<10.2f}")
        
        domain_df = pd.DataFrame(summary)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = f"{dataset_name.lower()}_aggregated_domain_analysis_{timestamp}.csv"
        domain_df.to_csv(outfile, index=False)
        print(f"\nSaved domain analysis to {outfile}")
        
        return summary

    def generate_final_report(self):
        print(f"\n{'='*60}")
        print("FINAL THESIS RESULTS SUMMARY")
        print(f"{'='*60}")
        
        model_results = [r for r in self.all_results if r['condition'] == 'Model']
        
        if len(model_results) == 0:
            print("No results to summarize.")
            return
        
        for dataset in ['UGC', 'NGC']:
            dataset_results = [r for r in model_results if r['dataset'] == dataset]
            if len(dataset_results) == 0:
                continue
            
            accuracies = [r['accuracy'] for r in dataset_results]
            precisions = [r['precision'] for r in dataset_results]
            recalls = [r['recall'] for r in dataset_results]
            f1s = [r['f1_score'] for r in dataset_results]
            kappas = [r['cohen_kappa'] for r in dataset_results]
            
            print(f"\n{dataset} Dataset Summary:")
            print(f"  Runs: {len(dataset_results)}")
            print(f"  Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
            print(f"  Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
            print(f"  Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
            print(f"  F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
            print(f"  Kappa: {np.mean(kappas):.3f}")
        
        results_df = pd.DataFrame(self.all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = f"thesis_comprehensive_results_{timestamp}.csv"
        results_df.to_csv(outfile, index=False)
        
        print(f"\nSaved all results to {outfile}")
        return results_df

    def clean_datasets(self):
        print("Cleaning datasets...") 
        
        ugc_df = pd.read_csv(self.config["ugc_file"])
        ugc_df = ugc_df.dropna(subset=['claim', 'label', 'domain'])
        ugc_df = ugc_df[ugc_df['claim'].astype(str).str.strip() != '']
        print(f"UGC rows: {len(ugc_df)}")

        ngc_df = pd.read_csv(self.config["ngc_file"])
        ngc_df = ngc_df.dropna(subset=['claim', 'label', 'domain'])
        ngc_df = ngc_df[ngc_df['claim'].astype(str).str.strip() != '']
        print(f"NGC rows: {len(ngc_df)}")
        
        return ugc_df, ngc_df

    def run_complete_evaluation(self):
        try:
            ugc_df, ngc_df = self.clean_datasets()
            
            print(f"\nFinal datasets loaded: UGC={len(ugc_df)}, NGC={len(ngc_df)}")
            
            for run_id in range(self.config["n_runs"]):
                print(f"\n{'#'*60}")
                print(f"# EXPERIMENTAL RUN {run_id + 1}/{self.config['n_runs']}")
                print(f"{'#'*60}")
                
                self.run_single_experiment(ugc_df, "UGC", run_id)
                self.run_single_experiment(ngc_df, "NGC", run_id)
            
            print(f"\n{'#'*60}")
            print("# STATISTICAL ANALYSIS ACROSS RUNS")
            print(f"{'#'*60}")
            
            self.perform_statistical_analysis("UGC")
            self.perform_statistical_analysis("NGC")
            
            print(f"\n{'#'*60}")
            print("# AGGREGATED DOMAIN ANALYSIS")
            print(f"{'#'*60}")
            
            self.generate_aggregated_domain_analysis("UGC")
            self.generate_aggregated_domain_analysis("NGC")
            
            final_results = self.generate_final_report()
            
            print(f"\n{'='*60}")
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Completed at: {datetime.now()}")
            
            return final_results
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# --------------------------------------------------------
# Main execution
# --------------------------------------------------------