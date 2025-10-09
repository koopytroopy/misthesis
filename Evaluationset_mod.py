import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, classification_report
from scipy import stats
import numpy as np
from time import sleep
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Thesis-Grade Misinformation Testing - Simplified Version")
print(f"Experiment started: {datetime.now()}")

# Configuration
CONFIG = {
    "ugc_file": "../data/ugc_master.csv",
    "ngc_file": "../data/ngc_master.csv", 
    "model_name": "llama-3.2-3b-instruct",
    "api_url": "http://127.0.0.1:1234/v1/completions",
    "n_runs": 5,  # Multiple runs for statistical reliability
    "random_seeds": [42, 123, 456, 789, 999]
}

class SimpleThesisEvaluator:
    def __init__(self, config):
        self.config = config
        self.all_results = []
        self.run_data_storage = {}  # Store detailed run data for analysis
        
    def generate_prompt(self, claim, domain):
        """Simple working prompt based on your successful version"""
        return f"""You are a fact-checking assistant reviewing a claim about {domain}.
Based on the content, your task is to determine whether the claim(s) are factually accurate or contain misinformation.

Only respond with one word: true or false.

Claim: "{claim}"

Answer:"""
    
    def classify_claim(self, claim, domain, run_id=0):
        """Classify claim using working API parameters with data validation"""
        # Handle NaN or non-string claims
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
                    "max_tokens": 20,
                    "temperature": 0.1,
                    "stop": ["\n"],
                    "seed": self.config["random_seeds"][run_id % len(self.config["random_seeds"])]
                }
            )

            if response.status_code != 200:
                return None

            output = response.json()["choices"][0]["text"].strip().lower()

            # Flexible output check - handle periods and extra text
            if "true" in output and "false" not in output:
                return 1
            elif "false" in output and "true" not in output:
                return 0
            elif output.startswith("true"):
                return 1
            elif output.startswith("false"):
                return 0
            else:
                return None

        except Exception as e:
            return None
    
    def generate_baseline_predictions(self, df, method='random'):
        """Generate baseline predictions for comparison with NaN protection"""
        np.random.seed(42)  # Fixed seed for reproducibility
        
        if method == 'random':
            return np.random.choice([0, 1], size=len(df))
        elif method == 'majority':
            majority_class = df['label'].mode()[0]
            return [majority_class] * len(df)
        elif method == 'keyword':
            # Simple keyword-based heuristic with NaN protection
            predictions = []
            suspicious_keywords = ['breaking', 'shocking', 'revealed', 'secret', 'exclusive']
            for claim in df['claim']:
                # Handle NaN or non-string claims
                if pd.isna(claim) or not isinstance(claim, str):
                    predictions.append(1)  # Default to true for missing claims
                else:
                    claim_lower = claim.lower()
                    if any(keyword in claim_lower for keyword in suspicious_keywords):
                        predictions.append(0)  # Likely false
                    else:
                        predictions.append(1)  # Likely true
            return predictions
    
    def print_overall_results(self, df_clean, label):
        """Print overall results in your preferred format"""
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
        
        # Also calculate Cohen's Kappa and MCC for thesis rigor
        kappa = cohen_kappa_score(df_clean["label"], df_clean["prediction"])
        mcc = matthews_corrcoef(df_clean["label"], df_clean["prediction"])
        print(f"Cohen's Kappa: {kappa:.2f}")
        print(f"Matthews Correlation Coefficient: {mcc:.2f}")
    
    def print_domain_results(self, df_clean, label):
        """Print domain-specific results in your preferred format"""
        print(f"\nDomain-Specific Results for {label}:")
        domains = df_clean["domain"].unique()
        domains = sorted([d for d in domains if pd.notna(d)])
        
        for domain in domains:
            print(f"\n{domain.capitalize()}:")
            subset = df_clean[df_clean["domain"] == domain]
            
            if len(subset) > 0:
                try:
                    domain_accuracy = accuracy_score(subset["label"], subset["prediction"])
                    domain_precision = precision_score(subset["label"], subset["prediction"], average='macro', zero_division=0)
                    domain_recall = recall_score(subset["label"], subset["prediction"], average='macro', zero_division=0)
                    domain_f1 = f1_score(subset["label"], subset["prediction"], average='macro', zero_division=0)
                    
                    print(f"Accuracy: {domain_accuracy:.2f}")
                    print(f"Precision: {domain_precision:.2f}")
                    print(f"Recall: {domain_recall:.2f}")
                    print(f"F1-Score: {domain_f1:.2f}")
                    
                    print("Confusion Matrix:")
                    print(confusion_matrix(subset["label"], subset["prediction"]))
                except Exception as e:
                    print(f"Error calculating metrics: {e}")
            else:
                print("No data available for this domain")
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, dataset_name, condition_name, run_id):
        """Calculate comprehensive evaluation metrics for statistical analysis"""
        # Remove NaN values
        valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true_clean = np.array(y_true)[valid_mask]
        y_pred_clean = np.array(y_pred)[valid_mask]
        
        if len(y_true_clean) == 0:
            return None
        
        # Basic metrics
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        precision = precision_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        recall = recall_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        f1 = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        
        # Advanced metrics
        try:
            kappa = cohen_kappa_score(y_true_clean, y_pred_clean)
            mcc = matthews_corrcoef(y_true_clean, y_pred_clean)
            
            # Classification report
            report = classification_report(y_true_clean, y_pred_clean, 
                                         output_dict=True, target_names=["False", "True"], 
                                         zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true_clean, y_pred_clean)
            
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
                'n_samples': len(y_true_clean)
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def run_single_experiment(self, df, dataset_name, run_id):
        """Run single experiment with comprehensive evaluation"""
        print(f"\n{'='*60}")
        print(f"{dataset_name} Dataset - Run {run_id + 1}/{self.config['n_runs']}")
        print(f"{'='*60}")
        print(f"Processing {len(df)} claims...")
        
        # Set random seed for this run
        np.random.seed(self.config['random_seeds'][run_id])
        
        # Model predictions
        predictions = []
        for idx, row in df.iterrows():
            if idx % 25 == 0:  # Progress every 25 claims
                print(f"Processing claim {idx + 1}/{len(df)}")
            
            pred = self.classify_claim(row["claim"], row["domain"], run_id)
            predictions.append(pred)
            sleep(0.1)
        
        # Generate baselines (only on first run to avoid repetition)
        if run_id == 0:
            random_baseline = self.generate_baseline_predictions(df, 'random')
            majority_baseline = self.generate_baseline_predictions(df, 'majority')
            keyword_baseline = self.generate_baseline_predictions(df, 'keyword')
        else:
            # Reuse same baselines for statistical consistency
            np.random.seed(42)
            random_baseline = self.generate_baseline_predictions(df, 'random')
            majority_baseline = self.generate_baseline_predictions(df, 'majority')
            keyword_baseline = self.generate_baseline_predictions(df, 'keyword')
        
        # Create results dataframe
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['random_baseline'] = random_baseline
        results_df['majority_baseline'] = majority_baseline
        results_df['keyword_baseline'] = keyword_baseline
        results_df['run_id'] = run_id
        
        # Clean data - remove rows with missing predictions
        df_clean = results_df.dropna(subset=['label', 'prediction'])
        
        if len(df_clean) < len(results_df):
            print(f"⚠️ Dropped {len(results_df) - len(df_clean)} rows due to missing label or prediction.")
        
        print(f"Successfully classified {len(df_clean)}/{len(results_df)} claims")
        
        # Print results in your preferred format
        self.print_overall_results(df_clean, f"{dataset_name} Run {run_id + 1}")
        self.print_domain_results(df_clean, f"{dataset_name} Run {run_id + 1}")
        
        # Calculate metrics for statistical tracking
        y_true = df_clean['label']
        
        # Model metrics
        model_metrics = self.calculate_comprehensive_metrics(
            y_true, df_clean['prediction'], dataset_name, 'Model', run_id
        )
        if model_metrics:
            self.all_results.append(model_metrics)
        
        # Baseline metrics (only calculate once)
        if run_id == 0:
            for baseline_name, baseline_preds in [
                ('Random', random_baseline),
                ('Majority', majority_baseline), 
                ('Keyword', keyword_baseline)
            ]:
                baseline_df = results_df.copy()
                baseline_df['prediction'] = baseline_preds
                baseline_clean = baseline_df.dropna(subset=['label', 'prediction'])
                
                baseline_metrics = self.calculate_comprehensive_metrics(
                    baseline_clean['label'], baseline_clean['prediction'], 
                    dataset_name, baseline_name, 0
                )
                if baseline_metrics:
                    self.all_results.append(baseline_metrics)
        
        # Store detailed data for later analysis
        storage_key = f"{dataset_name}_run_{run_id}"
        self.run_data_storage[storage_key] = {
            'dataframe': df_clean,
            'dataset': dataset_name,
            'run_id': run_id
        }
        
        # Save results for this run
        output_file = f"{dataset_name.lower()}_run_{run_id + 1}_results.csv"
        df_clean.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        
        return df_clean
    
    def perform_statistical_analysis(self, dataset_name):
        """Perform statistical analysis across runs"""
        print(f"\n{'='*60}")
        print(f"STATISTICAL ANALYSIS: {dataset_name}")
        print(f"{'='*60}")
        
        # Get model results for this dataset
        model_results = [r for r in self.all_results 
                        if r['dataset'] == dataset_name and r['condition'] == 'Model']
        
        if len(model_results) == 0:
            print("No valid results for statistical analysis")
            return
        
        # Extract metrics across runs
        accuracies = [r['accuracy'] for r in model_results]
        precisions = [r['precision'] for r in model_results]
        recalls = [r['recall'] for r in model_results]
        f1_scores = [r['f1_score'] for r in model_results]
        
        # Calculate statistics
        print(f"\nModel Performance across {len(model_results)} runs:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"  Mean Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        print(f"  Mean Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        print(f"  Mean F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        if len(accuracies) > 1:
            # Confidence intervals
            acc_ci = stats.t.interval(0.95, len(accuracies)-1, loc=np.mean(accuracies), scale=stats.sem(accuracies))
            f1_ci = stats.t.interval(0.95, len(f1_scores)-1, loc=np.mean(f1_scores), scale=stats.sem(f1_scores))
            
            print(f"\n95% Confidence Intervals:")
            print(f"  Accuracy: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
            print(f"  F1-Score: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
        
        # Compare to baselines
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
                
                # Statistical significance test (if multiple runs)
                if len(accuracies) > 1:
                    t_stat, p_value = stats.ttest_1samp(accuracies, baseline_acc)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"    Statistical Significance: p={p_value:.4f} {significance}")
    
    def generate_aggregated_domain_analysis(self, dataset_name):
        """Generate aggregated domain analysis across all runs"""
        print(f"\n{'='*60}")
        print(f"AGGREGATED DOMAIN ANALYSIS: {dataset_name}")
        print(f"{'='*60}")
        
        # Collect all runs for this dataset
        dataset_runs = []
        for key, stored_data in self.run_data_storage.items():
            if stored_data['dataset'] == dataset_name:
                dataset_runs.append(stored_data['dataframe'])
        
        if len(dataset_runs) == 0:
            print("No data available for domain analysis")
            return
        
        # Combine all runs
        all_data = pd.concat(dataset_runs, ignore_index=True)
        
        # Get unique domains
        domains = all_data['domain'].unique()
        domains = sorted([d for d in domains if pd.notna(d)])
        
        print(f"\nDomains found: {', '.join(domains)}")
        print(f"Total runs aggregated: {len(dataset_runs)}")
        
        # Analyze each domain
        domain_summary = []
        
        for domain in domains:
            domain_data = all_data[all_data['domain'] == domain]
            
            if len(domain_data) == 0:
                continue
            
            try:
                accuracy = accuracy_score(domain_data['label'], domain_data['prediction'])
                precision = precision_score(domain_data['label'], domain_data['prediction'], average='macro', zero_division=0)
                recall = recall_score(domain_data['label'], domain_data['prediction'], average='macro', zero_division=0)
                f1 = f1_score(domain_data['label'], domain_data['prediction'], average='macro', zero_division=0)
                
                domain_summary.append({
                    'domain': domain,
                    'n_samples': len(domain_data),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
            except Exception as e:
                print(f"Error analyzing domain {domain}: {e}")
        
        # Sort by accuracy
        domain_summary = sorted(domain_summary, key=lambda x: x['accuracy'], reverse=True)
        
        # Print summary table
        print(f"\nAggregated Domain Performance (across {len(dataset_runs)} runs):")
        print("-" * 80)
        print(f"{'Domain':<20} {'N':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        for stats in domain_summary:
            print(f"{stats['domain']:<20} {stats['n_samples']:<6} "
                  f"{stats['accuracy']:<10.2f} {stats['precision']:<10.2f} "
                  f"{stats['recall']:<10.2f} {stats['f1_score']:<10.2f}")
        
        print("-" * 80)
        
        # Cross-domain statistics
        if len(domain_summary) > 1:
            accuracies = [s['accuracy'] for s in domain_summary]
            f1_scores = [s['f1_score'] for s in domain_summary]
            
            print(f"\nCross-Domain Statistics:")
            print(f"  Mean Domain Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
            print(f"  Mean Domain F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
            print(f"  Best Domain: {domain_summary[0]['domain']} ({domain_summary[0]['accuracy']:.3f})")
            print(f"  Worst Domain: {domain_summary[-1]['domain']} ({domain_summary[-1]['accuracy']:.3f})")
            print(f"  Domain Accuracy Range: {max(accuracies) - min(accuracies):.3f}")
        
        # Save domain analysis
        domain_df = pd.DataFrame(domain_summary)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_file = f"{dataset_name.lower()}_aggregated_domain_analysis_{timestamp}.csv"
        domain_df.to_csv(domain_file, index=False)
        print(f"\nDomain analysis saved to: {domain_file}")
        
        return domain_summary
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n{'='*60}")
        print("FINAL THESIS RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # Overall model performance
        model_results = [r for r in self.all_results if r['condition'] == 'Model']
        
        if len(model_results) == 0:
            print("No results to report")
            return
        
        # Group by dataset
        for dataset in ['UGC', 'NGC']:
            dataset_results = [r for r in model_results if r['dataset'] == dataset]
            
            if len(dataset_results) > 0:
                accuracies = [r['accuracy'] for r in dataset_results]
                f1_scores = [r['f1_score'] for r in dataset_results]
                precisions = [r['precision'] for r in dataset_results]
                recalls = [r['recall'] for r in dataset_results]
                
                print(f"\n{dataset} Dataset Summary:")
                print(f"  Runs completed: {len(dataset_results)}")
                print(f"  Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
                print(f"  Mean Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
                print(f"  Mean Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
                print(f"  Mean F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
                print(f"  Best Run Accuracy: {max(accuracies):.3f}")
                print(f"  Cohen's Kappa: {np.mean([r['cohen_kappa'] for r in dataset_results]):.3f}")
        
        # Save detailed results
        results_df = pd.DataFrame(self.all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"thesis_comprehensive_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nComprehensive results saved to: {results_file}")
        
        return results_df
    
    def clean_datasets(self):
        """Clean datasets by removing rows with invalid claims"""
        print("Cleaning datasets...")
        
        # Load and clean UGC
        ugc_df = pd.read_csv(self.config["ugc_file"])
        ugc_before = len(ugc_df)
        ugc_df = ugc_df.dropna(subset=['claim', 'label', 'domain'])
        ugc_df = ugc_df[ugc_df['claim'].astype(str).str.strip() != '']
        ugc_after = len(ugc_df)
        print(f"UGC: {ugc_before} → {ugc_after} rows (removed {ugc_before - ugc_after})")
        
        # Load and clean NGC
        ngc_df = pd.read_csv(self.config["ngc_file"])
        ngc_before = len(ngc_df)
        ngc_df = ngc_df.dropna(subset=['claim', 'label', 'domain'])
        ngc_df = ngc_df[ngc_df['claim'].astype(str).str.strip() != '']
        ngc_after = len(ngc_df)
        print(f"NGC: {ngc_before} → {ngc_after} rows (removed {ngc_before - ngc_after})")
        
        return ugc_df, ngc_df
    
    def run_complete_evaluation(self):
        """Run the complete thesis evaluation"""
        try:
            # Clean datasets first
            ugc_df, ngc_df = self.clean_datasets()
            
            print(f"\nFinal datasets - UGC: {len(ugc_df)} samples, NGC: {len(ngc_df)} samples")
            
            # Run multiple experiments
            for run_id in range(self.config["n_runs"]):
                print(f"\n{'#'*60}")
                print(f"# EXPERIMENTAL RUN {run_id + 1}/{self.config['n_runs']}")
                print(f"{'#'*60}")
                
                # UGC Dataset
                self.run_single_experiment(ugc_df, "UGC", run_id)
                
                # NGC Dataset
                self.run_single_experiment(ngc_df, "NGC", run_id)
            
            # Statistical analysis
            print(f"\n{'#'*60}")
            print("# STATISTICAL ANALYSIS ACROSS RUNS")
            print(f"{'#'*60}")
            
            self.perform_statistical_analysis("UGC")
            self.perform_statistical_analysis("NGC")
            
            # Aggregated domain analysis
            print(f"\n{'#'*60}")
            print("# AGGREGATED DOMAIN ANALYSIS")
            print(f"{'#'*60}")
            
            self.generate_aggregated_domain_analysis("UGC")
            self.generate_aggregated_domain_analysis("NGC")
            
            # Final report
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

# Main execution
if __name__ == "__main__":
    evaluator = SimpleThesisEvaluator(CONFIG)
    results = evaluator.run_complete_evaluation()