import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef
from scipy import stats
import numpy as np
import json
import hashlib
from time import sleep
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Thesis-Grade Misinformation Testing Framework v1.0")
print(f"Experiment started: {datetime.now()}")

# Configuration
CONFIG = {
    "ugc_file": "ugc_master.csv",
    "ngc_file": "ngc_master.csv", 
    "model_name": "llama-3.3-70b-instruct",
    "api_url": "http://127.0.0.1:1234/v1/completions",
    "n_runs": 5,  # Multiple runs for statistical reliability
    "random_seeds": [42, 123, 456, 789, 999],
    "output_dir": "thesis_results",
    "confidence_level": 0.95
}

class MisinformationEvaluator:
    def __init__(self, config):
        self.config = config
        self.results_log = []
        
    def set_random_seed(self, seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        
    def generate_baseline_predictions(self, df, method='random'):
        """Generate baseline predictions for comparison"""
        if method == 'random':
            return np.random.choice([0, 1], size=len(df))
        elif method == 'majority':
            majority_class = df['label'].mode()[0]
            return [majority_class] * len(df)
        elif method == 'keyword':
            # Simple keyword-based heuristic
            predictions = []
            suspicious_keywords = ['breaking', 'shocking', 'revealed', 'secret', 'you won\'t believe']
            for claim in df['claim']:
                claim_lower = claim.lower()
                if any(keyword in claim_lower for keyword in suspicious_keywords):
                    predictions.append(0)  # Likely false
                else:
                    predictions.append(1)  # Likely true
            return predictions
        
    def calculate_confidence_interval(self, data, confidence=0.95):
        """Bootstrap confidence interval"""
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        return lower, upper
    
    def mcnemar_test(self, y_true, pred1, pred2):
        """McNemar's test for comparing two models"""
        # Create contingency table
        pred1_correct = (pred1 == y_true)
        pred2_correct = (pred2 == y_true)
        
        # McNemar's test focuses on disagreements
        b = np.sum(pred1_correct & ~pred2_correct)  # Model 1 right, Model 2 wrong
        c = np.sum(~pred1_correct & pred2_correct)  # Model 1 wrong, Model 2 right
        
        if b + c == 0:
            return 1.0  # No disagreements
        
        # McNemar's statistic with continuity correction
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        return p_value
    
    def analyze_failures(self, df, predictions, label_name):
        """Detailed error analysis"""
        df_analysis = df.copy()
        df_analysis['prediction'] = predictions
        df_analysis['correct'] = df_analysis['label'] == df_analysis['prediction']
        
        # Overall accuracy by domain
        domain_accuracy = df_analysis.groupby('domain')['correct'].agg(['mean', 'count'])
        
        # False positive and false negative analysis
        false_positives = df_analysis[(df_analysis['label'] == 0) & (df_analysis['prediction'] == 1)]
        false_negatives = df_analysis[(df_analysis['label'] == 1) & (df_analysis['prediction'] == 0)]
        
        # Claim length analysis
        df_analysis['claim_length'] = df_analysis['claim'].str.len()
        length_bins = pd.qcut(df_analysis['claim_length'], q=4, labels=['Short', 'Medium', 'Long', 'Very Long'])
        length_accuracy = df_analysis.groupby(length_bins)['correct'].mean()
        
        analysis_results = {
            'domain_accuracy': domain_accuracy,
            'false_positives_count': len(false_positives),
            'false_negatives_count': len(false_negatives),
            'length_accuracy': length_accuracy,
            'false_positive_examples': false_positives['claim'].head(5).tolist(),
            'false_negative_examples': false_negatives['claim'].head(5).tolist()
        }
        
        return analysis_results
    
    def classify_claim_with_uncertainty(self, claim, domain, run_id=0):
        """Enhanced classification with uncertainty handling - Fixed for instruction-tuned models"""
        prompt = f"""You are a fact-checking assistant. Your task is to determine whether the given claim is factually accurate or not.

Context: This claim appeared in {domain} content.
Claim: "{claim}"

Instructions:
- Classify the claim as either "True" or "False" based on factual accuracy
- Indicate your confidence level as "High", "Medium", or "Low"
- Provide a brief justification for your decision

Format your response exactly as follows:
Classification: [True/False]
Confidence: [High/Medium/Low]
Justification: [Brief explanation]"""

        try:
            response = requests.post(
                self.config["api_url"],
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.config["model_name"],
                    "prompt": prompt,
                    "max_tokens": 100,  # Reduced for direct response
                    "temperature": 0.0,  # More deterministic for instruction following
                    "stop": ["\n\nContext:", "\n\nClaim:", "Classification:", "\n\n"],
                    "seed": self.config["random_seeds"][run_id % len(self.config["random_seeds"])]
                }
            )

            if response.status_code != 200:
                print(f"API request failed with status code: {response.status_code}")
                return None, None, None

            # Get the response and handle potential data type issues
            try:
                response_json = response.json()
                full_response = response_json["choices"][0]["text"].strip()
                
                # Debug info to catch data type issues
                print(f"Debug - Response type: {type(full_response)}")
                print(f"Debug - Response content: {repr(full_response[:100])}")
                
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error extracting response: {e}")
                print(f"Raw response: {response.text}")
                return None, None, None
            
            # Extract classification and confidence
            classification = self.extract_classification(full_response)
            confidence = self.extract_confidence(full_response)
            
            return classification, confidence, full_response

        except Exception as e:
            print(f"Error in classification: {e}")
            return None, None, None
    
    def extract_classification(self, response):
        """Extract True/False from structured response - FIXED VERSION"""
        # Handle None or non-string responses
        if response is None:
            print("Warning: Received None response")
            return None
        
        # Convert to string if it's not already
        if not isinstance(response, str):
            print(f"Warning: Converting {type(response)} to string: {response}")
            response = str(response)
        
        response_lower = response.lower()
        
        # Look for structured format first
        if 'classification: true' in response_lower:
            return 1
        elif 'classification: false' in response_lower:
            return 0
        
        # Fallback to simple pattern matching
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('classification:'):
                if 'true' in line_lower:
                    return 1
                elif 'false' in line_lower:
                    return 0
        
        # Last resort - look anywhere in response
        if 'true' in response_lower and 'false' not in response_lower:
            return 1
        elif 'false' in response_lower and 'true' not in response_lower:
            return 0
        
        print(f"Warning: Could not extract classification from: {response[:100]}")
        return None
    
    def extract_confidence(self, response):
        """Extract confidence level from structured response - FIXED VERSION"""
        # Handle None or non-string responses
        if response is None:
            return 'Unknown'
        
        # Convert to string if it's not already
        if not isinstance(response, str):
            response = str(response)
        
        response_lower = response.lower()
        
        # Look for structured format first
        confidence_patterns = ['confidence: high', 'confidence: medium', 'confidence: low']
        for pattern in confidence_patterns:
            if pattern in response_lower:
                return pattern.split(': ')[1].title()
        
        # Fallback to line-by-line search
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('confidence:'):
                for level in ['high', 'medium', 'low']:
                    if level in line_lower:
                        return level.title()
        
        return 'Unknown'
    
    def run_single_experiment(self, df, dataset_name, run_id):
        """Run a single experimental run"""
        print(f"Running {dataset_name} - Run {run_id + 1}/{self.config['n_runs']}")
        
        self.set_random_seed(self.config['random_seeds'][run_id])
        
        # Model predictions
        predictions = []
        confidences = []
        reasoning_logs = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"  Processing claim {idx + 1}/{len(df)}")
            
            try:
                pred, conf, reasoning = self.classify_claim_with_uncertainty(
                    row["claim"], row["domain"], run_id
                )
                predictions.append(pred)
                confidences.append(conf)
                reasoning_logs.append(reasoning)
                
            except Exception as e:
                print(f"Error processing claim {idx + 1}: {e}")
                predictions.append(None)
                confidences.append('Unknown')
                reasoning_logs.append(str(e))
            
            sleep(0.15)
        
        # Generate baselines
        random_baseline = self.generate_baseline_predictions(df, 'random')
        majority_baseline = self.generate_baseline_predictions(df, 'majority')
        keyword_baseline = self.generate_baseline_predictions(df, 'keyword')
        
        # Clean data
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['confidence'] = confidences
        results_df['reasoning'] = reasoning_logs
        results_df['random_baseline'] = random_baseline
        results_df['majority_baseline'] = majority_baseline
        results_df['keyword_baseline'] = keyword_baseline
        
        # Remove rows with missing predictions
        clean_df = results_df.dropna(subset=['label', 'prediction'])
        
        if len(clean_df) < len(results_df):
            print(f"  ⚠️ Dropped {len(results_df) - len(clean_df)} rows due to missing predictions")
        
        return clean_df
    
    def calculate_metrics(self, y_true, y_pred, dataset_name, condition_name):
        """Calculate comprehensive evaluation metrics"""
        # Convert to numpy arrays and handle potential data type issues
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Remove any remaining NaN values
        valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
        
        if len(y_true) == 0:
            print(f"Warning: No valid predictions for {dataset_name} {condition_name}")
            return None
        
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Advanced metrics
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Detailed classification report
            report = classification_report(y_true, y_pred, output_dict=True, 
                                         target_names=["False", "True"], zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            return {
                'dataset': dataset_name,
                'condition': condition_name,
                'accuracy': accuracy,
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
        except Exception as e:
            print(f"Error calculating metrics for {dataset_name} {condition_name}: {e}")
            return {
                'dataset': dataset_name,
                'condition': condition_name,
                'accuracy': accuracy,
                'cohen_kappa': 0,
                'mcc': 0,
                'precision_false': 0,
                'recall_false': 0,
                'f1_false': 0,
                'precision_true': 0,
                'recall_true': 0,
                'f1_true': 0,
                'macro_f1': 0,
                'weighted_f1': 0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'n_samples': len(y_true)
            }
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation framework"""
        try:
            # Load datasets
            print("Loading datasets...")
            ugc_df = pd.read_csv(self.config["ugc_file"])
            ngc_df = pd.read_csv(self.config["ngc_file"])
            
            print(f"Loaded UGC: {len(ugc_df)} samples")
            print(f"Loaded NGC: {len(ngc_df)} samples")
            
            # Verify required columns exist
            required_cols = ['claim', 'label', 'domain']
            for df_name, df in [('UGC', ugc_df), ('NGC', ngc_df)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Error: {df_name} missing columns: {missing_cols}")
                    print(f"Available columns: {list(df.columns)}")
                    return None
            
            all_results = []
            all_run_data = []
            
            # Run multiple experiments
            for run_id in range(self.config["n_runs"]):
                print(f"\n=== EXPERIMENTAL RUN {run_id + 1} ===")
                
                try:
                    # UGC Dataset
                    ugc_results = self.run_single_experiment(ugc_df, "UGC", run_id)
                    all_run_data.append(('UGC', run_id, ugc_results))
                    
                    # NGC Dataset  
                    ngc_results = self.run_single_experiment(ngc_df, "NGC", run_id)
                    all_run_data.append(('NGC', run_id, ngc_results))
                    
                except Exception as e:
                    print(f"Error in run {run_id + 1}: {e}")
                    continue
            
            # Aggregate results across runs
            print("\n=== AGGREGATING RESULTS ===")
            
            datasets = {'UGC': [], 'NGC': []}
            for dataset_name, run_id, data in all_run_data:
                datasets[dataset_name].append(data)
            
            # Calculate metrics for each run and condition
            for dataset_name, run_datasets in datasets.items():
                dataset_metrics = []
                
                for run_id, run_data in enumerate(run_datasets):
                    if len(run_data) == 0:
                        continue
                        
                    y_true = run_data['label'].values
                    
                    # Model performance
                    model_metrics = self.calculate_metrics(
                        y_true, run_data['prediction'].values, 
                        dataset_name, 'Model'
                    )
                    if model_metrics:
                        model_metrics['run_id'] = run_id
                        dataset_metrics.append(model_metrics)
                    
                    # Baseline comparisons (only for first run to avoid repetition)
                    if run_id == 0:
                        for baseline_name in ['random_baseline', 'majority_baseline', 'keyword_baseline']:
                            baseline_metrics = self.calculate_metrics(
                                y_true, run_data[baseline_name].values,
                                dataset_name, baseline_name.replace('_baseline', '').title()
                            )
                            if baseline_metrics:
                                baseline_metrics['run_id'] = 0
                                dataset_metrics.append(baseline_metrics)
                
                all_results.extend(dataset_metrics)
                
                # Statistical analysis
                if len(run_datasets) > 0:
                    self.perform_statistical_analysis(run_datasets, dataset_name)
                    
                    # Error analysis (on first run)
                    if len(run_datasets) > 0:
                        self.perform_error_analysis(run_datasets[0], dataset_name)
            
            # Save comprehensive results
            if len(all_results) > 0:
                results_df = pd.DataFrame(all_results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"comprehensive_results_{timestamp}.csv"
                results_df.to_csv(results_file, index=False)
                
                # Generate summary report
                self.generate_summary_report(results_df)
                
                print(f"\n=== EVALUATION COMPLETE ===")
                print(f"Results saved to {results_file}")
                
                return results_df
            else:
                print("No results to save - all runs may have failed")
                return None
                
        except Exception as e:
            print(f"Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def perform_statistical_analysis(self, run_datasets, dataset_name):
        """Perform statistical significance testing"""
        print(f"\n--- Statistical Analysis: {dataset_name} ---")
        
        # Collect accuracies across runs
        model_accuracies = []
        baseline_accuracies = []
        
        for run_data in run_datasets:
            if len(run_data) == 0:
                continue
                
            y_true = run_data['label'].values
            
            # Only calculate if we have valid predictions
            valid_predictions = run_data['prediction'].dropna()
            if len(valid_predictions) == 0:
                continue
                
            model_acc = np.mean(y_true == run_data['prediction'].values)
            random_acc = np.mean(y_true == run_data['random_baseline'].values)
            
            model_accuracies.append(model_acc)
            baseline_accuracies.append(random_acc)
        
        if len(model_accuracies) == 0:
            print("No valid accuracies to analyze")
            return
            
        # Statistical tests
        model_mean = np.mean(model_accuracies)
        model_std = np.std(model_accuracies)
        baseline_mean = np.mean(baseline_accuracies)
        
        # Confidence interval for model performance
        if len(model_accuracies) > 1:
            ci_lower, ci_upper = self.calculate_confidence_interval(model_accuracies)
            
            # Paired t-test vs baseline
            if len(model_accuracies) > 1 and len(baseline_accuracies) > 1:
                t_stat, p_value = stats.ttest_rel(model_accuracies, baseline_accuracies)
            else:
                t_stat, p_value = 0, 1
        else:
            ci_lower = ci_upper = model_mean
            t_stat, p_value = 0, 1
        
        print(f"Model Accuracy: {model_mean:.3f} ± {model_std:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"Baseline Accuracy: {baseline_mean:.3f}")
        print(f"T-test vs Random Baseline: t={t_stat:.3f}, p={p_value:.3f}")
        
        # Effect size (Cohen's d)
        if model_std > 0 and np.std(baseline_accuracies) > 0:
            pooled_std = np.sqrt(((len(model_accuracies)-1) * model_std**2 + 
                                 (len(baseline_accuracies)-1) * np.std(baseline_accuracies)**2) / 
                                (len(model_accuracies) + len(baseline_accuracies) - 2))
            if pooled_std > 0:
                cohens_d = (model_mean - baseline_mean) / pooled_std
                print(f"Effect size (Cohen's d): {cohens_d:.3f}")
        
        # McNemar's test between model and keyword baseline
        if len(run_datasets) > 0:
            first_run = run_datasets[0]
            try:
                mcnemar_p = self.mcnemar_test(
                    first_run['label'].values,
                    first_run['prediction'].values,
                    first_run['keyword_baseline'].values
                )
                print(f"McNemar's test vs Keyword Baseline: p={mcnemar_p:.3f}")
            except:
                print("Could not perform McNemar's test")
    
    def perform_error_analysis(self, run_data, dataset_name):
        """Detailed error analysis"""
        print(f"\n--- Error Analysis: {dataset_name} ---")
        
        try:
            analysis = self.analyze_failures(run_data, run_data['prediction'].values, dataset_name)
            
            print("Domain-wise Accuracy:")
            print(analysis['domain_accuracy'])
            
            print(f"\nError Distribution:")
            print(f"False Positives: {analysis['false_positives_count']}")
            print(f"False Negatives: {analysis['false_negatives_count']}")
            
            print(f"\nAccuracy by Claim Length:")
            print(analysis['length_accuracy'])
            
            print(f"\nSample False Positive Claims:")
            for i, claim in enumerate(analysis['false_positive_examples'][:3], 1):
                print(f"{i}. {claim[:100]}...")
            
            print(f"\nSample False Negative Claims:")
            for i, claim in enumerate(analysis['false_negative_examples'][:3], 1):
                print(f"{i}. {claim[:100]}...")
            
            # Confidence analysis
            if 'confidence' in run_data.columns:
                conf_accuracy = run_data.groupby('confidence').apply(
                    lambda x: np.mean(x['label'] == x['prediction']) if len(x) > 0 else 0
                )
                print(f"\nAccuracy by Confidence Level:")
                print(conf_accuracy)
                
        except Exception as e:
            print(f"Error in error analysis: {e}")
    
    def generate_summary_report(self, results_df):
        """Generate executive summary of results"""
        print(f"\n{'='*60}")
        print("EXECUTIVE SUMMARY")
        print(f"{'='*60}")
        
        # Model performance summary
        model_results = results_df[results_df['condition'] == 'Model']
        
        for dataset in ['UGC', 'NGC']:
            dataset_results = model_results[model_results['dataset'] == dataset]
            
            if len(dataset_results) > 0:
                mean_acc = dataset_results['accuracy'].mean()
                std_acc = dataset_results['accuracy'].std()
                mean_f1 = dataset_results['macro_f1'].mean()
                
                print(f"\n{dataset} Dataset:")
                print(f"  Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
                print(f"  Mean Macro F1: {mean_f1:.3f}")
                
                # Best performance
                best_run = dataset_results.loc[dataset_results['accuracy'].idxmax()]
                print(f"  Best Run Accuracy: {best_run['accuracy']:.3f}")
        
        # Baseline comparison
        print(f"\nBaseline Comparisons:")
        baselines = results_df[results_df['condition'].isin(['Random', 'Majority', 'Keyword'])]
        for baseline_type in ['Random', 'Majority', 'Keyword']:
            baseline_data = baselines[baselines['condition'] == baseline_type]
            if len(baseline_data) > 0:
                mean_acc = baseline_data['accuracy'].mean()
                print(f"  {baseline_type} Baseline: {mean_acc:.3f}")

# Main execution
if __name__ == "__main__":
    evaluator = MisinformationEvaluator(CONFIG)
    results = evaluator.run_comprehensive_evaluation()