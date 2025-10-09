import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os

print("BERT Fine-tuning Script for Misinformation Detection")

# File paths
UGC_FILE = "ugc_master.csv"
NGC_FILE = "ngc_master.csv"

# Model configuration
MODEL_NAME = "bert-base-uncased"  
MAX_LENGTH = 512
BATCH_SIZE = 8  # Smaller batch size for CPU training
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

class MisinformationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_and_print_bert(test_df, predictions, label):
    """Calculate and print metrics similar to your original script"""
    
    # Calculate overall metrics
    accuracy = accuracy_score(test_df["label"], predictions)
    precision = precision_score(test_df["label"], predictions, average='weighted')
    recall = recall_score(test_df["label"], predictions, average='weighted')
    f1 = f1_score(test_df["label"], predictions, average='weighted')
    
    print(f"\nOverall Results for {label} (Weighted Averages):")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\nDetailed Classification Report for {label}:")
    print(classification_report(test_df["label"], predictions, target_names=["false", "true"]))
    print("Confusion Matrix:")
    print(confusion_matrix(test_df["label"], predictions))
    
    # Domain-specific results
    print(f"\nDomain-Specific Results for {label}:")
    domains = test_df["domain"].unique()
    for domain in domains:
        print(f"\n{domain.capitalize()}:")
        subset_mask = test_df["domain"] == domain
        subset_labels = test_df[subset_mask]["label"]
        subset_predictions = predictions[subset_mask]
        
        if len(subset_labels) > 0:
            domain_accuracy = accuracy_score(subset_labels, subset_predictions)
            domain_precision = precision_score(subset_labels, subset_predictions, average='weighted')
            domain_recall = recall_score(subset_labels, subset_predictions, average='weighted')
            domain_f1 = f1_score(subset_labels, subset_predictions, average='weighted')
            
            print(f"Accuracy:  {domain_accuracy:.4f} ({domain_accuracy*100:.2f}%)")
            print(f"Precision: {domain_precision:.4f} ({domain_precision*100:.2f}%)")
            print(f"Recall:    {domain_recall:.4f} ({domain_recall*100:.2f}%)")
            print(f"F1-Score:  {domain_f1:.4f} ({domain_f1*100:.2f}%)")
            
            print("\nDetailed breakdown:")
            print(classification_report(subset_labels, subset_predictions, target_names=["false", "true"]))
            print("Confusion Matrix:")
            print(confusion_matrix(subset_labels, subset_predictions))
        else:
            print("No data available for this domain")
    
    # Save results
    test_df_copy = test_df.copy()
    test_df_copy["prediction"] = predictions
    output_file = f"{label.lower()}_bert_results.csv"
    test_df_copy.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")

def main():
    try:
        # Load and combine datasets
        print("Loading datasets...")
        ugc_df = pd.read_csv(UGC_FILE)
        ngc_df = pd.read_csv(NGC_FILE)
        
        # Combine datasets
        combined_df = pd.concat([ugc_df, ngc_df], ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} samples")
        print(f"Class distribution: {combined_df['label'].value_counts().to_dict()}")
        
        # Split data (stratified to maintain class balance)
        train_df, test_df = train_test_split(
            combined_df, 
            test_size=0.3, 
            stratify=combined_df['label'], 
            random_state=42
        )
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        # Load tokenizer and model
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        
        # Create datasets
        train_dataset = MisinformationDataset(
            train_df['claim'].tolist(),
            train_df['label'].tolist(),
            tokenizer,
            MAX_LENGTH
        )
        
        test_dataset = MisinformationDataset(
            test_df['claim'].tolist(),
            test_df['label'].tolist(),
            tokenizer,
            MAX_LENGTH
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./bert_misinformation_model',
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            use_cpu=True,  # Force CPU usage
            dataloader_num_workers=0,  # Avoid multiprocessing issues on Mac
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Evaluate on test set
        print("Evaluating on test set...")
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        
        # Print results in format
        evaluate_and_print_bert(test_df.reset_index(drop=True), predicted_labels, "BERT Combined")
        
        # Save the model
        model.save_pretrained('./bert_misinformation_final')
        tokenizer.save_pretrained('./bert_misinformation_final')
        print("Model saved to './bert_misinformation_final'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset file - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()