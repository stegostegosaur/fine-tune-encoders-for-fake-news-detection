import argparse
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import os
import logging
import json

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(pred):
    """Computes and returns a dictionary of metrics from predictions."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer model on a test set.")

    # Model and Data arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the trained model and tokenizer are saved.")
    parser.add_argument("--test_file", type=str, default="data/processed/test.csv", help="Path to the test data CSV file.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the column containing labels.")

    # Output arguments
    parser.add_argument("--results_output_file", type=str, default=None, help="Optional path to save the evaluation results as a JSON file. If not provided, defaults to 'test_results.json' inside model_dir.")

    # Evaluation arguments
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation.")

    args = parser.parse_args()

    logger.info(f"Starting evaluation with arguments: {args}")

    # --- Check if model directory exists ---
    if not os.path.isdir(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return

    # --- Determine output file path ---
    if args.results_output_file:
        output_file = args.results_output_file
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        output_file = os.path.join(args.model_dir, "test_results.json")

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading model and tokenizer from: {args.model_dir}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except OSError:
        logger.error(f"Could not load model/tokenizer from {args.model_dir}. Ensure it's a valid Hugging Face model directory.")
        return

    # --- Load Test Data ---
    logger.info(f"Loading test data from: {args.test_file}")
    try:
        test_df = pd.read_csv(args.test_file)
    except FileNotFoundError:
        logger.error(f"Test file not found: {args.test_file}")
        return

    if args.text_column not in test_df.columns or args.label_column not in test_df.columns:
        logger.error(f"Text column ('{args.text_column}') or label column ('{args.label_column}') not found in test file.")
        return

    test_dataset = Dataset.from_pandas(test_df)

    # --- Map labels to IDs using the model's config ---
    label2id = model.config.label2id
    def map_labels(example):
        example[args.label_column] = label2id[example[args.label_column]]
        return example

    try:
        test_dataset = test_dataset.map(map_labels, batched=False)
        logger.info(f"Labels mapped using model's label2id config: {label2id}")
    except KeyError as e:
        logger.error(f"Label '{e.args[0]}' in test set is not present in the model's training data. Cannot map labels.")
        return

    # --- Tokenize Data ---
    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], padding="max_length", truncation=True, max_length=512)

    logger.info("Tokenizing test dataset...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Rename label column to 'labels' as expected by Trainer
    if args.label_column != 'labels':
        tokenized_test_dataset = tokenized_test_dataset.rename_column(args.label_column, "labels")

    # --- Setup Trainer for Prediction ---
    # We only need minimal TrainingArguments for evaluation
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "test_evaluation"), # Temporary directory for logs
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to="none" # Disable reporting for simple prediction
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # --- Make Predictions ---
    logger.info("Running predictions on the test set...")
    try:
        predictions = trainer.predict(tokenized_test_dataset)
        metrics = predictions.metrics
        logger.info(f"Evaluation Metrics: {metrics}")
    except Exception as e:
        logger.exception(f"An error occurred during prediction: {e}")
        return

    # --- Save Results ---
    if metrics:
        logger.info(f"Saving test results to: {output_file}")
        try:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info("Results saved successfully.")
        except IOError as e:
            logger.error(f"Failed to save results to {output_file}: {e}")

if __name__ == "__main__":
    main()
