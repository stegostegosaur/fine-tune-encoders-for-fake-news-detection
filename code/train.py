import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfFolder,
    notebook_login,
)
from datasets import Dataset, DatasetDict
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def freeze_layers(model, model_type, freeze_mode, num_total_layers_bert=12, num_total_layers_distilbert=6):

    if freeze_mode == "none": # Full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

        return

    if model_type == 'bert':
        encoder_layers = model.bert.encoder.layer
        embeddings = model.bert.embeddings
        num_total_layers = num_total_layers_bert
    elif model_type == 'distilbert':
        encoder_layers = model.distilbert.transformer.layer
        embeddings = model.distilbert.embeddings
        num_total_layers = num_total_layers_distilbert
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    for param in embeddings.parameters():
        param.requires_grad = False

    if freeze_mode == "all":
        num_layers_to_freeze = num_total_layers
    elif freeze_mode == "half":
        num_layers_to_freeze = num_total_layers // 2
    else:
        raise ValueError(f"Unsupported freeze mode: {freeze_mode}. Choose 'all', 'half', or 'none'.")


    for i, layer in enumerate(encoder_layers):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters(): # Ensure subsequent layers are trainable
                param.requires_grad = True

    # Ensure the classifier head is always trainable
    trainable_classifier = False
    if hasattr(model, 'classifier') and model.classifier is not None:
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable_classifier = True
    if hasattr(model, 'pre_classifier') and model.pre_classifier is not None: # For DistilBERT
         for param in model.pre_classifier.parameters():
            param.requires_grad = True
         trainable_classifier = True

    if not trainable_classifier:
        logger.warning("Could not find a standard classifier head to ensure it's trainable.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted') # Use 'binary' if specifically for binary tasks and want single class F1
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def main():
    parser = argparse.ArgumentParser(description="Train a transformer model for text classification.")
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased", help="Hugging Face model name or path.")
    parser.add_argument("--model_type", type=str, choices=['bert', 'distilbert'], required=True, help="Specify 'bert' or 'distilbert' for layer freezing logic.")
    parser.add_argument("--freeze_mode", type=str, choices=['all', 'half', 'none'], default='none', help="Layer freezing strategy: 'all', 'half', or 'none'.")

    # Data arguments
    parser.add_argument("--train_file", type=str, default="data/processed/train.csv", help="Path to the training data CSV file.")
    parser.add_argument("--eval_file", type=str, default=None, help="Optional path to the evaluation data CSV file.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the column containing labels.")
    parser.add_argument("--validation_split_size", type=float, default=0.1, help="If no eval_file, proportion of train to use for validation.")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results/model_output", help="Directory to save training outputs (model, logs).")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every X steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps if evaluation_strategy is 'steps'.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default="epoch", help="Evaluation strategy.")
    parser.add_argument("--save_strategy", type=str, choices=["steps", "epoch", "no"], default="epoch", help="Save strategy.")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training.")
    parser.add_argument("--metric_for_best_model", type=str, default="f1", help="Metric to use for best model selection.")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (requires compatible GPU).")

    # Hugging Face Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub after training.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository ID on Hugging Face Hub (e.g., your_username/your_model_name).")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face Hub token (optional, can be set via CLI login).")

    args = parser.parse_args()



    # --- Hugging Face Hub Login (if pushing to hub) ---
    if args.push_to_hub:
        if args.hub_model_id is None:
            logger.error("hub_model_id must be specified if push_to_hub is True.")
            return

        hub_token_to_use = args.hub_token or HfFolder.get_token()
        if not hub_token_to_use:

            try:
                notebook_login() # For interactive environments like Colab/Jupyter
            except Exception as e:
                logger.warning(f"Notebook login failed: {e}. Please login via 'huggingface-cli login' or provide --hub_token.")
                # Decide if to proceed without pushing or exit
                logger.warning("Proceeding without pushing to Hub as login failed.")
                args.push_to_hub = False # Disable pushing if login fails

    # --- Load Data ---
    try:
        train_df = pd.read_csv(args.train_file)
    except FileNotFoundError:
        logger.error(f"Training file not found: {args.train_file}")
        return

    if args.eval_file:
        try:
            eval_df = pd.read_csv(args.eval_file)
        except FileNotFoundError:
            logger.error(f"Evaluation file not found: {args.eval_file}")
            return
        # Create DatasetDict
        raw_datasets = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(eval_df)
        })
    else:
        if len(train_df) < 2 or args.validation_split_size <=0 or args.validation_split_size >=1:
             logger.warning("Training data too small or invalid validation_split_size. Proceeding without validation set.")
             raw_datasets = DatasetDict({'train': Dataset.from_pandas(train_df)})
             args.evaluation_strategy = "no" # Disable evaluation if no validation set
        else:
            train_pandas_df, eval_pandas_df = sklearn_train_test_split(
                train_df, test_size=args.validation_split_size, random_state=42,
                stratify=train_df[args.label_column] if args.label_column in train_df.columns else None
            )
            raw_datasets = DatasetDict({
                'train': Dataset.from_pandas(train_pandas_df),
                'validation': Dataset.from_pandas(eval_pandas_df)
            })


    # --- Determine num_labels ---
    # Ensure labels are 0-indexed if they are not already
    unique_labels = train_df[args.label_column].unique()
    label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(unique_labels)

    if num_labels < 2:
        logger.error(f"Not enough unique labels found in column '{args.label_column}'. Found {num_labels}. Need at least 2.")
        return

    def map_labels(example):
        example[args.label_column] = label2id[example[args.label_column]]
        return example

    raw_datasets = raw_datasets.map(map_labels, batched=False)


    # --- Load Tokenizer ---

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], padding="max_length", truncation=True, max_length=512)


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=[args.text_column] if args.text_column in raw_datasets['train'].column_names else None)
    # The Trainer expects the label column to be named 'labels'
    if args.label_column != 'labels':
         tokenized_datasets = tokenized_datasets.rename_column(args.label_column, "labels")


    if 'validation' not in tokenized_datasets and args.evaluation_strategy != "no":
        logger.warning(f"Validation set not available, but evaluation strategy is '{args.evaluation_strategy}'. Disabling evaluation.")
        args.evaluation_strategy = "no"

    # --- Load Model ---

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # --- Apply Layer Freezing ---
    if args.freeze_mode != 'none':
        freeze_layers(model, args.model_type, args.freeze_mode)
    else:

        # Ensure all parameters are trainable if freeze_mode is 'none' explicitly
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())



    # --- Training Arguments ---
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output_dir exists

    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "logging_dir": os.path.join(args.output_dir, 'logs'), # Specify logging directory
        "logging_steps": args.logging_steps,
        "evaluation_strategy": args.evaluation_strategy if 'validation' in tokenized_datasets else "no",
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "load_best_model_at_end": args.load_best_model_at_end if 'validation' in tokenized_datasets else False,
        "metric_for_best_model": args.metric_for_best_model if 'validation' in tokenized_datasets else None,
        "greater_is_better": True if args.metric_for_best_model in ["accuracy", "f1", "precision", "recall"] else None,
        "fp16": args.fp16,
        "report_to": "tensorboard", # Default logger, can add "wandb", "mlflow"
    }
    if args.push_to_hub:
        training_args_dict["push_to_hub"] = True
        training_args_dict["hub_model_id"] = args.hub_model_id
        if args.hub_token: # Only pass if explicitly provided, otherwise relies on global login
             training_args_dict["hub_token"] = args.hub_token

    training_args = TrainingArguments(**training_args_dict)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"), # Use .get for safety
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if tokenized_datasets.get("validation") else None,
    )

    # --- Train ---
    logger.info("Starting model training...")
    try:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too
        logger.info("Training finished successfully.")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if args.push_to_hub:
            logger.info(f"Pushing model and tokenizer to Hugging Face Hub: {args.hub_model_id}")
            try:
                trainer.push_to_hub(commit_message="End of training")
                logger.info("Model pushed to Hub successfully.")
            except Exception as e:
                logger.error(f"Failed to push to Hub: {e}")

    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()
