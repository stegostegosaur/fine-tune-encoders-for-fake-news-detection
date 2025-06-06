# Fine-Tuning BERT and DistilBERT for Fake News Detection

This repository contains a comprehensive pipeline for fine-tuning and evaluating transformer models (BERT and DistilBERT) for the task of fake news detection. The project is structured to facilitate experimentation with different model configurations and layer-freezing strategies.

## Project Goal

The primary goal of this project is to compare the performance of a standard `bert-base-uncased` model with a lighter, distilled version, `distilbert-base-uncased`. We evaluate two fine-tuning strategies for each model:
1.  **Full Encoder Freeze**: Only the final classification layer is trained. This is a form of feature extraction.
2.  **Half Encoder Freeze**: The first half of the model's encoder layers are frozen, while the latter half is fine-tuned along with the classification layer.

This results in a total of four experiments to analyze the trade-offs between model size, training complexity, and performance on the fake news detection task.

## Project Structure

```
├── code/
│   ├── clean_data.py       # Script to preprocess and split the raw data
│   ├── train.py            # Main script for model training and fine-tuning
│   └── test.py             # Script for evaluating a trained model on the test set
│
├── data/
│   ├── raw/                # (Ignored by Git) Place your raw dataset here
│   └── processed/          # (Ignored by Git) Stores train.csv and test.csv after cleaning
│
├── notebooks/
│   └── training_notebook.ipynb # Notebook version of the training script for interactive experimentation
│
├── results/
│   └── ...                 # (Ignored by Git) Stores model checkpoints, logs, and outputs
│
├── .gitignore              # Specifies files and directories to be ignored by Git
└── README.md               # You are here!
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/stegostegosaur/fine-tune-bert-for-fake-news-detection.git
    cd fine-tune-bert-for-fake-news-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    Place your raw dataset(s) (e.g., `WELFake_Dataset.csv`) into the `data/raw/` directory. This directory is ignored by Git, so the data will not be committed.

## Usage

The workflow is managed by three main scripts in the `code/` directory.

### 1. Preprocess the Data

First, run the `clean_data.py` script. This will load the raw data, perform cleaning (e.g., dropping rows with missing values), and split it into training and testing sets, saving them to `data/processed/`.

```bash
python code/clean_data.py
```

### 2. Train a Model

Next, use the `train.py` script to fine-tune a model. The script is highly configurable via command-line arguments. Here is an example of running one of the four experiments (DistilBERT with half its layers frozen):

```bash
python code/train.py \
    --model_name_or_path distilbert-base-uncased \
    --model_type distilbert \
    --freeze_mode half \
    --train_file data/processed/train.csv \
    --output_dir ./results/distilbert_half_frozen \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --fp16 \
    --push_to_hub \
    --hub_model_id your-username/distilbert-fakern-half-frozen
```
*   **Remember to change `your-username`** to your actual Hugging Face username.
*   Run this script four times, changing the `--model_*`, `--freeze_mode`, `--output_dir`, and `--hub_model_id` arguments for each of your experiments.

### 3. Evaluate the Model

After a model has been trained, evaluate its performance on the unseen test set using `test.py`. Point it to the directory where the trained model was saved.

```bash
python code/test.py \
    --model_dir ./results/distilbert_half_frozen \
    --test_file data/processed/test.csv
```
This will print the final metrics to the console and save them in a `test_results.json` file inside the model directory.

## Models on Hugging Face Hub

The final fine-tuned models from these experiments will be available on the Hugging Face Hub.

*   **BERT (All Layers Frozen):** [Link to be added]
*   **BERT (Half Layers Frozen):** [Link to be added]
*   **DistilBERT (All Layers Frozen):** [Link to be added]
*   **DistilBERT (Half Layers Frozen):** [Link to be added]
