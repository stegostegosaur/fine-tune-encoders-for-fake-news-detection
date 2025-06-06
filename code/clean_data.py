import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os

datapath = "data/raw/WELFake_Dataset.csv"


def preprocess(path):
    """Remove empty cells in cols ['title', 'text']"""
    df = pd.read_csv(path)
    df_clean = df.dropna() #558 titles and 39 texts
    df_clean = df_clean[df_clean['text'].str.len() > 1000]
    print("--- Cleaned DataFrame Info ---")
    print(df_clean.info())
    print("\n--- Cleaned DataFrame Description ---")
    print(df_clean.describe(include='all'))
    print(f"\nShape of cleaned DataFrame: {df_clean.shape}")

    return df_clean

def split_train_test(df, test_size=0.2, random_state=42):
    """Splits the DataFrame into training and testing sets and saves them."""

    target_col = 'label'

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    print(f"\nSplitting with stratification on column: '{target_col}'")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # Define output paths
    processed_dir = "data/processed"
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    # Create the directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Save the datasets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nTraining data saved to {train_path}")
    print(f"Test data saved to {test_path}")

    return train_df, test_df

if __name__ == "__main__":
    print(f"Starting preprocessing for dataset: {datapath}")
    cleaned_df = preprocess(datapath)

    print("\nStarting train-test split...")
    split_train_test(cleaned_df)
    print("\nScript finished.")
