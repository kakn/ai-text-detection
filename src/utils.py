# utils.py

import datetime
import os
import pickle
from typing import Tuple

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def compare_and_write_evasive_ids(evasive_ids_path="data/output/evasive_ids.txt", csv_dir="data/evasive_texts"):
    """
    1. Loads all IDs from evasive_ids.txt.
    2. For each CSV in [control.csv, basic.csv, advanced.csv], loads the 'id' column.
    3. Writes those actual IDs to a file 'actual_evasive_ids_<type>.txt'.
    4. Prints stats on missing/extra IDs.
    """
    # 1. Load reference IDs from text file
    with open(evasive_ids_path, "r") as f:
        reference_ids = [int(line.strip()) for line in f]
    reference_ids_set = set(reference_ids)
    
    # CSV files to check
    csv_files = {
        "control":   os.path.join(csv_dir, "control.csv"),
        "basic":     os.path.join(csv_dir, "basic.csv"),
        "advanced":  os.path.join(csv_dir, "advanced.csv"),
    }

    for name, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}. Skipping.")
            continue

        # 2. Load actual IDs from CSV
        df = pd.read_csv(csv_path)
        actual_ids = list(map(int, df["id"].tolist()))
        actual_ids_set = set(actual_ids)

        # 3. Write the actual IDs to a new text file
        out_file = os.path.join(csv_dir, f"actual_evasive_ids_{name}.txt")
        with open(out_file, "w") as f_out:
            for aid in actual_ids:
                f_out.write(str(aid) + "\n")

        # 4. Print stats
        missing_in_csv = reference_ids_set - actual_ids_set  # in txt, not in CSV
        extra_in_csv   = actual_ids_set - reference_ids_set  # in CSV, not in txt

        print(f"\n--- {name.upper()} CSV ---")
        print(f"  Total in evasive_ids.txt: {len(reference_ids)}")
        print(f"  Total in {name}.csv: {len(actual_ids)}")
        print(f"  Missing in {name}.csv (found in text file but not in CSV): {len(missing_in_csv)}")
        print(f"  Extra in {name}.csv (not in text file): {len(extra_in_csv)}")
        print(f"  Wrote actual IDs -> {out_file}")

def reorder_all_evasive_embeddings():
    base_dir = "data/llm/pretrained/uncompressed"

    # Map each evasive subdir to its actual ID file
    subdir_to_ids = {
        "advanced_pretrained_embeddings": "data/evasive_texts/actual_evasive_ids_advanced.txt",
        "basic_pretrained_embeddings":    "data/evasive_texts/actual_evasive_ids_basic.txt",
        "control_pretrained_embeddings":  "data/evasive_texts/actual_evasive_ids_control.txt",
    }

    for subdir, ids_file in subdir_to_ids.items():
        folder_path = os.path.join(base_dir, subdir)
        print(f"\nReordering embeddings in: {folder_path}")
        reorder_hidden_states(folder_path, ids_file, output_prefix="sorted")

def write_evasive_ids_file(subset_size=None, test_size=0.1, random_state=42, output_file="data/output/evasive_ids.txt"):
    """
    Loads the balanced dataset, splits into train/test sets,
    filters out up to 5000 AI-labeled samples from the test set,
    and writes those IDs to a text file.

    Args:
        subset_size (int or None): Optional limit for the dataset size.
        test_size (float): Fraction of data to put in the test set.
        random_state (int): Random seed for reproducible splits.
        output_file (str): Path to the file where we'll write the IDs.
    """
    # 1. Load the balanced dataset (the same one used in get_llama3_texts())
    dataset = load_balanced_dataset(subset_size=subset_size)

    # 2. Extract text, ID, and label fields
    texts = [entry['text'] for entry in dataset]
    ids = [entry['id'] for entry in dataset]
    labels = [1 if entry['source'] == 'ai' else 0 for entry in dataset]

    # 3. Split into train/test sets (same parameters as your original function)
    _, x_test, _, ids_test, _, labels_test = train_test_split(texts, ids, labels, test_size=test_size, random_state=random_state)

    # 4. Select only AI-labeled texts from the test set, up to 5000
    ids_test_ai = [id_ for id_, label in zip(ids_test, labels_test) if label == 1][:5000]

    # 5. Write these IDs to the specified file
    with open(output_file, "w") as f:
        for item_id in ids_test_ai:
            f.write(str(item_id) + "\n")

    print(f"✅ Wrote {len(ids_test_ai)} AI test IDs to '{output_file}'")

def load_ids_from_file(ids_path):
    """
    Loads IDs from a .pkl or .txt file.

    Args:
        ids_path (str): Path to the file containing the IDs.

    Returns:
        list of int: The list of IDs.
    """
    if ids_path.endswith(".pkl"):
        print(f"Loading IDs from pickle file: {ids_path}")
        with open(ids_path, "rb") as f:
            return pickle.load(f)
    
    elif ids_path.endswith(".txt"):
        print(f"Loading IDs from text file: {ids_path}")
        with open(ids_path, "r") as f:
            return [int(line.strip()) for line in f]

    else:
        raise ValueError(f"Unsupported ID file format: {ids_path} (must be .pkl or .txt)")

def reorder_hidden_states(uncompressed_dir: str, balanced_ids_path: str, output_prefix: str = "sorted"):
    """
    Reorders hidden_states.npy (and ids.npy) to match the order of IDs given 
    in balanced_ids.pkl (or .txt). Saves the sorted copies with new file names.
    Uses memory-mapping to handle large files efficiently.
    """
    # --- 1. Load uncompressed arrays using memory-mapping ---
    old_hs_path = os.path.join(uncompressed_dir, "hidden_states.npy")
    old_ids_path = os.path.join(uncompressed_dir, "ids.npy")

    if not (os.path.exists(old_hs_path) and os.path.exists(old_ids_path)):
        raise FileNotFoundError("hidden_states.npy or ids.npy not found in uncompressed_dir")

    print(f"Loading existing hidden_states and ids from:\n  {old_hs_path}\n  {old_ids_path}")

    # ✅ Load hidden states using memory-mapping (prevents memory overload)
    hidden_states = np.load(old_hs_path, mmap_mode="r")  # Shape: (num_samples, 512, hidden_dim)
    old_ids = np.load(old_ids_path)  # Shape: (num_samples,)

    print(f"✅ Loaded hidden_states of shape {hidden_states.shape} and ids of shape {old_ids.shape}")

    # --- 2. Load the target ID order from .pkl or .txt ---
    print(f"Loading target ID order from {balanced_ids_path}")
    target_ids = load_ids_from_file(balanced_ids_path)

    # Convert both ID lists to standard integers (avoids dtype mismatches)
    old_ids = old_ids.astype(int)  
    target_ids = list(map(int, target_ids))

    # --- 3. Create a lookup (dict) from ID -> index in the old array ---
    print("Creating ID -> index mapping for the old order...")
    id_to_old_idx = {id_val: idx for idx, id_val in enumerate(old_ids)}

    # Optional check: ensure every ID in target_ids is present in old_ids
    missing = [tid for tid in target_ids if tid not in id_to_old_idx]
    if missing:
        raise ValueError(f"Some IDs in the target list do not appear in the old IDs: {missing[:10]} (truncated)")

    # --- 4. Reorder the hidden states (without loading everything into RAM) ---
    print("Reordering hidden_states using memory-mapping...")
    sorted_indices = [id_to_old_idx[tid] for tid in tqdm(target_ids, desc="Mapping sorted indices", unit="ids")]

    # ✅ Create a new memory-mapped file to store sorted hidden states
    sorted_hs_path = os.path.join(uncompressed_dir, f"hidden_states_{output_prefix}.npy")
    sorted_ids_path = os.path.join(uncompressed_dir, f"ids_{output_prefix}.npy")

    hidden_states_sorted = np.memmap(sorted_hs_path, dtype=hidden_states.dtype, mode="w+", shape=hidden_states.shape)

    # Process and store in smaller chunks (prevents memory issues)
    chunk_size = 50_000  # Tune this based on your system
    for i in tqdm(range(0, len(sorted_indices), chunk_size), desc="Writing chunks", unit="chunks"):
        chunk_idx = sorted_indices[i : i + chunk_size]
        hidden_states_sorted[i : i + len(chunk_idx)] = hidden_states[chunk_idx]
    
    # Flush and delete the memmap to ensure proper writing**
    hidden_states_sorted.flush()
    del hidden_states_sorted  # Prevents corruption

    # ✅ Save the sorted IDs (since it's small, we can do it normally)
    np.save(sorted_ids_path, np.array(target_ids, dtype=np.int64))

    print(f"✅ Reordered hidden_states saved to: {sorted_hs_path}")
    print(f"✅ Reordered ids saved to:          {sorted_ids_path}")

def load_balanced_dataset(subset_size: int=None):
    """Loads the dataset, balancing the labels and shuffling the data"""
    dataset = datasets.load_dataset("artem9k/ai-text-detection-pile", cache_dir="./data", split="train")
    dataset = dataset.map(lambda x: {"label": 1 if x['source'] == 'ai' else 0})

    ai_texts = dataset.filter(lambda x: x['label'] == 1).shuffle(seed=42)
    non_ai_texts = dataset.filter(lambda x: x['label'] == 0).shuffle(seed=42)

    min_size = min(len(ai_texts), len(non_ai_texts))
    balanced_ai_texts = ai_texts.select(range(min_size))
    balanced_non_ai_texts = non_ai_texts.select(range(min_size))

    max_ai_text_length = max(len(text.split()) for text in balanced_ai_texts['text'])

    filtered_non_ai_texts = balanced_non_ai_texts.filter(lambda x: len(x['text'].split()) <= max_ai_text_length)
    num_to_deselect = len(balanced_non_ai_texts) - len(filtered_non_ai_texts)
    deselected_ai_texts = balanced_ai_texts.select(range(len(balanced_ai_texts) - num_to_deselect))

    balanced_dataset = datasets.concatenate_datasets([deselected_ai_texts, filtered_non_ai_texts]).shuffle(seed=42)
    
    if subset_size:
        return balanced_dataset.select(range(subset_size))
    return balanced_dataset

def get_data_statistics(X: pd.DataFrame, y: np.ndarray) -> str:
    """Calculates the average values of features for AI-generated and non-AI-generated texts."""
    ai_features = X[y == 1]
    human_features = X[y == 0]

    ai_means = ai_features.mean()
    human_means = human_features.mean()

    comparison_df = pd.DataFrame({'AI-Generated': ai_means, 'Human-Generated': human_means})
    comparison_str = comparison_df.to_string(float_format="%.2f")

    return comparison_str

def print_metrics(y_true, y_pred, verbose=True) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:")
        print(cm)

    return accuracy, precision, recall, f1, cm

def plot_feature_importance(importances, features, top_n=None, exclude_features=None, save_path="data/figures/feature_importance.png"):
    """
    Plots the top N most important features, showing both positive and negative impacts in a single plot,
    excluding specified feature types.

    Args:
        importances (pd.Series): The importance scores for the features.
        features (pd.Index): The names of the features.
        top_n (int): The number of top features to display. Defaults to None.
        exclude_features (list): List of substrings to exclude features by (e.g., ['tfidf', 'sbert'] to exclude these features).
        save_path (str): Path including filename where the plot will be saved.
    """
    if not top_n:
        top_n = len(features)

    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    
    if exclude_features:
        pattern = '|'.join(exclude_features)
        importance_df = importance_df[~importance_df['Feature'].str.contains(pattern)]
    
    # Separate positive and negative importances
    importance_df['Positive_Importance'] = importance_df['Importance'].abs()
    importance_df['Sign'] = importance_df['Importance'].apply(lambda x: 'AI Generated' if x > 0 else 'Human Generated')
    
    # Sort by absolute importance and get top N features
    top_features = importance_df.sort_values(by='Positive_Importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Positive_Importance', y='Feature', hue='Sign', data=top_features, dodge=False, palette={'AI Generated': '#1f77b4', 'Human Generated': '#d62728'})
    
    plt.title('Top Features')
    plt.xlabel('Absolute Importance')
    plt.ylabel('')
    plt.legend(title='Feature Source', loc='lower right')

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def get_example_texts(file_path: str='data/human_ai_generated_text/model_training_dataset.csv', print_texts: bool=True) -> Tuple[str, str]:
    """
    Function to read the dataset format and extracts one non-AI-generated (human) and one AI-generated text.
    Only one row is read from the dataset, due to its large size.

    Args:
        * file_path (str): The file path to the dataset.
        * print_texts (bool): A boolean determining whether or not to print the texts
    Returns: 
        A tuple containing two strings - the first is a non-AI-generated (human) text, the second is an AI-generated text.
    """
    dataset = pd.read_csv(file_path, nrows=1)

    human_text = dataset['human_text'].dropna().iloc[0]
    ai_text = dataset['ai_text'].dropna().iloc[0]

    if print_texts:
        print(f"Human Text: \n{human_text}\n")
        print(f"AI Text: \n{ai_text}")

    return human_text, ai_text

def format_time(seconds):
    """Formats time to display in an adaptive hours, minutes, and seconds format with proper singular and plural handling."""
    td = datetime.timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if td.days > 0:
        parts.append(f"{td.days} {'day' if td.days == 1 else 'days'}")
    if hours > 0:
        parts.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
    if minutes > 0:
        parts.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} {'second' if seconds == 1 else 'seconds'}")
    
    return " ".join(parts)

def calculate_roc_auc_per_prompt(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['text_label', 'prediction'])

    def calculate_roc_auc(group):
        if len(group) > 0:
            return roc_auc_score(group['text_label'], group['prediction'])
        else:
            return float('nan')
        
    roc_auc_results = data.groupby('unique_prompt_id').apply(calculate_roc_auc)
    roc_auc_df = roc_auc_results.reset_index()
    roc_auc_df.columns = ['unique_prompt_id', 'roc_auc']
    print(roc_auc_df)