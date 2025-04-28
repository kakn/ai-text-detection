# hybrid_classifier.py

import gc
import os
import pickle
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.patches import Patch
from sklearn.model_selection import ParameterGrid, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils import load_balanced_dataset, print_metrics

def _generate_expected_ids(subset_size=None):
    dataset = load_balanced_dataset(subset_size=subset_size)
    ids = np.array([ex['id'] for ex in dataset])
    return ids

class HybridClassifier:
    def __init__(self, use_fine_tuned: bool = False, pooling: str = "first_token", subset_size=None):
        """
        Args:
            use_fine_tuned (bool): Whether to load fine-tuned embeddings or pretrained.
            pooling (str): "first_token" (CLS alternative), "mean_pool" (average of all tokens),
                          or "max_pool" (maximum value across tokens for each dimension).
        """
        self.use_fine_tuned = use_fine_tuned
        self.pooling = pooling
        self.subset_size = subset_size

        self.model_save_path = 'data/saved_models/hybrid_torch'
        self._ensure_directory_exists(self.model_save_path)

        self.model_file = self._generate_model_file_name()
        print(f"Model will be saved to/loaded from: {self.model_file}")

        # Updated architecture variables for the simpler model
        self.hidden_dims = [512, 256, 128]   # New architecture: list of hidden dimensions
        self.dropout = 0.3                   # Dropout probability
        self.hidden_dim = 64  # Hidden layer dimension for the simpler model
        self.output_dim = 1    # Output layer dimension for binary classification
        self.batch_size = 64
        self.lr = 1e-6

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

        if use_fine_tuned:
            self.embedding_path = "data/llm/finetuned/uncompressed/full_dataset_finetuned_embeddings"
        else:
            self.embedding_path = "data/llm/pretrained/uncompressed/full_dataset_pretrained_embeddings"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _generate_model_file_name(self) -> str:
        """
        Generates a unique model file name based on the model's attributes.
        """
        fine_tuned_tag = "fine_tuned" if self.use_fine_tuned else "pretrained"
        subset_tag = f"subset_{self.subset_size}" if self.subset_size else "full"
        model_file_name = f"hybrid_model_{fine_tuned_tag}_{self.pooling}_{subset_tag}.pt"
        return os.path.join(self.model_save_path, model_file_name)

    def _ensure_directory_exists(self, path):
        os.makedirs(path, exist_ok=True)

    def load_embeddings(self, path: str, dataset_type: str = "main", expected_ids_override: np.ndarray = None) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding path not found: {path}")

        hidden_path = os.path.join(path, "hidden_states.npy")
        ids_path = os.path.join(path, "ids.npy")
        if not os.path.exists(hidden_path) or not os.path.exists(ids_path):
            raise FileNotFoundError(f"Expected hidden_states.npy and ids.npy in directory: {path}")

        print(f"Loading embeddings from directory: {path}")
        hidden_states = np.load(hidden_path, mmap_mode="r")
        # ids = np.load(ids_path, allow_pickle=True)

        # Attempt to load original IDs only if needed as fallback and file exists
        original_loaded_ids = None
        ids_npy_exists = os.path.exists(ids_path)
        if ids_npy_exists and expected_ids_override is None and dataset_type != "main":
             try:
                 original_loaded_ids = np.load(ids_path, allow_pickle=True)
             except Exception as e:
                 print(f"WARNING: Failed to load {ids_path} ({e}).")

        if dataset_type == "main":
            ids = _generate_expected_ids(subset_size=self.subset_size) # Regenerate for main dataset
        elif dataset_type == "evasive" and expected_ids_override is not None:
            ids = expected_ids_override # Use provided IDs for evasive
        elif original_loaded_ids is not None:
            ids = original_loaded_ids # Fallback ONLY if not main and no override
            print(f"WARNING: Using potentially problematic IDs loaded from {ids_path}")
        else:
            raise ValueError(f"Cannot determine IDs for {path}. ids.npy missing/unreadable and no valid override/generation strategy.")

        total_samples = hidden_states.shape[0]
        
        total_embedding_samples = hidden_states.shape[0] # Use a distinct name
        if len(ids) != total_embedding_samples:
            print(f"WARNING: ID count ({len(ids)}) vs embedding count ({total_embedding_samples}) mismatch for {path}. Truncating.")
            min_len = min(len(ids), total_embedding_samples)
            ids = ids[:min_len]
            # Reset total_samples based on alignment
            total_samples = min_len
        else:
             # If counts match, proceed with original total_samples logic respecting subset_size
             total_samples = total_embedding_samples
             if self.subset_size and self.subset_size < total_samples:
                 print(f"Applying subset_size {self.subset_size} to aligned data.")
                 total_samples = self.subset_size
                 ids = ids[:total_samples] # Also subset the substituted IDs

        if self.subset_size and self.subset_size < total_samples: 
            total_samples = self.subset_size

        chunk_size = 10000
        all_pooled = []
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = hidden_states[start_idx:end_idx]

            if self.pooling == "first_token":
                pooled_chunk = chunk[:, 0, :]
            elif self.pooling == "mean_pool":
                pooled_chunk = chunk.mean(axis=1)
            elif self.pooling == "max_pool":
                pooled_chunk = chunk.max(axis=1)
            else:
                raise ValueError("Pooling must be 'first_token', 'mean_pool', or 'max_pool'")

            all_pooled.append(pooled_chunk)

        X_emb_all = np.concatenate(all_pooled, axis=0)
        ids_all = ids

        print(f"Embeddings loaded with shape {X_emb_all.shape}")
        return X_emb_all

    def get_labels_from_ids(self, ids: np.ndarray) -> np.ndarray:
        """Given an array of IDs, fetch the corresponding labels."""
        dataset = load_balanced_dataset()
        dataset_df = dataset.to_pandas()
        dataset_df.rename(columns={"label": "ai_generated"}, inplace=True)
        dataset_df = dataset_df[["id", "text", "ai_generated"]]
        label_lookup = dataset_df.set_index("id")["ai_generated"].to_dict()
        labels = np.array([label_lookup[i] for i in ids])
        return labels

    def load_handcrafted_features(self) -> Tuple[np.ndarray, np.ndarray]:
        output_dir = "data/saved_data"
        subset_tag = "full"
        features_save_path = os.path.join(output_dir, subset_tag, "features.pkl")
        labels_save_path = os.path.join(output_dir, subset_tag, "labels.pkl")

        if not os.path.exists(features_save_path) or not os.path.exists(labels_save_path):
            raise FileNotFoundError(f"Missing features or labels in {output_dir}/{subset_tag}. Run feature extraction first.")

        print(f"Loading handcrafted features from {features_save_path}...")
        with open(features_save_path, 'rb') as f:
            X = pickle.load(f)

        print(f"Loading labels from {labels_save_path}...")
        with open(labels_save_path, 'rb') as f:
            y = pickle.load(f)

        if self.subset_size and self.subset_size < X.shape[0]:
            print(f"Using subset of size {self.subset_size} out of {X.shape[0]}")
            X = X[:self.subset_size]
            y = y[:self.subset_size]

        print(f"Handcrafted features loaded with shape {X.shape}")
        return X, y

    def concatenate_features(self, X: np.ndarray, X_emb: np.ndarray) -> np.ndarray:
        """Concatenates handcrafted features with embeddings."""
        if X.shape[0] != X_emb.shape[0]:
            raise ValueError(f"Shape mismatch: X {X.shape}, X_emb {X_emb.shape}")

        merged_X = np.concatenate((X, X_emb), axis=1)
        print(f"Features concatenated. New shape: {merged_X.shape}")
        return merged_X

    def check_label_match(self, y1: np.ndarray, y2: np.ndarray, split_name: str) -> None:
        """Checks if the labels match exactly. Prints debug info if not."""
        mismatches = np.where(y1 != y2)[0]
        if mismatches.size > 0:
            print(f"Warning: {mismatches.size} mismatches found in {split_name} labels!")
            # Print first 20 mismatch indices and their values
            n_to_print = 20
            print(f"Showing first {n_to_print} mismatches:")
            for i in mismatches[:n_to_print]:
                print(f"Index {i}: handcrafted label = {y1[i]}, embedding label = {y2[i]}")
            # Optionally, print statistics on mismatches
            unique_hc, counts_hc = np.unique(y1[mismatches], return_counts=True)
            unique_emb, counts_emb = np.unique(y2[mismatches], return_counts=True)
            print("Mismatch label distribution in handcrafted labels:")
            print(dict(zip(unique_hc, counts_hc)))
            print("Mismatch label distribution in embedding labels:")
            print(dict(zip(unique_emb, counts_emb)))
            raise ValueError(f"Warning: {mismatches.size} mismatches found in {split_name} labels!")
        else:
            print(f"Labels match perfectly for {split_name} split.")

    def load_data(self) -> None:
        X_hc, y_hc = self.load_handcrafted_features()
        X_emb = self.load_embeddings(self.embedding_path, dataset_type="main")

        if X_hc.shape[0] != X_emb.shape[0]:
            raise ValueError(f"Handcrafted rows ({X_hc.shape[0]}) != Embedding rows ({X_emb.shape[0]}). "
                            "They must match one-to-one.")

        X_full = np.concatenate([X_hc, X_emb], axis=1)
        y_full = y_hc
        
        X_train, X_temp, y_train, y_temp = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        print(f"Final shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    def load_model(self):
        if os.path.exists(self.model_file):
            print(f"Loading model from {self.model_file}...")
            if self.X_train is None:
                self.load_data()
            if self.model is None:
                self.build_simple_model()  # Use the simpler model by default

            # Load weights onto the correct device
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            self.model.to(self.device)
            return True
        return False

    def save_model(self):
        print(f"Saving model to {self.model_file}...")
        torch.save(self.model.state_dict(), self.model_file)  # Save only the state_dict for the PyTorch model

    def build_model(self) -> None:
        """
        Initializes the deeper feed-forward neural network with multiple hidden layers,
        batch normalization, ReLU activations, and dropout.
        """
        input_dim = self.X_train.shape[1]
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        print(f"Model built with architecture:\n{self.model}")

    def build_simple_model(self) -> None:
        """
        Initializes a simpler feed-forward neural network with a single hidden layer.
        """
        input_dim = self.X_train.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        ).to(self.device)
        print(f"Simple model built with architecture:\n{self.model}")

    def train(self, batch_size, lr, patience) -> int: # Change signature, add return type hint
        """
        Trains the neural network with early stopping and no fixed epoch count.

        Args:
            batch_size (int): Batch size.
            lr (float): Learning rate.
            patience (int): Number of epochs to wait for improvement before stopping early.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` or `build_simple_model()` first.")

        train_loader = DataLoader(
            TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(self.X_val, dtype=torch.float32), torch.tensor(self.y_val, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_loss, patience_counter = float('inf'), 0
        best_epoch = -1
        best_model_state = None
        epoch = 0

        while True:  # Runs indefinitely until early stopping
            epoch += 1
            self.model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            
            for X, y in progress_bar:
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X).squeeze(1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation Step
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.model(X).squeeze(1)
                    loss = criterion(logits, y)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = self.model.state_dict()
                print("Validation loss improved! Saving model...")
            else:
                patience_counter += 1
                print(f"No improvement. Early stopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered. Restoring best model.")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                else:
                    print("Warning: Early stopping triggered, but no 'best_model_state' was saved.")
                    if best_epoch == -1: best_epoch = epoch # Handle case where no improvement seen
                break # Keep break after restoring state
        
        return best_epoch

    def evaluate(self) -> None:
        """
        Evaluates the trained model on the test set and prints additional metrics.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` or `build_simple_model()` first.")

        test_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), 
                                     torch.tensor(self.y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
            
            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                logits = self.model(X_batch).squeeze(1)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                preds = (probs >= 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        print_metrics(all_labels, all_preds, verbose=True)
    
    def evaluate_evasive_texts(self) -> None:
        """
        Evaluates the trained model on evasive texts (control, basic, advanced).
        """
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` or `build_simple_model()` first.")

        # Load human texts from the test set
        human_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32))
        human_loader = DataLoader(human_dataset, batch_size=32, shuffle=False)

        # Extract human texts and labels
        human_X, human_y = [], []
        with torch.no_grad():
            for X_batch, y_batch in human_loader:
                human_X.extend(X_batch.cpu().numpy())
                human_y.extend(y_batch.cpu().numpy())

        human_X = np.array(human_X)
        human_y = np.array(human_y)

        evasive_types = ["control", "basic", "advanced"]

        for e_t in evasive_types:
            print(f"\nEvaluating on {e_t.capitalize()} evasive texts...")

            # Load embeddings and their IDs
            embedding_path = f"data/llm/pretrained/uncompressed/{e_t}_pretrained_embeddings"
            # X_emb_evasive, ids_evasive = self.load_embeddings(embedding_path)

            feature_path = f"data/evasive_texts/feature/{e_t}/full_features.pkl"
            # ids_hc_path = f"data/evasive_texts/feature/{e_t}/full_ids.pkl"
            labels_path = f"data/evasive_texts/feature/{e_t}/full_labels.pkl"
            csv_path = f"data/evasive_texts/{e_t}.csv"

            # Check essential feature/label/CSV files exist
            required_files = [feature_path, labels_path, csv_path] # Use CSV path
            if not all(os.path.exists(p) for p in required_files):
                if not os.path.exists(embedding_path):
                    print(f"WARNING: Skipping {e_t} - Embedding path AND required feature/label/csv files missing.")
                    continue
                missing = [f for f in required_files if not os.path.exists(f)]
                raise FileNotFoundError(f"Missing required files for {e_t}: {', '.join(missing)}")
            elif not os.path.exists(embedding_path):
                print(f"WARNING: Skipping {e_t} - Embedding path missing: {embedding_path}")
                continue

            with open(feature_path, 'rb') as f: X_evasive_hc = pickle.load(f)
            # with open(ids_hc_path, 'rb') as f: ids_hc = pickle.load(f)
            with open(labels_path, 'rb') as f: y_evasive = pickle.load(f)

            print(f"Loading trusted IDs from {csv_path}...")
            try:
                evasive_df = pd.read_csv(csv_path)
                if 'id' not in evasive_df.columns:
                    raise ValueError(f"'id' column not found in {csv_path}")
                ids_from_csv = evasive_df['id'].dropna().astype(str).values
                if evasive_df['id'].isnull().any():
                    print(f"WARNING: Dropped {evasive_df['id'].isnull().sum()} NaN IDs from {csv_path}")
            except Exception as e:
                print(f"ERROR: Failed to load or process IDs from {csv_path}: {e}")
                continue

            # Load embeddings, providing the CSV IDs as the override
            X_emb_evasive = self.load_embeddings( # Remove the ", _"
                embedding_path,
                dataset_type="evasive",
                expected_ids_override=ids_from_csv # Pass IDs from CSV
            )
            # ids_evasive = ids_from_csv # This variable isn't strictly needed anymore

            # Verify ID alignment instead of label check
            # if not np.array_equal(ids_evasive, ids_hc):
            #     mismatch_idx = np.where(ids_evasive != ids_hc)[0]
            #     raise ValueError(
            #         f"ID mismatch between embeddings and features for {e_t} evasive texts. "
            #         f"First mismatch at index {mismatch_idx[0]}: "
            #         f"Embedding ID={ids_evasive[mismatch_idx[0]]}, "
            #         f"Feature ID={ids_hc[mismatch_idx[0]]}"
            #     )

            # Verify ID alignment check is now redundant because we forced ids_evasive = ids_hc
            # Check shapes instead as basic sanity check passed from load_embeddings implicitly
            if X_evasive_hc.shape[0] != X_emb_evasive.shape[0]:
                raise ValueError(f"Shape mismatch for {e_t}: HC features ({X_evasive_hc.shape[0]}) vs Embeddings ({X_emb_evasive.shape[0]})")
            if X_evasive_hc.shape[0] != len(ids_from_csv): # Check against CSV IDs
                raise ValueError(f"Shape mismatch for {e_t}: HC features ({X_evasive_hc.shape[0]}) vs IDs from CSV ({len(ids_from_csv)})")
            if X_evasive_hc.shape[0] != len(y_evasive): # Check labels match features
                raise ValueError(f"Shape mismatch for {e_t}: HC features ({X_evasive_hc.shape[0]}) vs HC Labels ({len(y_evasive)})")

            # Merge: handcrafted + embeddings
            X_evasive_combined = np.concatenate((X_evasive_hc, X_emb_evasive), axis=1)

            # Truncate human texts to match length of evasive
            human_X_truncated = human_X[:len(X_evasive_combined)]
            human_y_truncated = human_y[:len(X_evasive_combined)]

            # Combine them
            combined_X = np.concatenate((human_X_truncated, X_evasive_combined), axis=0)
            combined_y = np.concatenate((human_y_truncated, y_evasive), axis=0)

            # Evaluate
            combined_dataset = TensorDataset(torch.tensor(combined_X, dtype=torch.float32), torch.tensor(combined_y, dtype=torch.float32))
            combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in combined_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    logits = self.model(X_batch).squeeze(1)
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.cpu().numpy())

            print(f"Evaluation for {e_t.capitalize()} evasive texts:")
            print_metrics(all_labels, all_preds, verbose=True)

    def run(self, force_retrain: bool = False, use_simple_model: bool = True) -> None:
        """Runs the full pipeline: data loading, model initialization, training, and evaluation."""
        print("Running full hybrid classifier pipeline...")
        if self.load_model() and not force_retrain:
            print("Model loaded successfully. Skipping training.")
        else:
            self.load_data()
            if use_simple_model:
                self.build_simple_model()
            else:
                self.build_model()
            self.train(batch_size=self.batch_size, lr=self.lr)
            self.save_model()
        self.evaluate()
        self.evaluate_evasive_texts()

    def evaluate_on_set(self, X_set: np.ndarray, y_set: np.ndarray, set_name: str = "Validation") -> float:
        """Evaluates the current model on a given dataset partition and returns accuracy."""
        if self.model is None:
            raise ValueError("Model is not built or loaded.")
        if X_set is None or y_set is None:
            raise ValueError(f"{set_name} data not available.")

        print(f"Evaluating on {set_name} set...")
        dataset = TensorDataset(torch.tensor(X_set, dtype=torch.float32),
                                torch.tensor(y_set, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # Use self.batch_size

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch).squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = np.mean(all_preds == all_labels)
        print(f"{set_name} Accuracy: {accuracy:.4f}")
        # print_metrics(all_labels, all_preds, verbose=False)
        return accuracy # Return accuracy for comparison

    def grid_search(self, param_grid: dict) -> Tuple[Optional[Dict], float, Optional[int]]: # MODIFIED: return type hint
        print("Starting grid search...")
        print("Loading data for grid search...")
        try:
            self.load_data() # Uses ID substitution and warns on mismatch
        except Exception as e:
             print(f"Failed to load data before grid search: {e}")
             return None, None

        if self.X_train is None or self.X_val is None:
             print("Train or Validation data is None after loading.")
             return None, None

        print(f"Data loaded: Train={self.X_train.shape}, Val={self.X_val.shape}")

        best_score = -1.0
        best_params = None
        best_epoch_for_best_params = None
        results = []

        grid = ParameterGrid(param_grid)
        num_combinations = len(grid)
        print(f"Grid search over {num_combinations} parameter combinations.")

        # Although we create a new instance for final training, good practice if modifying self
        original_dropout = self.dropout
        original_hidden_dim = self.hidden_dim
        original_lr = self.lr
        original_batch_size = self.batch_size

        for i, params in enumerate(grid):
            print(f"\n--- Combination {i+1}/{num_combinations} ---")
            print(f"Parameters: {params}")

            # Extract params (init vs train separation less critical if modifying self)
            # We just need to set params before build/train
            init_params = {k: v for k, v in params.items() if k in ['hidden_dim']}
            train_params = {k: v for k, v in params.items() if k in ['lr', 'batch_size']}

            # Set architecture params before building
            if 'hidden_dim' in init_params:
                self.hidden_dim = init_params['hidden_dim'] # Set hidden_dim for this iteration
            else:
                self.hidden_dim = original_hidden_dim # Reset to default

            if self.X_train is None: raise ValueError("X_train is None before building model in grid search")
            self.build_simple_model()

            current_batch_size = train_params.get('batch_size', original_batch_size)
            current_lr = train_params.get('lr', original_lr)
            # current_patience = train_params.get('patience', 10)

            # Train uses self.X_train, self.y_train, self.X_val, self.y_val
            # Define max epochs for grid search runs
            grid_search_patience = 20
            print(f"Training with lr={current_lr}, batch_size={current_batch_size}, hidden_dim={self.hidden_dim}, patience={grid_search_patience}...")
            best_epoch_run = self.train(batch_size=current_batch_size, lr=current_lr, patience=grid_search_patience)

            val_accuracy = self.evaluate_on_set(self.X_val, self.y_val, set_name="Validation")
            results.append({'params': params, 'val_accuracy': val_accuracy})

            if val_accuracy > best_score:
                print(f"✨ New best score: {val_accuracy:.4f} (previous: {best_score:.4f})")
                best_score = val_accuracy
                best_params = params
                best_epoch_for_best_params = best_epoch_run

            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

        self.dropout = original_dropout # Restore original dropout
        self.hidden_dim = original_hidden_dim # Restore original hidden_dim
        self.lr = original_lr
        self.batch_size = original_batch_size

        print("\n--- Grid Search Complete ---")
        print(f"Best Validation Accuracy: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Epoch for Best Params: {best_epoch_for_best_params}")

        return best_params, best_score, best_epoch_for_best_params

    def train_fixed_epochs(self, batch_size: int, lr: float, num_epochs: int):
        """Minimal version: Trains for fixed epochs, no validation."""
        if self.model is None: raise ValueError("Model not built.")
        if self.X_train is None or self.y_train is None: raise ValueError("Training data missing.")
        if num_epochs <= 0: 
            print(f"Warning: num_epochs ({num_epochs}) <= 0, skipping."); return

        print(f"Starting fixed training: {num_epochs} epochs, lr={lr}, bs={batch_size}")
        train_loader = DataLoader(
            TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Fixed", leave=False)
            for X, y in progress_bar:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X).squeeze(1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs} Fixed - Train Loss: {avg_train_loss:.4f}")
        print("Fixed epoch training finished.")

    def visualize_feature_importance(self, max_display=20):
        """
        Main method to generate both regular and aggregated SHAP visualizations.
        
        Args:
            max_display (int): Maximum number of features to display
        """
        # Load model and data first (this is the slow part)
        if self.model is None:
            print("Loading model...")
            if not self.load_model():
                raise ValueError("Model not loaded or trained. Run train() or load_model() first.")
        
        # Load data if not already loaded
        if self.X_train is None:
            print("Loading data...")
            self.load_data()
        
        print("Preparing for SHAP analysis...")
        
        # Convert PyTorch model to sklearn-like model for SHAP
        class HybridModelWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                
            def predict(self, X):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    return self.model(X_tensor).squeeze(1).cpu().numpy()
        
        # Create the wrapper
        model_wrapper = HybridModelWrapper(self.model, self.device)
        
        # Get feature names - combine handcrafted feature names with embedding features
        features_path = 'data/saved_data/full/features.pkl'
        with open(features_path, 'rb') as f:
            feature_df = pickle.load(f)
        
        # Get handcrafted feature names
        handcrafted_features = feature_df.columns.tolist()
        n_handcrafted = len(handcrafted_features)
        
        # Create generic names for embedding dimensions (limited to what we need)
        n_embeddings = self.X_train.shape[1] - n_handcrafted
        embedding_features = [f"emb_{i}" for i in range(n_embeddings)]
        
        # Combine both feature sets (will be properly limited in the plotting functions)
        all_features = handcrafted_features + embedding_features
        
        # Select a subset of the training data for SHAP calculation (for performance)
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.X_train), min(1000, len(self.X_train)), replace=False)
        X_sample = self.X_train[sample_indices]
        
        # Initialize the SHAP explainer
        print("Creating SHAP explainer with memory optimization...")
        background_summary = shap.kmeans(X_sample, 50)  # Summarize with 50 representative samples
        explainer = shap.KernelExplainer(model_wrapper.predict, background_summary)
        
        # Calculate SHAP values (this is computation-intensive)
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Generate individual feature importance plot (simpler, like FeatureClassifierModel)
        self._generate_individual_importance_plot(shap_values, X_sample, all_features, max_display)
        
        # Generate the aggregated plot comparing embeddings vs handcrafted
        self._generate_violin_comparison_plot(shap_values, X_sample, n_handcrafted)
        
        print("Both SHAP visualizations completed successfully")

    def _generate_individual_importance_plot(self, shap_values, X_sample, feature_names, max_display=20):
        """
        Creates a standard SHAP summary plot showing individual feature importance.
        
        Args:
            shap_values: Pre-calculated SHAP values
            X_sample: Sample data for visualization
            feature_names: List of feature names
            max_display: Maximum number of features to display
        """
        print("Generating individual feature importance plot...")
        
        # Make sure max_display is within range of the actual feature count
        # This prevents index errors when SHAP tries to show more features
        feature_count = min(len(feature_names), X_sample.shape[1])
        max_display = min(max_display, feature_count)
        
        # Create plot with only the needed features
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample,
            feature_names=feature_names[:feature_count],  # Only use valid features
            max_display=max_display,  # Limit display count
            show=False
        )
        
        plt.title('Feature Importance')
        
        # Save the plot
        save_path = 'data/figures/hybrid_shap_individual.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual feature importance plot saved to {save_path}")

    def _generate_violin_comparison_plot(self, shap_values, X_sample, n_handcrafted):
        """
        Creates a violin plot comparing the impact of embeddings vs handcrafted features.
        
        Args:
            shap_values: Pre-calculated SHAP values
            X_sample: Sample data for visualization
            n_handcrafted: Number of handcrafted features
        """
        print("Generating embeddings vs handcrafted features comparison plot...")
        
        # Separate SHAP values
        shap_handcrafted = shap_values[:, :n_handcrafted]
        shap_embeddings = shap_values[:, n_handcrafted:]
        
        # Get average SHAP values per sample for each group
        shap_handcrafted_avg = np.mean(shap_handcrafted, axis=1)
        shap_embeddings_avg = np.mean(shap_embeddings, axis=1)
        
        # Create DataFrame for seaborn
        plot_data = []
        for value in shap_embeddings_avg:
            plot_data.append({'Group': 'Embeddings', 'SHAP Value': value})
        for value in shap_handcrafted_avg:
            plot_data.append({'Group': 'Handcrafted', 'SHAP Value': value})
        
        df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        
        # Add color background
        ax.axhspan(0, df['SHAP Value'].max() * 1.1, alpha=0.1, color='red')
        ax.axhspan(df['SHAP Value'].min() * 1.1, 0, alpha=0.1, color='blue')
        
        # Create violin plot
        sns.violinplot(
            x='Group', 
            y='SHAP Value', 
            data=df, 
            ax=ax,
            palette=["#ff9999", "#9999ff"], 
            cut=0, 
            inner='quartile'
        )
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add mean values as text
        emb_mean = np.mean(shap_embeddings_avg)
        hc_mean = np.mean(shap_handcrafted_avg)
        
        x_pos = {'Embeddings': 0, 'Handcrafted': 1}
        for group, mean_val in zip(['Embeddings', 'Handcrafted'], [emb_mean, hc_mean]):
            x = x_pos[group]
            ax.annotate(
                f'{mean_val:.3f}', 
                xy=(x, mean_val),
                xytext=(x, mean_val + (0.02 if mean_val > 0 else -0.02)),
                ha='center',
                fontweight='bold'
            )
        
        # Set labels and title
        ax.set_ylabel('SHAP Value', fontsize=14)
        ax.set_title('Feature Group Impact on Model Predictions', fontsize=16)
        
        # Add legend for color meaning
        legend_elements = [
            Patch(facecolor='red', alpha=0.1, label='Predicts AI'),
            Patch(facecolor='blue', alpha=0.1, label='Predicts Human')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save plot
        save_path = 'data/figures/hybrid_shap_aggregated.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {save_path}")

def train_hybrid_model(subset_size = None, use_simple_model: bool = True, use_fine_tuned: bool = False, pooling: str = "first_token") -> None:
    """
    Initializes HybridClassifier, performs grid search, and trains/evaluates the best model.
    NOTE: use_simple_model argument is now ignored because grid search uses build_model.
    """
    print("Initializing hybrid classifier for Grid Search...")

    param_grid = {
        'lr': [5e-7, 7.5e-7, 1e-6, 3e-6],
        'hidden_dim': [32, 64, 128],
        'batch_size': [64, 128]
    }

    # Pooling is passed from the function argument (defaulting to first token)
    hybrid_model_searcher = HybridClassifier(
        subset_size=subset_size,
        use_fine_tuned=use_fine_tuned,
        pooling=pooling
    )

    best_params, best_score, optimal_epochs = hybrid_model_searcher.grid_search(param_grid)

    if best_params and optimal_epochs is not None and optimal_epochs > 0:
        print(f"\n--- Retraining final model on COMBINED Train+Val data for {optimal_epochs} epochs ---")

        # Extract best params (use defaults matching grid)
        final_hidden_dim = best_params.get('hidden_dim', 128)
        final_lr = best_params.get('lr', 1e-4)
        final_batch_size = best_params.get('batch_size', 64)
        print(f"Using best params: hidden_dim={final_hidden_dim}, lr={final_lr}, batch_size={final_batch_size}")

        # Create final classifier instance
        final_classifier = HybridClassifier(
            subset_size=subset_size, use_fine_tuned=use_fine_tuned, pooling=pooling
        )
        final_classifier.hidden_dim = final_hidden_dim

        # Load data
        print("Loading data for final model...")
        final_classifier.load_data()
        if final_classifier.X_train is None or final_classifier.X_val is None \
        or final_classifier.y_train is None or final_classifier.y_val is None:
            raise RuntimeError("Failed to load train/val data for final training.")

        # Combine Train and Validation sets
        print("Combining Train and Validation sets for final training...")
        X_train_final = np.concatenate((final_classifier.X_train, final_classifier.X_val), axis=0)
        y_train_final = np.concatenate((final_classifier.y_train, final_classifier.y_val), axis=0)
        print(f"Final training set shape: {X_train_final.shape}")

        # Assign combined data
        final_classifier.X_train = X_train_final
        final_classifier.y_train = y_train_final
        final_classifier.X_val = None # Clear val data
        final_classifier.y_val = None

        # Build model
        print("Building final model (using build_simple_model)...")
        final_classifier.build_simple_model()

        print(f"Training final model for fixed {optimal_epochs} epochs...")
        final_classifier.train_fixed_epochs(
            batch_size=final_batch_size,
            lr=final_lr,
            num_epochs=optimal_epochs # Pass the captured epoch number
        )

        # Evaluate and Save final model
        print("\n--- Evaluating final model on Test Set ---")
        final_classifier.evaluate()
        print("\n--- Evaluating final model on Evasive Texts ---")
        final_classifier.evaluate_evasive_texts()
        final_classifier.save_model()
        print(f"Final model trained on combined data saved to: {final_classifier.model_file}")

    elif best_params and (optimal_epochs is None or optimal_epochs <= 0): # Add check for bad epoch count
        print("Grid search found best parameters, but optimal epoch count was invalid (< 1).")
        print("No final model trained.")
    else:
        print("Grid search did not complete successfully or find best parameters. No final model trained.")

def debug_none_ids_thorough():
    """Identifies the exact source of None IDs in embedding files."""
    print("🔍 Checking original dataset...")
    dataset = load_balanced_dataset()
    original_ids = [ex['id'] for ex in dataset]
    num_none_original = sum(1 for _id in original_ids if _id is None)
    print(f"Original dataset has {num_none_original} None IDs\n")

    print("🔍 Checking main dataset embeddings...")
    main_emb_dir = "data/llm/pretrained/uncompressed/full_dataset_pretrained_embeddings"
    main_ids = np.load(os.path.join(main_emb_dir, "ids.npy"), allow_pickle=True)
    main_none_mask = main_ids == None
    print(f"Main embeddings contain {np.sum(main_none_mask)} None IDs")

    if np.any(main_none_mask):
        print("\nMain embeddings have None IDs! Investigating origin...")
        # Find first 3 problematic indices
        bad_indices = np.where(main_none_mask)[0][:3]
        for idx in bad_indices:
            print(f"\nIndex {idx}:")
            # Reconstruct original dataset position
            original_idx = idx  # Only valid if subset_size=None and no shuffling
            print(f"Original dataset ID: {original_ids[original_idx]}")
            print(f"Original text snippet: {dataset[original_idx]['text'][:50]}...")

    # 3. Check evasive datasets
    print("\n🔍 Checking evasive CSVs...")
    for name in ["control", "basic", "advanced"]:
        csv_path = f"data/evasive_texts/{name}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            num_none = df['id'].isnull().sum()
            print(f"{name} CSV: {num_none} None IDs")
            if num_none > 0:
                print(f"First bad row:\n{df[df['id'].isnull()].iloc[0]}")
        else:
            print(f"{name} CSV not found")

if __name__ == "__main__":
    train_hybrid_model(
        subset_size=None,
        use_simple_model=True,
        use_fine_tuned=False,
        pooling="first_token"
    )