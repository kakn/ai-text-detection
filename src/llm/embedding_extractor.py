# embedding_extractor.py

import gc
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import format_time, load_balanced_dataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

class TextDataset(Dataset):
    def __init__(self, texts, ids):
        self.texts = texts
        self.ids = ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.ids[idx]

class EmbeddingExtractor:
    """
    A class for loading a balanced dataset and extracting embeddings
    from a (fine-tuned or base) LLM (e.g., DistilBERT).
    """
    def __init__(self, subset_size=None, random_seed=42, max_length=512, model_name="distilbert-base-uncased"):
        self.subset_size = subset_size
        self.random_seed = random_seed
        self.max_length = max_length
        self.dataset = None  # Will hold a DatasetDict with train/val/test
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def load_and_prepare_data(self):
        print("Loading dataset...")
        raw_dataset = load_balanced_dataset(self.subset_size)  
        label_feature = ClassLabel(names=['human', 'ai'])

        def label_encode(example):
            example['labels'] = label_feature.str2int(example['source'])
            return example

        remove_columns = ['source']
        labeled_dataset = raw_dataset.map(label_encode, remove_columns=remove_columns)

        print("Tokenizing dataset...")
        processed_dataset = labeled_dataset.map(self.preprocess_function, batched=True, num_proc=8)
        self.dataset = processed_dataset.to_pandas()

        if "text" not in self.dataset.columns or "id" not in self.dataset.columns:
            raise ValueError("Dataset is missing required columns: 'text' or 'id'")

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length
        )
        model_inputs['labels'] = examples['labels']
        return model_inputs

    def merge_and_cleanup_embeddings(self, output_dir: str, dataset_name: str, use_fine_tuned: bool):
        prefix = f"{dataset_name}_{'finetuned' if use_fine_tuned else 'pretrained'}_"
        chunk_files = sorted((f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".npz")), 
                             key=lambda x: (x.endswith("_final.npz"), x))
        if not chunk_files:
            print(f"No chunk files found for {dataset_name}, skipping merge.")
            return

        if use_fine_tuned:
            final_dir = os.path.join(output_dir, "finetuned", "uncompressed", f"{dataset_name}_finetuned_embeddings")
        else:
            final_dir = os.path.join(output_dir, "pretrained", "uncompressed", f"{dataset_name}_pretrained_embeddings")
        os.makedirs(final_dir, exist_ok=True)

        # Calculate total size and hidden_dim
        with np.load(os.path.join(output_dir, chunk_files[0]), allow_pickle=True) as data:
            hidden_dim = data["hidden_states"].shape[2]
        total_samples = sum(np.load(os.path.join(output_dir, f), allow_pickle=True)["hidden_states"].shape[0] for f in chunk_files)

        # Create memory-mapped array and preallocate a list for IDs
        temp_memmap_path = os.path.join(output_dir, f"{prefix}temp_hs.dat")
        merged_hs_memmap = np.memmap(temp_memmap_path, dtype=np.float16, mode="w+", shape=(total_samples, 512, hidden_dim))
        merged_ids = [None] * total_samples

        # Function to merge a subset of chunks
        def merge_chunk_subset(subset, start_idx, merged_hs_memmap):
            local_idx = start_idx
            for chunk_file in subset:
                with np.load(os.path.join(output_dir, chunk_file), allow_pickle=True) as data:
                    chunk_hs, chunk_ids = data["hidden_states"], data["ids"]
                    end_idx = local_idx + chunk_hs.shape[0]
                    merged_hs_memmap[local_idx:end_idx] = chunk_hs
                    merged_ids[local_idx:end_idx] = list(chunk_ids)
                    local_idx = end_idx
            return local_idx - start_idx

        num_threads = min(8, len(chunk_files))
        chunk_subsets = np.array_split(chunk_files, num_threads)
        start_indices = np.cumsum([0] + [np.load(os.path.join(output_dir, f), allow_pickle=True)["hidden_states"].shape[0] for f in chunk_files[:-1]])

        with ThreadPoolExecutor(max_workers=num_threads) as executor, tqdm(total=total_samples, desc="Merging chunks") as pbar:
            futures = [executor.submit(merge_chunk_subset, subset, start_idx, merged_hs_memmap)
                    for subset, start_idx in zip(chunk_subsets, start_indices)]
            for future in as_completed(futures):
                pbar.update(future.result())

        # Save merged embeddings and IDs as uncompressed .npy files
        final_hidden_path = os.path.join(final_dir, "hidden_states.npy")
        final_ids_path = os.path.join(final_dir, "ids.npy")
        merged_hs_memmap.flush()
        hidden_states_array = np.array(merged_hs_memmap)  # Convert memmap to a regular array
        np.save(final_hidden_path, hidden_states_array)
        np.save(final_ids_path, np.array(merged_ids))

        # Clean up memory-mapped array and temporary file
        del merged_hs_memmap
        if os.path.exists(temp_memmap_path):
            os.remove(temp_memmap_path)

        # Delete the original chunk files
        print("Final merge successful. Deleting original chunk files...")
        for chunk_file in chunk_files:
            os.remove(os.path.join(output_dir, chunk_file))
        print(f"Final merged files saved in: {final_dir}")

    def extract_embeddings(self, use_fine_tuned: bool = True, model_path_finetuned: str = "data/distilbert_tired_model_output", 
                           model_path_pretrained: str = "distilbert-base-uncased", output_dir: str = "data/llm", 
                           batch_size: int = 512, num_workers: int = 8, chunk_size: int = 1000):
        """
        Extracts the final hidden state (DistilBERT) for the dataset 
        plus the evasive sets (control/basic/advanced).
        Everything is zero-padded to [batch_size, 512, hidden_dim].
        """
        start_time = time.time()

        if self.dataset is None:
            raise ValueError("No dataset found. Please call `load_and_prepare_data()` first.")

        # Decide which model to load
        model_path = model_path_finetuned if use_fine_tuned else model_path_pretrained
        print(f"Using model from: {model_path}")

        # Load model & tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Make sure output dirs exist
        os.makedirs(output_dir, exist_ok=True)

        def process_split(texts, ids, dataset_name):
            dataloader = DataLoader(
                TextDataset(texts, ids),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            all_hidden_states = []
            all_ids = []
            chunk_counter = 0

            for batch_texts, batch_ids in tqdm(dataloader, desc=f"Processing {dataset_name}", unit="batch"):
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    with autocast():
                        outputs = model(**inputs, output_hidden_states=True)

                    last_layer = outputs.hidden_states[-1].detach().cpu().float().numpy()

                # Convert to float16 to save space
                last_layer = last_layer.astype(np.float16)

                # Zero-pad to shape [b, 512, hidden_dim]
                b, seq_len, hidden_dim = last_layer.shape
                padded = np.zeros((b, 512, hidden_dim), dtype=np.float16)
                padded[:, :seq_len, :] = last_layer[:, :seq_len, :]

                all_hidden_states.append(padded)
                all_ids.extend(batch_ids)

                # Save chunk if needed
                total_so_far = sum(x.shape[0] for x in all_hidden_states)
                if total_so_far >= chunk_size:
                    file_name = f"{dataset_name}_{'finetuned' if use_fine_tuned else 'pretrained'}_chunk{chunk_counter:05d}.npz"
                    file_path = os.path.join(output_dir, file_name)
                    np.savez_compressed(
                        file_path,
                        hidden_states=np.concatenate(all_hidden_states),
                        ids=np.array(all_ids),
                    )
                    all_hidden_states, all_ids = [], []
                    chunk_counter += 1

                # Cleanup
                del outputs, last_layer, padded, inputs
                gc.collect()
                torch.cuda.empty_cache()

            # Save leftover
            if all_hidden_states:
                file_name = f"{dataset_name}_{'finetuned' if use_fine_tuned else 'pretrained'}_final.npz"
                file_path = os.path.join(output_dir, file_name)
                np.savez_compressed(
                    file_path,
                    hidden_states=np.concatenate(all_hidden_states),
                    ids=np.array(all_ids),
                )
                print(f"Final {dataset_name} embeddings saved -> {file_path}")

        texts = self.dataset["text"].tolist()
        ids = self.dataset["id"].tolist()
        dataset_name = "full_dataset"
        process_split(texts, ids, dataset_name)
        self.merge_and_cleanup_embeddings(output_dir, dataset_name, use_fine_tuned)

        # Always process the evasive sets
        evasive_sets = [
            ("data/evasive_texts/control.csv",  "control"),
            ("data/evasive_texts/basic.csv",    "basic"),
            ("data/evasive_texts/advanced.csv", "advanced"),
        ]
        for path, name in evasive_sets:
            df_e = pd.read_csv(path)
            texts = df_e["rewritten_text"].tolist()
            ids = df_e["id"].tolist()
            process_split(texts, ids, name)
            self.merge_and_cleanup_embeddings(output_dir, name, use_fine_tuned)

        print(f"✅ Embedding extraction completed in {format_time(time.time() - start_time)}")

def uncompress_npz_to_npy(npz_path, out_dir):
    """
    Uncompresses a single .npz file into multiple .npy files, one for each key.
    If a key has already been extracted, it is skipped.
    """
    os.makedirs(out_dir, exist_ok=True)

    with np.load(npz_path) as data:
        keys = data.files
        for key in tqdm(keys, desc=f"Decompressing {os.path.basename(npz_path)}", leave=False):
            out_file = os.path.join(out_dir, f"{key}.npy")

            if os.path.exists(out_file):  # Check if the file already exists
                print(f"  Skipping {key}, already extracted.")
                continue  # Skip already extracted files

            arr = data[key]  # Entire array is loaded into RAM here
            np.save(out_file, arr)

def uncompress_multiple_npz_to_npy(base_dir, out_dir):
    """
    Finds .npz files in `base_dir`, uncompresses each into a subdirectory of `out_dir`.
    Displays an outer tqdm for the list of files and an *inner* tqdm for the keys inside each file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Gather all .npz files in base_dir (non-recursive)
    all_npz_files = [f for f in os.listdir(base_dir) if f.endswith(".npz")]
    print(f"Found {len(all_npz_files)} .npz files in {base_dir}.")

    for npz_file in tqdm(all_npz_files, desc="Uncompressing NPZ files"):
        npz_path = os.path.join(base_dir, npz_file)
        file_stem = os.path.splitext(npz_file)[0]
        npz_out_dir = os.path.join(out_dir, file_stem)
        uncompress_npz_to_npy(npz_path, npz_out_dir)