import concurrent.futures
import os

import language_tool_python
import pandas as pd
from tqdm import tqdm

from src.utils import load_balanced_dataset

class ErrorFeatureExtractor:
    def __init__(self, csv_file_name: str):
        self.csv_file_name = csv_file_name
        self.max_threads = min(16, os.cpu_count())
        self.language_tool = language_tool_python.LanguageTool('en-US', 
                                   config={'maxCheckThreads': self.max_threads})

    def load_data(self, subset_size=None):
        dataset = load_balanced_dataset(subset_size)
        df = dataset.to_pandas()
        ids = df['id'].tolist()
        texts = df['text'].tolist()
        return ids, texts

    def check_text(self, text):
        matches = self.language_tool.check(text)
        return len(matches)

    def process_data_parallel(self, texts):
        results = [None] * len(texts)  # Preallocate a list for results
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self.check_text, text): i for i, text in enumerate(texts)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts), desc="Extracting error counts in parallel"):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def save_results(self, ids, error_counts, output_file):
        results_df = pd.DataFrame({
            'id': ids,
            'error_count': error_counts
        })
        print(error_counts)
        results_df.to_csv(output_file, index=False)

def extract_error_features_old():
    extractor = ErrorFeatureExtractor('data/error_features.csv')
    print(f"CPU core count: {extractor.max_threads}")
    ids, texts = extractor.load_data()
    error_counts = extractor.process_data_parallel(texts)
    extractor.save_results(ids, error_counts)

def extract_error_features():
    datasets = [
        'data/evasive_texts/control.csv', 
        'data/evasive_texts/basic.csv', 
        'data/evasive_texts/advanced.csv'
    ]

    for dataset in datasets:
        dataset_name = os.path.basename(dataset).replace('.csv', '')
        output_dir = f'data/evasive_texts/feature/{dataset_name}'
        os.makedirs(output_dir, exist_ok=True)
        csv_file_name = os.path.join(output_dir, f'{dataset_name}_error_features.csv')

        extractor = ErrorFeatureExtractor(csv_file_name)
        print(f"Processing {dataset}")
        print(f"CPU core count: {extractor.max_threads}")

        df = pd.read_csv(dataset)
        ids = df['id'].tolist()
        texts = df['rewritten_text'].tolist()

        error_counts = extractor.process_data_parallel(texts)
        extractor.save_results(ids, error_counts, csv_file_name)
        print(f"Error features saved to {csv_file_name}")

if __name__ == "__main__":
    extract_error_features()