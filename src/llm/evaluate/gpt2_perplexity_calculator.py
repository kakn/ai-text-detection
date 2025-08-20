import os
import time

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from src.utils import format_time, load_balanced_dataset

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class GPT2PerplexityCalculator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
        self.model.eval()
        self.max_length = 1024
        self.nlp = spacy.load("en_core_web_sm")
        self.csv_file_name = 'data/gpt2_perplexity_features.csv'

    def calculate_perplexity(self, text):
        """Implement perplexity calculation based on loaded model and tokenizer."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        tokens = tokens.to('cuda')
        with torch.no_grad():
            outputs = self.model(**tokens, labels=tokens.input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def get_perplexity_scores(self, texts):
        max_perplexities = np.zeros(len(texts))
        mean_perplexities = np.zeros(len(texts))

        for i, text in tqdm(enumerate(texts), total=len(texts), desc="Calculating perplexity"):
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            if not sentences:
                max_perplexities[i] = np.nan
                mean_perplexities[i] = np.nan
                continue

            sentence_perplexities = [self.calculate_perplexity(sentence) for sentence in sentences]

            max_perplexities[i] = np.nanmax(sentence_perplexities)
            mean_perplexities[i] = np.nanmean(sentence_perplexities)

        return np.column_stack((max_perplexities, mean_perplexities))

    def process_dataset(self):
        dataset = load_balanced_dataset()
        df = dataset.to_pandas()
        texts = df['text'].tolist()
        ids = df['id'].tolist()
        print(f'Number of texts: {len(texts)}')

        results = self.get_perplexity_scores(texts)
        results_df = pd.DataFrame({
            'id': ids,
            'max_perplexity': results[:, 0],
            'mean_perplexity': results[:, 1]
        })
        results_df.to_csv(self.csv_file_name, index=False)

def get_perplexities():
    start_time = time.time()
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Current CUDA device:", torch.cuda.current_device())

    calculator = GPT2PerplexityCalculator()

    calculator.process_dataset()

    print(f"Total runtime: {format_time(time.time() - start_time)}")

def calculate_evasive_perplexities():
    start_time = time.time()
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Current CUDA device:", torch.cuda.current_device())

    calculator = GPT2PerplexityCalculator()

    evasive_datasets = [
        'data/evasive_texts/control.csv', 
        'data/evasive_texts/basic.csv', 
        'data/evasive_texts/advanced.csv'
    ]
    
    for dataset in evasive_datasets:
        df = pd.read_csv(dataset)
        texts = df['rewritten_text'].tolist()
        ids = df['id'].tolist()
        print(f'Number of texts: {len(texts)}')

        results = calculator.get_perplexity_scores(texts)
        results_df = pd.DataFrame({
            'id': ids,
            'max_perplexity': results[:, 0],
            'mean_perplexity': results[:, 1]
        })

        dataset_name = os.path.basename(dataset).replace('.csv', '')
        output_dir = f'data/evasive_texts/feature/{dataset_name}'
        os.makedirs(output_dir, exist_ok=True)
        perplexity_file_name = os.path.join(output_dir, f'{dataset_name}_perplexity.csv')
        
        results_df.to_csv(perplexity_file_name, index=False)
        print(f'Perplexity scores saved to {perplexity_file_name}')

    print(f"Total runtime: {format_time(time.time() - start_time)}")

if __name__ == '__main__':
    get_perplexities()