import csv
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import format_time, load_balanced_dataset

ALLOWED_POS = {
    'RB', 'RBR', 'RBS',  # Adverbs
    'IN',  # Prepositions
    'CC',  # Conjunctions
    'JJ', 'JJR', 'JJS',  # Adjectives
    'DT',  # Determiners
    'PRP', 'PRP$', 'WP', 'WP$',  # Pronouns
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'  # Verbs
}

class WordFrequencyManager:
    def __init__(self, cpu_core_count=16, threshold=2.0, subset_size=None):
        self.cpu_core_count = cpu_core_count
        self.threshold = threshold
        self.subset_size = subset_size
        self.train_data = None

    def split_data(self):
        dataset = load_balanced_dataset(self.subset_size).to_pandas()
        self.train_data, _ = train_test_split(dataset, test_size=0.1, random_state=42)

    @staticmethod
    def process_text(text: str) -> Counter:
        words = word_tokenize(text.lower())
        pos_tags = pos_tag(words)
        filtered_words = [word for word, pos in pos_tags if pos in ALLOWED_POS]
        return Counter(filtered_words)
    
    def calculate_word_frequencies(self, texts: List[str], label) -> Counter:
        word_counter = Counter()
        
        with ProcessPoolExecutor(max_workers=self.cpu_core_count) as executor:
            results = list(tqdm(executor.map(WordFrequencyManager.process_text, texts), total=len(texts), desc=f"Calculating {label} frequencies"))
        for result in results:
            word_counter.update(result)
        
        return word_counter

    def compute_word_ratios(self) -> List[Dict[str, float]]:
        ai_texts = self.train_data[self.train_data['source'] == 'ai']['text'].tolist()
        human_texts = self.train_data[self.train_data['source'] == 'human']['text'].tolist()
        
        ai_word_freq = self.calculate_word_frequencies(ai_texts, "AI")
        human_word_freq = self.calculate_word_frequencies(human_texts, "human")
        
        all_word_freq = ai_word_freq + human_word_freq
        total_word_count = sum(all_word_freq.values())
        mean_freq = total_word_count / len(all_word_freq)

        word_ratios = []

        for word in ai_word_freq.keys():
            ai_freq = ai_word_freq.get(word, 0)
            human_freq = human_word_freq.get(word, 0)
            total_freq = ai_freq + human_freq

            if mean_freq <= total_freq and human_freq > 0:
                ratio = ai_freq / human_freq
                if ratio >= self.threshold:
                    total_occurrences = ai_freq + human_freq
                    adjusted_ratio = math.log(ratio) * math.log(total_occurrences + 1)
                    word_ratios.append({
                        'word': word,
                        'ai_freq': ai_freq,
                        'human_freq': human_freq,
                        'ratio': ratio,
                        'total_occurrences': total_occurrences,
                        'weighted_ratio': adjusted_ratio
                    })

        return word_ratios

    def save_to_csv(self, word_ratios: List[Dict[str, float]]):
        subset_tag = f"{self.subset_size}" if self.subset_size else "full"
        output_path = f"data/saved_data/{subset_tag}/ai_indicative_words.csv"
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["word", "ai_freq", "human_freq", "ratio", "total_occurrences", "weighted_ratio"])
            sorted_word_ratios = sorted(word_ratios, key=lambda x: -x["weighted_ratio"])
            for item in sorted_word_ratios:
                if item["word"].isalpha():
                    writer.writerow([item["word"], item["ai_freq"], item["human_freq"], item["ratio"], item["total_occurrences"], item["weighted_ratio"]])

def calculate_word_frequencies(cpu_core_count=16, subset_size=None):
    start_time = time.time()
    manager = WordFrequencyManager(cpu_core_count=cpu_core_count, subset_size=subset_size)
    manager.split_data()
    word_ratios = manager.compute_word_ratios()
    manager.save_to_csv(word_ratios)
    print(f"Extracted word frequency ratios in {format_time(time.time() - start_time)}")

if __name__ == "__main__":
    calculate_word_frequencies()