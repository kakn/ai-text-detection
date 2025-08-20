# feature_extraction.py

import math
import multiprocessing
import os
import pickle
import re
import time
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
import textstat
from joblib import Parallel, delayed
from nltk import FreqDist, ngrams, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from pathos.multiprocessing import ProcessPool
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spacy.matcher import Matcher
from textblob import TextBlob
from tqdm import tqdm

from src.utils import (
    format_time,
    get_data_statistics,
    load_balanced_dataset,
    plot_feature_importance,
    print_metrics,
)

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

class FeatureExtractor:
    """
    Collection of machine learning methods to collect features for an AI text classifier.

    Args:
        dataset_path (str): The path to the dataset to be trained and tested on
        random_seed (int): The random seed to use in data preprocessing 
    """
    def __init__(self):
        self.cpu_core_count = os.cpu_count()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.discourse_markers = self.load_discourse_markers()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.matcher = Matcher(self.nlp.vocab)
        passive_rules = [
            [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}],
            [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBZ'}],
            [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'RB'}, {'TAG': 'VBN'}],
        ]
        self.matcher.add('Passive', passive_rules)
    
    def load_discourse_markers(self):
        """Loads discourse markers from a file into a list."""
        markers = []
        with open('data/features_from_data/discourse_markers.txt', 'r') as file:
            for line in file:
                markers.append(line.strip().lower())
        return markers
    
    def get_test_data_subset(self, subset_size=None):
        """Fetches a subset of the data or the entire dataset if no size is specified."""
        dataset = load_balanced_dataset()
        df = dataset.to_pandas()
        
        if subset_size:
            df = df.iloc[:subset_size].copy()
        
        df['ai_generated'] = df['source'].map({'ai': 1, 'human': 0})
        return df[['id', 'text', 'ai_generated']]
    
    def get_data_subset(self, dataset_path: str, subset_size=None):
        dataset = pd.read_csv(dataset_path)
        if subset_size:
            dataset = dataset.iloc[:subset_size].copy()
        dataset['ai_generated'] = 1  # Always set to 1 for this special set
        dataset.rename(columns={'rewritten_text': 'text'}, inplace=True)
        return dataset[['id', 'text', 'ai_generated']]

    def load_or_process_data(self, dataset_path: str, feature_files=None, output_dir=None, subset_size=None) -> Tuple[pd.DataFrame, np.ndarray]:
        if not feature_files:
            feature_files = {
            'llm_pred': 'data/features_from_data/llama_feedback_results_dnd.csv',
            'error_count': 'data/features_from_data/error_features.csv',
            'max_perplexity': 'data/features_from_data/feature/gpt2_perplexity.csv',
            'mean_perplexity': 'data/features_from_data/feature/gpt2_perplexity.csv'
            }
        if not output_dir:
            subset_tag = f"{subset_size}" if subset_size else "full"
            output_dir = f"data/saved_data/{subset_tag}"

        features_save_path = os.path.join(output_dir, "features.pkl")
        labels_save_path = os.path.join(output_dir, "labels.pkl")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(features_save_path) and os.path.exists(labels_save_path):
            print("Loading existing feature data...")
            with open(features_save_path, 'rb') as f:
                extracted_features = pickle.load(f)
            with open(labels_save_path, 'rb') as f:
                y = pickle.load(f)
        else:
            data_subset = self.get_data_subset(dataset_path, subset_size)
            extracted_features = self.extract_features_from_dataset(data_subset, feature_files)
            y = data_subset['ai_generated'].values

            with open(features_save_path, 'wb') as f:
                pickle.dump(extracted_features, f)
            with open(labels_save_path, 'wb') as f:
                pickle.dump(y, f)

        return extracted_features, y

    def get_features_from_file(self, file_name, feature_name, ids):
        df = pd.read_csv(file_name)
        
        if 'id' not in df.columns or feature_name not in df.columns:
            raise ValueError(f"File {file_name} must contain 'id' and '{feature_name}' columns")
        
        file_ids = df['id'].tolist()
        missing_ids = [provided_id for provided_id in ids if provided_id not in file_ids]
        if missing_ids:
            raise ValueError(f"There are {len(missing_ids)} provided IDs missing in the file.")

        filtered_df = df[df['id'].isin(ids)]
        reindexed_df = filtered_df.set_index('id').reindex(ids).reset_index()
        features_df = pd.DataFrame(reindexed_df[feature_name], columns=[feature_name])
        return features_df
        
    def extract_features_from_dataset(self, data_subset: pd.DataFrame, feature_files: Dict[str, str]) -> pd.DataFrame:
        texts = data_subset['text'].tolist()
        ids = data_subset['id'].tolist()
        feature_dataset = []
        start_time = time.time()
        print("Extracting features...")
        pool = ProcessPool(nodes=self.cpu_core_count)
        futures = pool.map(self.extract_features_from_text, texts)
        for result in futures:
            feature_dataset.append(result)
        pool.close()
        pool.join()
        print(f"Extracted features in {format_time(time.time() - start_time)}")
        features_df = pd.DataFrame(feature_dataset)

        for feature_name, file_path in feature_files.items():
            additional_feature_df = self.get_features_from_file(file_path, feature_name, ids)
            if feature_name == 'llm_pred':
                additional_feature_df.fillna(0, inplace=True)
            features_df = pd.concat([features_df, additional_feature_df], axis=1)

        tfidf_features = self.extract_tfidf_features(texts)
        sbert_features = self.extract_sentence_bert_features(texts)
        sbert_distances = self.calculate_average_sentence_bert_distance(sbert_features)
        tfidf_features_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        sbert_features_df = pd.DataFrame(sbert_features, columns=[f'sbert_{i}' for i in range(sbert_features.shape[1])])
        sbert_distances_df = pd.DataFrame(sbert_distances, columns=['sentence_bert_distance'])
        concatenated_df = pd.concat([features_df, tfidf_features_df, sbert_features_df, sbert_distances_df], axis=1)

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_data = imputer.fit_transform(concatenated_df)
        imputed_final_df = pd.DataFrame(imputed_data, columns=concatenated_df.columns)
        return imputed_final_df
    
    def extract_features_from_text(self, text: str) -> Dict[str, float]:
        features = {}

        # Semantic features
        features['polarity'], features['subjectivity'] = self.get_sentiment_features(text)

        # List lookup features
        features['stop_word_count'] = self.count_stop_words(text)
        features['special_character_count'] = self.count_special_characters(text)
        features['discourse_marker_count'] = self.count_discourse_markers(text)

        # Document features
        features['words_per_paragraph_mean'], features['words_per_paragraph_std'] = self.get_words_per_paragraph(text)
        features['sentences_per_paragraph_mean'], features['sentences_per_paragraph_std'] = self.get_sentences_per_paragraph(text)
        features['words_per_sentence_mean'], features['words_per_sentence_std'] = self.get_words_per_sentence(text)
        features['unique_words_per_sentence_mean'], features['unique_words_per_sentence_std'] = self.get_unique_words_per_sentence(text)
        features['word_count'] = self.get_word_count(text)
        features['unique_word_count'] = self.get_unique_word_count(text)
        features['unique_words_relative'] = self.get_unique_words_relative(text)
        features['paragraph_count'] = self.get_paragraph_count(text)
        features['sentence_count'] = self.get_sentence_count(text)
        features['punctuation_count'] = self.get_punctuation_count(text)
        features['quotation_count'] = self.get_quotation_count(text)
        features['character_count'] = self.get_character_count(text)
        features['uppercase_words_relative'] = self.get_uppercase_words_relative(text)
        features['personal_pronoun_count'] = self.get_personal_pronoun_count(text)
        features['personal_pronoun_relative'] = self.get_personal_pronoun_relative(text)
        features['pos_per_sentence_mean'] = self.get_pos_per_sentence_mean(text)

        # Error-based features
        features['multi_blank_count'] = self.count_multiple_blanks(text)
        
        # Readability features
        features['flesch_reading_ease'], features['flesch_kincaid_grade'] = self.get_readability_scores(text)

        # New features
        features['dependent_clauses_count_avg'] = self.get_average_num_dependent_clauses(text)
        features['passive_voice_count'] = self.count_passive_voice(text)
        features['unigram_entropy'], features['bigram_entropy'], features['trigram_entropy']  = self.get_ngram_entropy(text)
        features['burstiness'] = self.calculate_burstiness(text)
        features['syntactic_tree_depth_mean'] = self.calculate_mean_syntactic_tree_depth(text)
        features['list_item_count'] = self.count_list_items(text)
        return features
    
    def get_sentiment_features(self, text: str) -> Tuple[float, float]:
        """Extracts sentiment features (polarity and subjectivity) from the given text."""
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def count_stop_words(self, text: str) -> int:
        """Counts stop words in the given text."""
        words = word_tokenize(text)
        stop_word_count = sum(word.lower() in self.stop_words for word in words)
        return stop_word_count

    def count_special_characters(self, text: str) -> int:
        """Counts special characters (non-letters) in the given text."""
        special_char_count = len(re.findall(r'[\W_]', text)) - text.count(' ')
        return special_char_count
    
    def count_discourse_markers(self, text: str) -> int:
        """Counts discourse markers (like 'furthermore', 'nonetheless', etc.) in the given text."""
        count = 0
        text = text.lower()
        for marker in self.discourse_markers:
            pattern = r'\b' + re.escape(marker) + r'\b'
            count += len(re.findall(pattern, text))
        return count
    
    def get_words_per_paragraph(self, text):
        paragraphs = [p for p in text.split('\n') if p]
        words_counts = [len(word_tokenize(p)) for p in paragraphs]
        mean = np.mean(words_counts)
        std = np.std(words_counts)
        return mean, std

    def get_sentences_per_paragraph(self, text):
        paragraphs = [p for p in text.split('\n') if p]
        sentences_counts = [len(sent_tokenize(p)) for p in paragraphs]
        mean = np.mean(sentences_counts)
        std = np.std(sentences_counts)
        return mean, std

    def get_words_per_sentence(self, text):
        sentences = sent_tokenize(text)
        words_counts = [len(word_tokenize(sent)) for sent in sentences]
        mean = np.mean(words_counts)
        std = np.std(words_counts)
        return mean, std

    def get_unique_words_per_sentence(self, text):
        sentences = sent_tokenize(text)
        unique_counts = [len(set(word_tokenize(sent))) for sent in sentences]
        mean = np.mean(unique_counts)
        std = np.std(unique_counts)
        return mean, std
    
    def get_word_count(self, text):
        return len(word_tokenize(text))

    def get_unique_word_count(self, text):
        return len(set(word_tokenize(text)))

    def get_unique_words_relative(self, text):
        words = word_tokenize(text)
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    def get_paragraph_count(self, text):
        return len([p for p in text.split('\n') if p])

    def get_sentence_count(self, text):
        return len(sent_tokenize(text))

    def get_punctuation_count(self, text):
        return len([c for c in text if c in ".,;:!?"])

    def get_quotation_count(self, text):
        return text.count('"')

    def get_character_count(self, text):
        return len(text)

    def get_uppercase_words_relative(self, text):
        words = word_tokenize(text)
        uppercase_words = [word for word in words if word.isupper()]
        return len(uppercase_words) / len(words) if words else 0

    def get_personal_pronoun_count(self, text):
        personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        words = word_tokenize(text.lower())
        return len([word for word in words if word in personal_pronouns])

    def get_personal_pronoun_relative(self, text):
        personal_pronouns_count = self.get_personal_pronoun_count(text)
        total_words = len(word_tokenize(text))
        return personal_pronouns_count / total_words if total_words else 0
    
    def get_pos_per_sentence_mean(self, text):
        """
        Calculates the mean number of unique POS tags per sentence in the given text.
        """
        sentences = sent_tokenize(text)
        unique_pos_counts = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            pos_only = [tag for word, tag in pos_tags]
            unique_pos_count = len(set(pos_only))
            unique_pos_counts.append(unique_pos_count)

        mean_unique_pos_per_sentence = np.mean(unique_pos_counts)        
        return mean_unique_pos_per_sentence

    def detect_spelling_and_grammar_errors(self, text: str) -> int:
        """Detects the number of spelling and grammar errors in a given text."""
        matches = self.language_tool.check(text)
        return len(matches)
    
    def count_multiple_blanks(self, text: str) -> int:
        """Counts occurrences of multiple consecutive spaces (blanks) in a text."""
        pattern = r'\s{2,}'
        return len(re.findall(pattern, text))
    
    def get_readability_scores(self, text: str) -> Tuple[float, float]:
        ease = textstat.flesch_reading_ease(text)
        grade_level = textstat.flesch_kincaid_grade(text)
        return ease, grade_level
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extracts TF-IDF features from the dataset."""
        print("Extracting TF-IDF features...")
        start_time = time.time()
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        print(f"Extracted TF-IDF features in {format_time(time.time() - start_time)}")
        return tfidf_features

    def extract_sentence_bert_features(self, texts: List[str]) -> np.ndarray:
        """Extracts Sentence-BERT features from the dataset."""
        print("Extracting Sentence-BERT features...")
        start_time = time.time()
        sentence_bert_features = np.array(self.sentence_model.encode(texts))
        print(f"Extracted Sentence-BERT features in {format_time(time.time() - start_time)}")
        return sentence_bert_features

    def calculate_average_sentence_bert_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Computes the average cosine distance from each text's embedding to all other text embeddings.
        
        Parameters:
        - embeddings (np.ndarray): A NumPy array of shape (n_texts, n_features), where each row represents
        the Sentence-BERT embedding of a text.
        
        Returns:
        - np.ndarray: A one-dimensional array of shape (n_texts,), where each element is the average cosine
        distance of a text to all other texts.
        """
        num_texts = embeddings.shape[0]
        average_distances = np.zeros(num_texts)
        cpu_count = self.cpu_core_count
        initial_batch_size = max(1, num_texts // cpu_count)
        batch_size = initial_batch_size

        def compute_batch(start, end):
            batch_distances = squareform(pdist(embeddings[start:end], 'cosine'))
            np.fill_diagonal(batch_distances, np.nan)
            return np.nanmean(batch_distances, axis=1)

        while batch_size > 0:
            try:
                results = Parallel(n_jobs=cpu_count)(
                    delayed(compute_batch)(start, min(start + batch_size, num_texts))
                    for start in tqdm(range(0, num_texts, batch_size),
                                    desc=f"Calculating average Sentence-BERT distances with batch size {batch_size}")
                )

                for start, batch_result in zip(range(0, num_texts, batch_size), results):
                    end = min(start + batch_size, num_texts)
                    average_distances[start:end] = batch_result
                
                break  # If successful, exit the loop
            except MemoryError:
                print(f"Batch size {batch_size} too large, reducing...")
                batch_size //= 2  # Reduce batch size in half if a MemoryError occurs

        return average_distances
    
    def get_average_num_dependent_clauses(self, text: str) -> float:
        doc = self.nlp(text)
        total_clauses = 0
        sentences = list(doc.sents)
        clause_types = ['advcl', 'relcl', 'acl', 'csubj', 'ccomp', 'xcomp']
        for sentence in sentences:
            clauses = [tok for tok in sentence if tok.dep_ in clause_types]
            total_clauses += len(clauses)
        return total_clauses / len(sentences) if sentences else np.nan

    def count_passive_voice(self, text):
        doc = self.nlp(text)
        passive_count = 0
        for sent in doc.sents:
            matches = self.matcher(sent)
            if any(self.nlp.vocab.strings[match_id] == 'Passive' for match_id, start, end in matches):
                passive_count += 1
        return passive_count

    def get_ngram_entropy(self, text, max_n=3):
        """
        Calculate the entropy for n-grams with n ranging from 1 to max_n.
        Returns a tuple of entropy values.
        """
        words = word_tokenize(text.lower())
        entropies = []

        for n in range(1, max_n + 1):
            n_grams = list(ngrams(words, n)) if n > 1 else words  # Handle unigrams directly
            if not n_grams:
                entropies.append(0)
                continue
            freq_dist = FreqDist(n_grams)
            total_n_grams = len(n_grams)
            entropy = -sum((freq / total_n_grams) * math.log(freq / total_n_grams, 2) for freq in freq_dist.values())
            entropies.append(entropy)

        return tuple(entropies)

    def calculate_burstiness(self, text):
        words = word_tokenize(text)
        freq_dist = FreqDist(words)
        total_words = len(words)
        if total_words == 0 or len(freq_dist) == 0:  # Check if no words or no valid frequency distribution
            return np.nan  # Return zero burstiness for texts with no words or only stop words

        mean = total_words / len(freq_dist)  # Average frequency
        var = sum((freq_dist[word] - mean) ** 2 for word in freq_dist) / len(freq_dist)  # Variance
        return var / mean  # Burstiness index

    def calculate_mean_syntactic_tree_depth(self, text: str):
        """
        Calculate the mean syntactic tree depth of all sentences in a given text.
        """
        doc = self.nlp(text)

        def get_tree_depth(token):
            """Iteratively calculate the depth of the tree rooted at the given token."""
            max_depth = 0
            stack = [(token, 1)]  # Start with the root token and depth 1
            while stack:
                current_token, depth = stack.pop()
                max_depth = max(max_depth, depth)
                stack.extend((child, depth + 1) for child in current_token.children)
            return max_depth
        
        sentence_depths = [get_tree_depth(sent.root) for sent in doc.sents]
        return sum(sentence_depths) / len(sentence_depths) if sentence_depths else np.nan
    
    def count_list_items(self, text):
        pattern = re.compile(r'^\s*([\*\-\+]|(\d+\.))\s', re.MULTILINE)
        matches = pattern.findall(text)
        return len(matches)

def extract_features_from_evasive_texts():
    multiprocessing.freeze_support()
    start_time = time.time()

    feature_extractor = FeatureExtractor()
    print("Number of CPU cores:", feature_extractor.cpu_core_count)

    datasets = ['advanced']

    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        dataset_path = f'data/evasive_texts/{dataset}.csv'
        feature_files = {
            'llm_pred': f'data/evasive_texts/feature/{dataset}/{dataset}_ai_feedback.csv',
            'error_count': f'data/evasive_texts/feature/{dataset}/{dataset}_error_features.csv',
            'max_perplexity': f'data/evasive_texts/feature/{dataset}/{dataset}_perplexity.csv',
            'mean_perplexity': f'data/evasive_texts/feature/{dataset}/{dataset}_perplexity.csv'
        }
        output_dir = f'data/evasive_texts/feature/{dataset}/'
        
        X, y = feature_extractor.load_or_process_data(
            dataset_path,
            feature_files,
            output_dir,
            subset_size=None
        )
        print(f"Completed processing {dataset} dataset.")
        print(get_data_statistics(X, y))

    print(f"Total runtime: {format_time(time.time() - start_time)}")

def main():
    multiprocessing.freeze_support()
    start_time = time.time()

    feature_extractor = FeatureExtractor()
    print("Number of CPU cores:", feature_extractor.cpu_core_count)
    X, y = feature_extractor.load_or_process_data(subset_size=1000)

    print(get_data_statistics(X, y))

    train_and_test_logistic_regression(X, y)

    print(f"Total runtime: {format_time(time.time() - start_time)}")

def train_and_test_logistic_regression(X, y):
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    model = LogisticRegression(max_iter=100000)
    print("Fitting model...")
    model.fit(X_train, y_train)

    feature_importance = pd.DataFrame(model.coef_[0], index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importance)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)

    print_metrics(y_test, y_pred)

    feature_importance = pd.Series(model.coef_[0], index=X.columns)
    plot_feature_importance(feature_importance, X.columns, exclude_features=['tfidf', 'sbert'])

if __name__ == '__main__':
    main()