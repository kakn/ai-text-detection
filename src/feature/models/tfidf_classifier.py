import os
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

from src.utils import format_time, load_balanced_dataset, print_metrics

class TfidfClassifier:
    def __init__(self, subset_size=None, test_size=0.1, random_state=42, model_name='xgboost'):
        self.subset_size = subset_size
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        self.model_name = model_name.lower()
        self.model_save_path = f'data/saved_models/tfidf/{self.model_name}'
        self.tfidf_save_path = f'data/saved_models/tfidf/{self.model_name}/tfidf_matrix.pkl'
        self.grid_search_results_path = f'data/saved_models/tfidf/{self.model_name}/grid_search_results.pkl'
        self.model_file = os.path.join(self.model_save_path, 'tfidfclassifier_model.pkl')
        self.vectorizer_file = os.path.join(self.model_save_path, 'tfidf_vectorizer.pkl')
        self._ensure_directory_exists(self.model_save_path)
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_name == 'xgboost':
            return XGBClassifier(random_state=42)
        else:
            return LogisticRegression(max_iter=10000)

    def _ensure_directory_exists(self, path):
        os.makedirs(path, exist_ok=True)

    def load_data(self):
        print("Loading dataset...")
        dataset = load_balanced_dataset(subset_size=self.subset_size)
        texts = [entry['text'] for entry in dataset]
        labels = [1 if entry['source'] == 'ai' else 0 for entry in dataset]
        return texts, labels

    def preprocess_texts(self, texts):
        print("Lowercasing texts...")
        lowercased_texts = [text.lower() for text in tqdm(texts, desc="Lowercasing texts")]
        return lowercased_texts

    def prepare_data(self, texts, labels):
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=self.test_size, random_state=self.random_state)

        if os.path.exists(self.tfidf_save_path) and os.path.exists(self.vectorizer_file):
            print("Loading TF-IDF matrix from disk...")
            X_train = joblib.load(self.tfidf_save_path)
            print("TF-IDF matrix loaded successfully.")
        else:
            X_train = self.preprocess_texts(X_train)
            X_train = self.vectorizer.fit_transform(tqdm(X_train, desc="TFIDF Vectorization"))
            print("Saving TF-IDF matrix and vectorizer to disk...")
            joblib.dump(X_train, self.tfidf_save_path)
            joblib.dump(self.vectorizer, self.vectorizer_file)
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        print("Training model...")
        if self.model_name == 'xgboost':
            param_grid = {
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [2000, 2500, 3000, 3500, 4000],
                'colsample_bytree': [0.6, 0.8],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [0.01, 0.1]
            }

            grid_search = RandomizedSearchCV(self.model, param_grid, scoring='accuracy', n_jobs=64, n_iter=20, cv=3, random_state=42, verbose=10)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            self.model = grid_search.best_estimator_

            print(f"Saving grid search results to {self.grid_search_results_path}...")
            joblib.dump({'best_params': best_params, 'best_score': grid_search.best_score_, 'cv_results': grid_search.cv_results_}, self.grid_search_results_path)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        print("Making predictions...")
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        print("Evaluating model...")
        print(classification_report(y_true, y_pred))
        print_metrics(y_true, y_pred)

    def save_model(self):
        print(f"Saving model to {self.model_file} and vectorizer to {self.vectorizer_file}...")
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.vectorizer, self.vectorizer_file)

    def load_model(self):
        if (os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file)):
            print(f"Loading model from {self.model_file} and vectorizer from {self.vectorizer_file}...")
            self.model = joblib.load(self.model_file)
            self.vectorizer = joblib.load(self.vectorizer_file)
            return True
        return False

    def run(self):
        if self.load_model():
            print("Model loaded successfully. Skipping training.")
            texts, labels = self.load_data()
            _, X_test, _, y_test = self.prepare_data(texts, labels)
        else:
            texts, labels = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(texts, labels)
            self.train(X_train, y_train)
            self.save_model()
        X_test = self.preprocess_texts(X_test)
        X_test = self.vectorizer.transform(X_test)
        y_pred = self.predict(X_test)
        self.evaluate(y_test, y_pred)

def visualize_tfidf_features():
    classifier = TfidfClassifier()
    classifier.load_model()

    texts, labels = classifier.load_data()
    X_train, _, _, _ = classifier.prepare_data(texts, labels)

    print("Extracting feature names...")
    feature_names = classifier.vectorizer.get_feature_names_out()

    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(classifier.model)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_train)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_train.toarray(), feature_names=feature_names, show=False)
    plt.title('Feature Importance')
    plt.savefig('data/tfidf_shap_summary_array_final_unfiltered.png')
    plt.close()
    print("SHAP summary plot saved.")

def run_model(subset_size=None, model_name='xgboost'):
    start_time = time.time()
    classifier = TfidfClassifier(subset_size=subset_size, model_name=model_name)
    classifier.run()
    print(f"Total runtime: {format_time(time.time() - start_time)}")

def evaluate_evasive_texts():
    # Initialize the classifier and load the model
    classifier = TfidfClassifier()
    if not classifier.load_model():
        print("Model could not be loaded. Exiting.")
        return
    
    # Load the normal dataset to get labels
    texts, labels = classifier.load_data()
    
    # Split the data into training and test sets
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    # Predict and evaluate on the normal dataset
    X_test_preprocessed = classifier.preprocess_texts(X_test)
    X_test_transformed = classifier.vectorizer.transform(tqdm(X_test_preprocessed, desc="TFIDF Vectorization"))
    y_pred = classifier.predict(X_test_transformed)
    print("Evaluation for the normal dataset:")
    print(classification_report(y_test, y_pred))
    print_metrics(y_test, y_pred)
    
    # Define the evasive text types
    evasive_types = ["control", "basic", "advanced"]

    X_test_human = [text for text, label in zip(X_test, y_test) if label == 0]
    
    for e_t in evasive_types:
        evasive_texts_path = f"data/evasive_texts/{e_t}.csv"
        evasive_df = pd.read_csv(evasive_texts_path)
        evasive_texts = evasive_df['rewritten_text'].tolist()
        num_evasive_texts = len(evasive_texts)
        evasive_labels = [1] * num_evasive_texts

        human_texts = X_test_human[:num_evasive_texts]
        human_labels = [0] * num_evasive_texts

        print(f"evasive_texts type: {type(evasive_texts)}")  # Should be list
        print(f"human_texts type: {type(human_texts)}")      # Should be list

        combined_texts = evasive_texts + human_texts
        combined_labels = evasive_labels + human_labels

        combined_texts_preprocessed = classifier.preprocess_texts(combined_texts)
        X_combined = classifier.vectorizer.transform(tqdm(combined_texts_preprocessed, desc="TFIDF Vectorization"))
        y_pred_combined = classifier.predict(X_combined)

        print(f"Evaluation for {e_t} dataset:")
        print(classification_report(combined_labels, y_pred_combined))
        print_metrics(combined_labels, y_pred_combined)

if __name__ == "__main__":
    run_model()