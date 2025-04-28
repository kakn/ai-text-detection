import os
import time
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.feature.feature_extraction import FeatureExtractor
from src.utils import format_time, print_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FeatureClassifierModel:
    def __init__(self, subset_size=None, test_size=0.1, random_state=42):
        self.subset_size = subset_size
        self.test_size = test_size
        self.random_state = random_state
        self.model_save_path = 'data/saved_models/feature_old/xgboost'
        self.grid_search_results_path = f'{self.model_save_path}/grid_search_results.pkl'
        self.model_file = os.path.join(self.model_save_path, 'xgboost_model.pkl')
        self._ensure_directory_exists(self.model_save_path)
        self.model = XGBClassifier(random_state=random_state)
        self.feature_names = self.load_feature_names()

    def _ensure_directory_exists(self, path):
        os.makedirs(path, exist_ok=True)

    def load_feature_names(self):
        feature_path = 'data/saved_data/full/features.pkl'
        if os.path.exists(feature_path):
            df = pd.read_pickle(feature_path)
            return df.columns.tolist()
        else:
            raise FileNotFoundError(f'{feature_path} not found.')

    def load_data(self):
        feature_extractor = FeatureExtractor()
        print("Number of CPU cores:", feature_extractor.cpu_core_count)
        X, y = feature_extractor.load_or_process_data(self.subset_size)
        return X, y
    
    def load_data_from_pkl(self):
        feature_path = 'data/saved_data/1000/features.pkl'
        labels_path = 'data/saved_data/1000/labels.pkl'
        if os.path.exists(feature_path) and os.path.exists(labels_path):
            X = pd.read_pickle(feature_path)
            y = pd.read_pickle(labels_path)
            return X, y
        else:
            raise FileNotFoundError('Features or labels file not found.')
        
    def prepare_data(self, X, y):
        # print(f"Num columns before dropping: {len(X.columns.tolist())}")
        # columns_to_drop = [
        #     'dependent_clauses_count_avg',
        #     'passive_voice_count',
        #     'unigram_entropy',
        #     'bigram_entropy',
        #     'trigram_entropy',
        #     'burstiness',
        #     'syntactic_tree_depth_mean',
        #     'list_item_count'
        # ]
        # X = X.drop(columns=columns_to_drop, errors='ignore')
        # print(f"Num columns after dropping: {len(X.columns.tolist())}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=self.test_size, random_state=self.random_state)
        self.X_train = X_train
        self.y_train = y_train 
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        print("Training model with hyperparameter tuning...")
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [2000, 2500, 3000, 3500, 4000],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.6, 0.8],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0.01, 0.1]
        }

        grid_search = RandomizedSearchCV(self.model, param_grid, scoring='accuracy', n_jobs=48, n_iter=20, cv=3, random_state=self.random_state, verbose=10)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        self.model = grid_search.best_estimator_

        print(f"Saving grid search results to {self.grid_search_results_path}...")
        joblib.dump({'best_params': best_params, 'best_score': grid_search.best_score_, 'cv_results': grid_search.cv_results_}, self.grid_search_results_path)

    def predict(self, X_test):
        print("Making predictions...")
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        print("Evaluating model...")
        print(classification_report(y_true, y_pred))
        print_metrics(y_true, y_pred)

    def save_model(self):
        print(f"Saving model to {self.model_file}...")
        joblib.dump(self.model, self.model_file)

    def load_model(self):
        if os.path.exists(self.model_file):
            print(f"Loading model from {self.model_file}...")
            self.model = joblib.load(self.model_file)
            return True
        return False

    def run(self):
        if self.load_model():
            print("Model loaded successfully. Skipping training.")
            X, y = self.load_data()
            _, X_test, _, y_test = self.prepare_data(X, y)
        else:
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            self.train(X_train, y_train)
            self.save_model()
        y_pred = self.predict(X_test)
        self.evaluate(y_test, y_pred)

    def visualize_feature_importance(self):
        X, y = self.load_data_from_pkl()
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_train)

        shap.summary_plot(shap_values, X_train, feature_names=self.feature_names, show=False)
        plt.title('Feature Importance')
        plt.savefig('data/figures/shap_summary.png')
        plt.close()

    def get_evasive_predictions(self):
        self.load_model()
        X, y = self.load_data()
        _, X_test, _, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Extract human texts from the test set
        X_test_human = X_test[y_test == 0]

        # Convert human texts to DataFrame with correct feature dimensions
        X_test_human = pd.DataFrame(X_test_human, columns=self.feature_names)

        print("Human test data shape:", X_test_human.shape)

        evasive_types = ["control", "basic", "advanced"]
        columns_to_drop = [
            'dependent_clauses_count_avg',
            'passive_voice_count',
            'unigram_entropy',
            'bigram_entropy',
            'trigram_entropy',
            'burstiness',
            'syntactic_tree_depth_mean',
            'list_item_count'
        ]
        for e_t in evasive_types:
            path_to_features = f"data/evasive_texts/feature/{e_t}/"
            features_path = os.path.join(path_to_features, "full_features.pkl")
            labels_path = os.path.join(path_to_features, "full_labels.pkl")

            X_evasive = pd.read_pickle(features_path)
            y_evasive = pd.read_pickle(labels_path)

            print(f"{e_t.capitalize()} data shape before reindexing:", X_evasive.shape)

            # Ensure both datasets have the same columns
            X_evasive = X_evasive.reindex(columns=self.feature_names, fill_value=0)

            print(f"{e_t.capitalize()} data shape after reindexing:", X_evasive.shape)

            # Truncate human texts to match the length of evasive texts
            X_test_human_truncated = X_test_human.iloc[:len(X_evasive)]
            y_test_human_truncated = [0] * len(X_test_human_truncated)

            print("Truncated human test data shape:", X_test_human_truncated.shape)

            # Combine evasive data with truncated human data
            combined_X_test = pd.concat([X_test_human_truncated, X_evasive])
            combined_y_test = y_test_human_truncated + y_evasive.tolist()

            print("Combined test data shape:", combined_X_test.shape)

            combined_X_test = combined_X_test.drop(columns=columns_to_drop, errors='ignore')

            # Convert to numpy arrays for scaling and prediction
            combined_X_test = combined_X_test.values
            combined_y_test = np.array(combined_y_test)

            # Scale the combined data
            scaler = StandardScaler()
            combined_X_test_scaled = scaler.fit_transform(combined_X_test)

            # Make predictions
            y_pred = self.predict(combined_X_test_scaled)

            # Evaluate
            print(f"Evaluation for {e_t} dataset:")
            self.evaluate(combined_y_test, y_pred)

def run_model(subset_size=None):
    start_time = time.time()
    classifier = FeatureClassifierModel(subset_size=subset_size)
    classifier.run()
    print(f"Total runtime: {format_time(time.time() - start_time)}")

def visualize_features():
    classifier = FeatureClassifierModel()
    classifier.load_model()
    classifier.visualize_feature_importance()

def get_evasive_pred():
    classifier = FeatureClassifierModel()
    classifier.get_evasive_predictions()

if __name__ == "__main__":
    run_model()