import csv
import time

import pandas as pd
import statsmodels.formula.api as smf

from src.utils import format_time, print_metrics
from src.viz import compute_label_balance

class MixedEffectsModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.result = None

    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)

    def prepare_data(self):
        print("Preparing data...")

        # Define the mapping for 'unique_prompt_id'
        complexity_mapping = {'S': 'Simple', 'M': 'Moderate', 'D': 'Detailed'}
        order_mapping = {'Y': 'Yes/No', 'N': 'No/Yes'}

        # Map 'unique_prompt_id' values to task_description, output_instruction and yes_no_order
        self.df['task_description'] = self.df['unique_prompt_id'].apply(lambda x: complexity_mapping[x[0]])
        self.df['output_instruction'] = self.df['unique_prompt_id'].apply(lambda x: complexity_mapping[x[2]])
        # self.df['output_instruction'] = self.df['unique_prompt_id'].apply(lambda x: f"{complexity_mapping[x[2]]} {order_mapping[x[1]]}")
        self.df['yes_no_order'] = self.df['unique_prompt_id'].apply(lambda x: order_mapping[x[1]])

        # Encode categorical variables as factors
        self.df['task_description'] = self.df['task_description'].astype('category')
        self.df['output_instruction'] = self.df['output_instruction'].astype('category')
        self.df['yes_no_order'] = self.df['yes_no_order'].astype('category')
        
        # Create the 'correct' variable
        # Treat NAs in predictions as incorrect, which is the same as setting 'correct' to 0
        self.df['correct'] = ((self.df['prediction'].notna()) & (self.df['prediction'] == self.df['text_label'])).astype(int)

    def fit_model(self):
        print("Fitting model...")
        # Mixed effects logistic regression model
        self.model = smf.mixedlm(
            "correct ~ task_description + output_instruction + yes_no_order",
            data=self.df,
            groups=self.df["text_id"],
            re_formula="1"
        )
        self.result = self.model.fit()

    def print_summary(self):
        if self.result is not None:
            print(self.result.summary())
        else:
            print("Model has not been fit yet.")

    def write_metrics_to_csv(self, output_path):
        grouped = self.df.groupby('unique_prompt_id')
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['unique_prompt_id', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'None Count', 'Label Balance'])
            
            for unique_prompt_id, group in grouped:
                predictions = group['prediction'].tolist()
                labels = group['text_label'].tolist()

                valid_indices = [i for i, pred in enumerate(predictions) if not pd.isna(pred)]
                valid_predictions = [predictions[i] for i in valid_indices]
                valid_labels = [labels[i] for i in valid_indices]

                accuracy, precision, recall, f1, conf_matrix = print_metrics(valid_labels, valid_predictions, verbose=False)
                label_balance = compute_label_balance(conf_matrix)
                conf_matrix = " ".join(map(str, conf_matrix.flatten()))
                none_count = len(predictions) - len(valid_predictions)

                writer.writerow([unique_prompt_id, accuracy, precision, recall, f1, str(conf_matrix), none_count, label_balance])
                print(f"Unique Prompt ID: {unique_prompt_id}")
                print(f"Metrics: {accuracy, precision, recall, f1, conf_matrix}, None count: {none_count}, Label balance: {label_balance}")

def build_stats_model():
    start_time = time.time()
    data_path = 'data/experiment_results/prompt_experiment_results.csv'
    model = MixedEffectsModel(data_path)
    model.load_data()
    model.prepare_data()
    model.fit_model()
    model.print_summary()
    model.write_metrics_to_csv('data/experiment_results/prompt_metrics.csv')
    print(f"Total runtime: {format_time(time.time() - start_time)}")

if __name__ == "__main__":
    build_stats_model()