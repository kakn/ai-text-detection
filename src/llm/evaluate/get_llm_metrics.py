import pandas as pd

from src.utils import load_balanced_dataset, print_metrics
from sklearn.model_selection import train_test_split

def process_zero_shot_predictions(csv_file='data/features_from_data/llama_feedback_results_dnd.csv'):
    df = pd.read_csv(csv_file)
    df['llm_pred'] = df['llm_pred'].fillna(0)

    y_true = df['label'].tolist()
    y_pred = df['llm_pred'].tolist()
    print_metrics(y_true, y_pred)

def get_distilbert_evasive_metrics():
    # Load human predictions
    human_pred_file = 'data/evasive_texts/llm/human/human_distilbert_pred.csv'
    human_df = pd.read_csv(human_pred_file)

    evasive_datasets = [
        'data/evasive_texts/llm/control/control_distilbert_pred.csv',
        'data/evasive_texts/llm/basic/basic_distilbert_pred.csv',
        'data/evasive_texts/llm/advanced/advanced_distilbert_pred.csv'
    ]

    for evasive_pred_file in evasive_datasets:
        evasive_df = pd.read_csv(evasive_pred_file)
        
        # Truncate human predictions to match the length of current evasive predictions
        human_truncated = human_df.head(len(evasive_df))
        
        # Merge the two predictions
        combined_df = pd.concat([human_truncated, evasive_df])
        
        y_true = combined_df['y_true'].tolist()
        y_pred = combined_df['y_pred'].tolist()
        
        # Print metrics for the combined dataset
        print(f"Metrics for {evasive_pred_file.split('/')[-1].replace('_distilbert_pred.csv', '')}:")
        print_metrics(y_true, y_pred)

def get_zero_shot_evasive_metrics():
    # Load the balanced dataset
    dataset = load_balanced_dataset(subset_size=None)

    texts = [entry['text'] for entry in dataset]
    ids = [entry['id'] for entry in dataset]
    labels = [1 if entry['source'] == 'ai' else 0 for entry in dataset]

    # Split into training and test sets
    x_train, x_test, ids_train, ids_test, labels_train, labels_test = train_test_split(
        texts, ids, labels, test_size=0.1, random_state=42
    )

    # Extract human test IDs
    ids_test_human = [id_ for id_, label in zip(ids_test, labels_test) if label == 0][:5000]

    # Load zero-shot predictions
    zero_shot_preds_file = 'data/features_from_data/llama_feedback_results_dnd.csv'
    zero_shot_df = pd.read_csv(zero_shot_preds_file)
    zero_shot_df.fillna(0, inplace=True)
    # Match the human test IDs to the zero-shot predictions
    zero_shot_human_df = zero_shot_df[zero_shot_df['id'].isin(ids_test_human)]
    y_pred_human = zero_shot_human_df['llm_pred'].tolist()
    y_true_human = zero_shot_human_df['label'].tolist()

    evasive_datasets = [
        'data/evasive_texts/feature/control/control_ai_feedback.csv',
        'data/evasive_texts/feature/basic/basic_ai_feedback.csv',
        'data/evasive_texts/feature/advanced/advanced_ai_feedback.csv'
    ]

    for evasive_pred_file in evasive_datasets:
        evasive_df = pd.read_csv(evasive_pred_file)
        evasive_df.fillna(0, inplace=True)
        # Truncate human predictions to match the length of current evasive predictions
        y_pred_human_truncated = y_pred_human[:len(evasive_df)]
        y_true_human_truncated = y_true_human[:len(evasive_df)]
        
        # Merge the predictions
        y_pred_combined = y_pred_human_truncated + evasive_df['pred'].tolist()
        y_true_combined = y_true_human_truncated + evasive_df['label'].tolist()

        # Print metrics for the combined dataset
        print(f"Metrics for {evasive_pred_file.split('/')[-1].replace('_ai_feedback.csv', '')}:")
        print_metrics(y_true_combined, y_pred_combined)
    
