import os
import time

import evaluate
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils import format_time, load_balanced_dataset, print_metrics

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True  # Optimize cuDNN for current GPU workload
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs

class FineTunedLLM:
    """
    A classifier for distinguishing between human and AI-generated text using an LLM.
    """
    def __init__(self, dataset_path="artem9k/ai-text-detection-pile", model_name='distilbert-base-uncased', # meta-llama/Meta-Llama-3-8B
                 num_labels=2, output_dir='./data/model_output', random_seed=42, max_length=512, load_existing_model=False):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_existing_model:
            print(f"Loading existing model and tokenizer from {output_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        else:
            print(f"Initializing a new model from {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, id2label={0: "HUMAN", 1: "AI"}, label2id={"HUMAN": 0, "AI": 1}
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.model.to(self.device)

    def load_and_prepare_data(self, subset_size=None, include_id=False):
        print("Loading dataset...")
        raw_dataset = load_balanced_dataset(subset_size)
        
        label_feature = ClassLabel(names=['human', 'ai'])
        
        def label_encode(example):
            example['labels'] = label_feature.str2int(example['source'])
            return example

        remove_columns = ['source']
        if not include_id:
            remove_columns.append('id')
        labeled_dataset = raw_dataset.map(label_encode, remove_columns=remove_columns)

        print("Tokenizing dataset...")
        processed_dataset = labeled_dataset.map(self.preprocess_function, batched=True)
        
        print("Splitting data into train, validation and test sets...")
        df = processed_dataset.to_pandas()
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=self.random_seed)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=self.random_seed)
        self.test_ids = test_df["id"].tolist() if include_id else None
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'val': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length
        )
        model_inputs['labels'] = examples['labels']
        return model_inputs

    def compute_metrics(self, eval_pred):
        metric = evaluate.combine({
            "accuracy": evaluate.load("accuracy"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
            "f1": evaluate.load("f1")
        })
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self, training_args):
        print("Training model...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['val'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        return trainer

    def save_model(self):
        print("Saving model...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def predict_test_set(self):
        """Generates predictions for the entire test set using PyTorch."""
        self.model.eval()
        predictions = []
        
        for example in tqdm(self.dataset['test'], desc="Making predictions"):
            inputs = self.tokenizer(
                example['text'], 
                return_tensors="pt", 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to the same device as the model
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax(dim=-1).item()
                predictions.append(predicted_class_id)

        return predictions
    
    def make_predictions(self, texts):
        """Generates predictions for a list of texts."""
        self.model.eval()
        predictions = []

        for text in tqdm(texts, desc="Making predictions"):
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to the same device as the model
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax(dim=-1).item()
                predictions.append(predicted_class_id)

        return predictions

def build_model(output_dir, subset_size=None):
    classifier = FineTunedLLM(output_dir=output_dir)
    classifier.load_and_prepare_data(subset_size)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            classifier.model_name, num_labels=2, id2label={0: "HUMAN", 1: "AI"}, label2id={"HUMAN": 0, "AI": 1}
        )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64, 96]),  # Adjusted batch sizes
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
            "warmup_steps": trial.suggest_int("warmup_steps", 0, 500)
        }

    training_args = TrainingArguments(
        output_dir=classifier.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,  # Adjusted for larger batch sizes
        fp16=True,  # Mixed precision training
        dataloader_pin_memory=True,
        dataloader_drop_last=True
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=classifier.dataset['train'],
        eval_dataset=classifier.dataset['val'],
        tokenizer=classifier.tokenizer,
        data_collator=classifier.data_collator,
        compute_metrics=classifier.compute_metrics
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=10,
        hp_space=hp_space
    )

    print(f"Best Hyperparameters: {best_trial.hyperparameters}")

    final_training_args = TrainingArguments(
        output_dir=classifier.output_dir,
        evaluation_strategy="epoch",
        learning_rate=best_trial.hyperparameters['learning_rate'],
        per_device_train_batch_size=best_trial.hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=best_trial.hyperparameters['per_device_train_batch_size'],
        num_train_epochs=best_trial.hyperparameters['num_train_epochs'],
        weight_decay=best_trial.hyperparameters['weight_decay'],
        warmup_steps=best_trial.hyperparameters['warmup_steps'],
        save_strategy="epoch",
        push_to_hub=False,
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,  # Adjusted for larger batch sizes
        fp16=True,  # Mixed precision training
        dataloader_pin_memory=True,
        dataloader_drop_last=True
    )

    classifier.train(final_training_args)
    classifier.save_model()
    return classifier

def load_and_evaluate_model(subset_size=None):
    start_time = time.time()

    output_dir = './data/distilbert_tired_model_output'
    clean_empty_folder(output_dir)
    load_existing_model = os.path.exists(output_dir)
    
    if load_existing_model:
        classifier = FineTunedLLM(load_existing_model=True, output_dir=output_dir)
        classifier.load_and_prepare_data(subset_size)
    else:
        classifier = build_model(output_dir, subset_size)

    y_pred = classifier.predict_test_set()
    y_true = [example['labels'] for example in classifier.dataset['test']]
    print_metrics(y_true, y_pred)

    print(f"Total runtime: {format_time(time.time() - start_time)}")

def clean_empty_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path) and not os.listdir(folder_path):
        os.rmdir(folder_path)
        print(f"Deleted empty folder: {folder_path}")

def make_evasive_predictions():
    start_time = time.time()

    output_dir = './data/distilbert_tired_model_output'
    classifier = FineTunedLLM(load_existing_model=True, output_dir=output_dir)

    # Part 1: Make human predictions
    dataset = load_balanced_dataset(subset_size=None)

    texts_og = [entry['text'] for entry in dataset]
    ids_og = [entry['id'] for entry in dataset]
    labels_og = [1 if entry['source'] == 'ai' else 0 for entry in dataset]

    x_train, x_test, ids_train, ids_test, labels_train, labels_test = train_test_split(
        texts_og, ids_og, labels_og, test_size=0.1, random_state=42
    )

    x_test_human = [text for text, label in zip(x_test, labels_test) if label == 0][:5000]
    ids_test_human = [id_ for id_, label in zip(ids_test, labels_test) if label == 0][:5000]
    labels_test_human = [0] * len(x_test_human)

    human_output_dir = 'data/evasive_texts/llm/human'
    os.makedirs(human_output_dir, exist_ok=True)
    human_dbert_file_name = os.path.join(human_output_dir, 'human_distilbert_pred.csv')

    y_pred_human = classifier.make_predictions(x_test_human)
    results_df_human = pd.DataFrame({
        'id': ids_test_human,
        'y_pred': y_pred_human,
        'y_true': labels_test_human
    })
    
    results_df_human.to_csv(human_dbert_file_name, index=False)
    print(f"Human predictions saved to {human_dbert_file_name}")

    print_metrics(labels_test_human, y_pred_human)

    # Part 2: Make evasive text predictions    
    evasive_datasets = [
        'data/evasive_texts/control.csv', 
        'data/evasive_texts/basic.csv', 
        'data/evasive_texts/advanced.csv'
    ]
    
    for dataset in evasive_datasets:
        df = pd.read_csv(dataset)
        texts = df['rewritten_text'].tolist()
        ids = df['id'].tolist()
        labels = [1] * len(texts)  # Assuming all texts are labeled as AI-generated

        y_pred = classifier.make_predictions(texts)  # Custom prediction method

        # Write to CSV with id, y_pred, y_true columns for each id
        dataset_name = os.path.basename(dataset).replace('.csv', '')
        evasive_output_dir = f'data/evasive_texts/llm/{dataset_name}'
        os.makedirs(evasive_output_dir, exist_ok=True)
        evasive_dbert_file_name = os.path.join(evasive_output_dir, f'{dataset_name}_distilbert_pred.csv')
        
        results_df_evasive = pd.DataFrame({
            'id': ids,
            'y_pred': y_pred,
            'y_true': labels
        })
        
        results_df_evasive.to_csv(evasive_dbert_file_name, index=False)
        print(f"Evasive predictions saved to {evasive_dbert_file_name}")

        print_metrics(labels, y_pred)

    print(f"Total runtime: {format_time(time.time() - start_time)}")