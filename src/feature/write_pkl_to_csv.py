import os
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np

def write_pkl_to_csv(pkl_folder='data/saved_data/full'):
    pkl_files_to_write_to_csv = ['features.pkl', 'labels.pkl']

    for pkl_file in tqdm(pkl_files_to_write_to_csv, desc="Processing files"):
        pkl_path = os.path.join(pkl_folder, pkl_file)
        csv_path = os.path.splitext(pkl_path)[0] + '.csv'
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(csv_path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
        
        print(f"Saved {csv_path}")

def get_pkl_sizes(pkl_folder='data/saved_data/full'):
    pkl_files = ['features.pkl', 'labels.pkl']
    for file in pkl_files:
        file_path = os.path.join(pkl_folder, file)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                rows, cols = data.shape
                print(f'{file} has {rows} rows and {cols} columns')
            elif isinstance(data, pd.Series):
                rows = data.shape[0]
                print(f'{file} has {rows} rows and 1 column (Series)')
            elif isinstance(data, np.ndarray):
                rows = data.shape[0]
                cols = 1 if len(data.shape) == 1 else data.shape[1]
                print(f'{file} has {rows} rows and {cols} columns (NumPy array)')
            else:
                print(f'{file} is neither a DataFrame, Series, nor NumPy array')
        else:
            print(f'{file} not found in {pkl_folder}')