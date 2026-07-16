import os
import json
import pickle
import numpy as np
from collections import OrderedDict
from typing import List

# Set base path
BASE_DIR = "/home/zh/workspace/BridgeDTA"
DATASETS = ['davis', 'kiba']

def process_affinity(affinity, dataset):
    """Process affinity data"""
    if dataset == 'davis':
        return [-np.log10(y/1e9) for y in affinity]
    return affinity

def write_data_to_file(filepath, rows, cols, affinity):
    """Write data to file"""
    with open(filepath, 'w') as f:
        for pair_ind in range(len(rows)):
            data = [
                rows[pair_ind],
                cols[pair_ind],
                affinity[rows[pair_ind], cols[pair_ind]]
            ]
            f.write(' '.join(map(str, data)) + '\n')

if __name__ == "__main__":
    for dataset in DATASETS:
        # Set dataset path
        fpath = os.path.join(BASE_DIR, 'data', dataset)
        processed_dir = os.path.join(fpath, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        # Load fold data
        train_fold = json.load(open(os.path.join(fpath, "folds", "train_fold_setting1.txt")))
        valid_fold = json.load(open(os.path.join(fpath, "folds", "test_fold_setting1.txt")))
        folds = train_fold + [valid_fold]
        
        # Prepare validation and training folds
        valid_ids = [5, 4, 3, 2, 1]
        valid_folds = [folds[vid] for vid in valid_ids]
        train_folds: List[List[int]] = [
            sum([folds[j] for j in range(6) if j != valid_ids[i]], [])
            for i in range(5)
        ]

        # Load data
        ligands = json.load(open(os.path.join(fpath, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        proteins = json.load(open(os.path.join(fpath, "proteins.txt")), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(os.path.join(fpath, "Y"), "rb"), encoding='latin1')
        affinity = process_affinity(affinity, dataset)
        affinity = np.asarray(affinity)

        # Process training and test data
        for i in range(5):
            train_fold = train_folds[i]
            valid_fold = valid_folds[i]
            
            for opt in ['train', 'test']:
                rows, cols = np.where(np.isnan(affinity) == False)
                fold_data = train_fold if opt == 'train' else valid_fold
                fold_data = np.array(fold_data)  # Convert to numpy array
                
                # Use boolean indexing
                rows = rows[fold_data]
                cols = cols[fold_data]

                if i == 0:
                    # Generate standard data
                    print(f'Generating standard {opt} data for {dataset}')
                    write_data_to_file(
                        os.path.join(processed_dir, f'{opt}.txt'),
                        rows, cols, affinity
                    )

                # Generate 5-fold validation data
                print(f'Generating fold {i+1} {opt} data for {dataset}')
                write_data_to_file(
                    os.path.join(processed_dir, f'{opt}_{i}.txt'),
                    rows, cols, affinity
                )

