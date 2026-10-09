import os
import sys
import json
import pickle
import numpy as np
from collections import OrderedDict
from typing import List

# Resolve the repository root (this file lives in <repo>/datahelper).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Base path
BASE_DIR = _REPO_ROOT
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
        train_folds = json.load(open(os.path.join(fpath, "folds", "train_fold_setting1.txt")))
        fixed_test_fold = json.load(open(os.path.join(fpath, "folds", "test_fold_setting1.txt")))
        
        # Prepare validation and training folds
        valid_folds = train_folds
        fit_folds: List[List[int]] = [
            sum([train_folds[j] for j in range(5) if j != i], [])
            for i in range(5)
        ]
        full_train_fold = sum(train_folds, [])

        # Load data
        ligands = json.load(open(os.path.join(fpath, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        proteins = json.load(open(os.path.join(fpath, "proteins.txt")), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(os.path.join(fpath, "Y"), "rb"), encoding='latin1')
        affinity = process_affinity(affinity, dataset)
        affinity = np.asarray(affinity)

        rows, cols = np.where(np.isnan(affinity) == False)

        # Generate the fixed outer training and test partitions
        write_data_to_file(
            os.path.join(processed_dir, 'train.txt'),
            rows[np.array(full_train_fold)], cols[np.array(full_train_fold)], affinity
        )
        write_data_to_file(
            os.path.join(processed_dir, 'test.txt'),
            rows[np.array(fixed_test_fold)], cols[np.array(fixed_test_fold)], affinity
        )

        # Generate internal five-fold training and validation partitions
        for i in range(5):
            train_fold = fit_folds[i]
            valid_fold = valid_folds[i]
            
            for opt in ['train', 'test']:
                fold_data = train_fold if opt == 'train' else valid_fold
                fold_data = np.array(fold_data)  # Convert to numpy array
                
                # Use boolean indexing
                fold_rows = rows[fold_data]
                fold_cols = cols[fold_data]

                # Generate 5-fold validation data
                print(f'Generating fold {i+1} {opt} data for {dataset}')
                write_data_to_file(
                    os.path.join(processed_dir, f'{opt}_{i}.txt'),
                    fold_rows, fold_cols, affinity
                )

