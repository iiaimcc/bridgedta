"""
GS-DTA 5-fold cross-validation script.
Usage: python gsdta.py <fold>  (fold: 0-4)
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch_geometric

import os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from paths import setup_imports, get_repo_root, apply_training_cli_overrides
setup_imports()

from trainer import DTATrainer
from models.GS_DTA.gs_dta import GS_DTA
from models.GS_DTA.create_data import GS_DTADataset

# Get fold number
fold = sys.argv[1] if len(sys.argv) > 1 else "0"

def gs_dta_forward_step(batch, model):
    """
    Forward step function specific to GS-DTA model
    """
    # Pass the entire batch directly to the model
    output = model(batch)
    labels = batch.y
    
    # Ensure correct dimensions
    output = output.view(-1)
    labels = labels.view(-1)
    
    return output, labels

def get_gs_dta_dataloader(dataset, batch_size, shuffle=False):
    """
    Custom dataloader function for GS-DTA model using PyG's DataLoader
    """
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=['x', 'edge_index'],
        collate_fn=collate_gs_dta
    )

def collate_gs_dta(data_list):
    """
    Custom collate function for GS-DTA data
    """
    # First get all target data
    targets = torch.stack([data.target for data in data_list])
    
    # Use PyG's default batch method to process other data
    batch = torch_geometric.data.Batch.from_data_list(data_list)
    
    # Ensure target data is correctly added to the batch
    batch.target = targets
    
    return batch

class TrainingArguments:
    def __init__(self):
        self.model_name = f"GS_DTA_{fold}"
        self.learning_rate = 5e-4
        self.num_epochs = 1000
        self.batch_size = 512
        self.seed = 42
        self.log_steps = 20
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = fold
        # Allow command-line overrides (e.g. --dataset davis --num-epochs 1)
        apply_training_cli_overrides(self)

def main():
    args = TrainingArguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = str(get_repo_root())

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load datasets
    train_set = GS_DTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='train',
        cold_start=args.cold_start,
        device=device
    )
    test_set = GS_DTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='test',
        cold_start=args.cold_start,
        device=device
    )

    # Initialize model
    model = GS_DTA().to(device)
    print(f"Model initialized, using device: {device}")

    # Initialize trainer
    trainer = DTATrainer(
        model=model,
        train_args=args,
        train_dataset=train_set,
        val_dataset=test_set,
        forward_step=gs_dta_forward_step,
        get_dataloader_fn=get_gs_dta_dataloader
    )

    # Train model
    print('Starting training...')
    best_model = trainer.train()

    # Test model
    print('Testing model...')
    predictions, test_metrics = trainer.predict(test_set)
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
