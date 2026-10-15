"""
MultiKD-DTA 5-fold cross-validation script.
Usage: python multikd_dta.py <fold>  (fold: 0-4)
"""
import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from paths import setup_imports, get_repo_root, apply_training_cli_overrides
setup_imports()

from trainer import DTATrainer
from models.MultiKD_DTA.multikd_dta import Predictor
from models.MultiKD_DTA.create_data import MultiKD_DTADataset

# Get fold number
fold = sys.argv[1] if len(sys.argv) > 1 else "0"

# Load esm protein features
base_dir = str(get_repo_root())
z = np.load(f'{base_dir}/models/MultiKD_DTA/features/kiba.npz', allow_pickle=True)
z = z['dict'][()]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def multikd_dta_forward_step(batch, model):
    """
    Forward step function specific to MultiKD-DTA model
    """
    # Process protein/target data
    target = [torch.from_numpy(z[batch.pid[i].item()].squeeze())[:1000, :] for i in range(len(batch.pid))]
    target = pad_sequence(target).permute(1, 0, 2)
    target = target.to(device)
    batch = batch.to(device)
    # Forward pass
    _, output = model(batch, target)
    labels = batch.y.view(-1)
    
    return output, labels

def get_multikd_dta_dataloader(dataset, batch_size, shuffle=False):
    """
    Custom dataloader function for MultiKD-DTA model
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )

class TrainingArguments:
    def __init__(self):
        self.model_name = f"MultiKD_DTA_{fold}"
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 500
        self.batch_size = 256
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

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load datasets
    train_set = MultiKD_DTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='train',
        cold_start=args.cold_start
    )
    
    test_set = MultiKD_DTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='test',
        cold_start=args.cold_start
    )

    # Initialize model
    model = Predictor().to(device)
    print(f"Model initialized, using device: {device}")

    # Initialize trainer
    trainer = DTATrainer(
        model=model,
        train_args=args,
        train_dataset=train_set,
        val_dataset=test_set,
        forward_step=multikd_dta_forward_step,
        get_dataloader_fn=get_multikd_dta_dataloader
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
