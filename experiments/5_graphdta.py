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
from models.GraphDTA.graphdta import GraphDTA
from models.GraphDTA.prepare_data import GraphDTADataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def graphdta_forward_step(batch, model):
    """
    Forward step function specific to GraphDTA model
    Args:
        batch: A batch of PyG data objects
        model: The GraphDTA model instance
    Returns:
        scores: Model predictions
        labels: Ground truth labels
    """
    batch = batch.to(device)
    output = model(batch)
    output = output.view(-1)
    labels = batch.y.view(-1)
    return output, labels

def get_graphdta_dataloader(dataset, batch_size, shuffle=False):
    """
    Custom dataloader function for GraphDTA model
    Args:
        dataset: The dataset to convert
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
    Returns:
        DataLoader: A PyG DataLoader configured for GraphDTA data
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle
    )

class TrainingArguments:
    def __init__(self):
        self.model_name = "GraphDTA"
        self.learning_rate = 5e-4
        self.weight_decay = 5e-6
        self.num_epochs = 1000
        self.batch_size = 512
        self.seed = 42
        self.log_steps = 20
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = ""
        # Allow command-line overrides (e.g. --dataset davis --num-epochs 1)
        apply_training_cli_overrides(self)

def main():
    args = TrainingArguments()
    base_dir = str(get_repo_root())

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load datasets
    train_set = GraphDTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='train',
        cold_start=args.cold_start
    )
    
    test_set = GraphDTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='test',
        cold_start=args.cold_start
    )

    # Initialize model
    model = GraphDTA().to(device)
    print(f"Model initialized, using device: {device}")


    # Initialize trainer
    trainer = DTATrainer(
        model=model,
        train_args=args,
        train_dataset=train_set,
        val_dataset=test_set,
        forward_step=graphdta_forward_step,
        get_dataloader_fn=get_graphdta_dataloader
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
