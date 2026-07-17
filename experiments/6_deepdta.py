import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

import os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from paths import setup_imports, get_repo_root, apply_training_cli_overrides
setup_imports()

from trainer import DTATrainer
from models.DeepDTA.deepdta import DeepDTA
from models.DeepDTA.prepare_data import DeepDTADataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure cudnn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.cuda.empty_cache()

def deepdta_forward_step(batch, model):
    """
    Forward step function specific to DeepDTA model
    Args:
        batch: A batch of PyG data objects
        model: The DeepDTA model instance
    Returns:
        scores: Model predictions
        labels: Ground truth labels
    """
    protein, ligand, target = batch
    protein, ligand, target = protein.to(device), ligand.to(device), target.to(device)
    output = model(protein, ligand)
    return output, target

def collate_deepdta(batch):
    smiles, fasta, Y = zip(*batch)
    smiles = torch.stack(smiles, dim=0)
    fasta = torch.stack(fasta, dim=0)
    Y = torch.stack(Y, dim=0)
    return fasta, smiles, Y  # Note: order swapped here to be consistent with forward_step

def get_deepdta_dataloader(dataset, batch_size, shuffle=False):
    """
    Custom dataloader function for DeepDTA model
    Args:
        dataset: The dataset to convert
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
    Returns:
        DataLoader: A PyG DataLoader configured for DeepDTA data
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_deepdta
    )

class TrainingArguments:
    def __init__(self):
        self.model_name = "DeepDTA"
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 500
        self.batch_size = 256
        self.seed = 42
        self.log_steps = 20
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = ""
        self.channel = 32
        self.protein_kernel = 12    # [8, 12]
        self.ligand_kernel = 8     # [4, 8] 
        # Allow command-line overrides (e.g. --dataset davis --num-epochs 1)
        apply_training_cli_overrides(self)

def main():
    args = TrainingArguments()
    base_dir = str(get_repo_root())

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load datasets
    train_set = DeepDTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode='train',
        cold_start=args.cold_start
    )
    
    test_set = DeepDTADataset(
        root=f'{base_dir}/data',
        dataset=args.dataset.lower(),
        mode="test",
        cold_start=args.cold_start
    )

    # Initialize model - fix vocab size calculation
    protein_vocab_size = len(train_set.fasta_vocab)  # already includes special tokens
    ligand_vocab_size = len(train_set.smiles_vocab)  # already includes special tokens
    
    print(f"Protein vocab size: {protein_vocab_size}")
    print(f"Ligand vocab size: {ligand_vocab_size}")
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    model = DeepDTA(protein_vocab_size, ligand_vocab_size, args.channel, args.protein_kernel, args.ligand_kernel).to(device)
    print(f"Model initialized, using device: {device}")

    # Initialize trainer
    trainer = DTATrainer(
        model=model,
        train_args=args,
        train_dataset=train_set,
        val_dataset=test_set,
        forward_step=deepdta_forward_step,
        get_dataloader_fn=get_deepdta_dataloader
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