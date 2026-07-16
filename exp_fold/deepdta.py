"""
DeepDTA 5-fold cross-validation script.
Usage: python deepdta.py <fold>  (fold: 0-4)
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append("/home/zh/workspace/BridgeDTA")

from trainer import DTATrainer
from models.DeepDTA.deepdta import DeepDTA
from models.DeepDTA.prepare_data import DeepDTADataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获取 fold 编号
fold = sys.argv[1] if len(sys.argv) > 1 else "0"

def deepdta_forward_step(batch, model):
    """
    Forward step function specific to DeepDTA model
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
    return fasta, smiles, Y  # 注意：这里交换了顺序，保持与forward_step一致

def get_deepdta_dataloader(dataset, batch_size, shuffle=False):
    """
    Custom dataloader function for DeepDTA model
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
        self.model_name = f"DeepDTA_{fold}"
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 500
        self.batch_size = 256
        self.seed = 42
        self.log_steps = 20
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = fold
        self.channel = 32
        self.protein_kernel = 12    # [8, 12]
        self.ligand_kernel = 8     # [4, 8] 

def main():
    args = TrainingArguments()
    base_dir = "/home/zh/workspace/BridgeDTA"

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

    # Initialize model
    protein_vocab_size = len(train_set.fasta_vocab)
    ligand_vocab_size = len(train_set.smiles_vocab)
    
    print(f"蛋白质词汇表大小: {protein_vocab_size}")
    print(f"配体词汇表大小: {ligand_vocab_size}")
    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")
    
    model = DeepDTA(protein_vocab_size, ligand_vocab_size, args.channel, args.protein_kernel, args.ligand_kernel).to(device)
    print(f"模型已初始化，使用设备: {device}")

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
