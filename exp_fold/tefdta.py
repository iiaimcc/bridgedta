"""
TEFDTA 5-fold cross-validation script.
Usage: python tefdta.py <fold>  (fold: 0-4)
"""
import sys
import torch
import logging
import pandas as pd
from torch.utils.data import DataLoader, random_split

sys.path.append("/home/zh/workspace/BridgeDTA")

from trainer import DTATrainer
from models.TEFDTA.tefdta import TEFDTA
from models.TEFDTA.process_data import TEFDTADataset, CHARPROTLEN, MACCSLEN

# 获取 fold 编号
fold = sys.argv[1] if len(sys.argv) > 1 else "0"

def tefdta_forward_step(batch, model):
    """
    Forward step function specific to TEFDTA model
    """
    smiles, fasta, labels = batch
    scores = model(smiles, fasta).view(-1)
    return scores, labels

def get_tefdta_dataloader(dataset, batch_size, shuffle=False):
    """
    Custom dataloader function for TEFDTA model
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

class TrainingArguments:
    def __init__(self):
        self.model_name = f"TEFDTA_{fold}"
        self.learning_rate = 0.0001
        self.num_epochs = 200
        self.batch_size = 256
        self.seed = 42
        self.log_steps = 10
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = fold

def main():
    args = TrainingArguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = "/home/zh/workspace/BridgeDTA"

    # 设置数据集相关参数
    if args.dataset == 'davis':
        max_smiles_len = 85
        max_fasta_len = 1000
    elif args.dataset == 'kiba':
        max_smiles_len = 100
        max_fasta_len = 1000
    elif args.dataset == 'Bind':
        max_smiles_len = 100
        max_fasta_len = 1000

    # 创建训练和测试数据集
    train_set = TEFDTADataset(
        root=f"{base_dir}/data",
        dataset=args.dataset.lower(),
        mode='train',
        cold_start=args.cold_start,
        device=device,
        max_smiles_len=max_smiles_len,
        max_fasta_len=max_fasta_len
    )

    test_set = TEFDTADataset(
        root=f"{base_dir}/data",
        dataset=args.dataset.lower(),
        mode='test',
        cold_start=args.cold_start,
        device=device,
        max_smiles_len=max_smiles_len,
        max_fasta_len=max_fasta_len
    )

    # 初始化模型
    model = TEFDTA(
        smiles_char_set_len=MACCSLEN,
        fasta_char_set_len=CHARPROTLEN+1
    ).to(device)

    # 初始化训练器
    trainer = DTATrainer(
        model=model,
        train_args=args,
        train_dataset=train_set,
        val_dataset=test_set,
        forward_step=tefdta_forward_step,
        get_dataloader_fn=get_tefdta_dataloader
    )

    # 训练模型
    print('开始训练...')
    best_model = trainer.train()

    # 测试模型
    print('测试模型...')
    predictions, test_metrics = trainer.predict(test_set)
    print(f"测试指标: {test_metrics}")

if __name__ == "__main__":
    main()
