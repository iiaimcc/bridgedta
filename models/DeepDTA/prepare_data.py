import sys
sys.path.append("/home/zh/workspace/BridgeDTA")

from torch.utils.data import Dataset
import torch
from rdkit import Chem
from datahelper.dataset import DTADataset

class DeepDTADataset(DTADataset):
    def __init__(self, root, dataset, mode, cold_start, device='cpu', max_smiles_len=200, max_fasta_len=2000):
        super().__init__(root, dataset, mode, cold_start, device)
        self.max_smiles_len = max_smiles_len
        self.max_fasta_len = max_fasta_len

        # 预定义SMILES和蛋白质序列的标准字符集
        smiles_chars = set(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'H',
                           '(', ')', '[', ']', '=', '#', '+', '-', '\\', '/', '@', '.', ':', '1', '2', '3',
                           '4', '5', '6', '7', '8', '9', '0', 'c', 'n', 'o', 's', 'p', 'se', 'te', 'b'])
        protein_chars = set(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 
                            'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'U', 'O'])

        # 从实际数据中收集字符
        for smiles in self.smiles:
            for char in smiles:
                smiles_chars.add(char)
        for fasta in self.fasta:
            for char in fasta:
                protein_chars.add(char)

        # 添加特殊标记并创建有序词汇表
        self.smiles_vocab = ['<PAD>', '<UNK>'] + sorted(list(smiles_chars))
        self.fasta_vocab = ['<PAD>', '<UNK>'] + sorted(list(protein_chars))

        # 创建字符到索引的映射
        self.smiles_dict = {char: idx for idx, char in enumerate(self.smiles_vocab)}
        self.fasta_dict = {char: idx for idx, char in enumerate(self.fasta_vocab)}

        print(f"SMILES词汇表大小: {len(self.smiles_vocab)}")
        print(f"蛋白质词汇表大小: {len(self.fasta_vocab)}")

    def __getitem__(self, idx):
        smiles, fasta, Y = super().__getitem__(idx)

        # 转换SMILES和蛋白质序列为索引张量，处理未知字符
        smiles_indices = []
        for char in smiles[:self.max_smiles_len]:
            if char in self.smiles_dict:
                smiles_indices.append(self.smiles_dict[char])
            else:
                smiles_indices.append(self.smiles_dict['<UNK>'])
        
        # 填充到固定长度
        while len(smiles_indices) < self.max_smiles_len:
            smiles_indices.append(self.smiles_dict['<PAD>'])

        fasta_indices = []
        for char in fasta[:self.max_fasta_len]:
            if char in self.fasta_dict:
                fasta_indices.append(self.fasta_dict[char])
            else:
                fasta_indices.append(self.fasta_dict['<UNK>'])
        
        # 填充到固定长度
        while len(fasta_indices) < self.max_fasta_len:
            fasta_indices.append(self.fasta_dict['<PAD>'])

        # Debug: 检查索引是否越界
        assert max(smiles_indices) < len(self.smiles_vocab), f"SMILES索引超界: {max(smiles_indices)} >= {len(self.smiles_vocab)}"
        assert max(fasta_indices) < len(self.fasta_vocab), f"FASTA索引超界: {max(fasta_indices)} >= {len(self.fasta_vocab)}"

        return torch.tensor(smiles_indices, device=self.device), torch.tensor(fasta_indices, device=self.device), torch.tensor(Y, dtype=torch.float, device=self.device)
    
    def collate_fn(self, batch):
        smiles, fasta, Y = zip(*batch)
        smiles = torch.stack(smiles, dim=0)
        fasta = torch.stack(fasta, dim=0)
        Y = torch.stack(Y, dim=0)
        return smiles, fasta, Y