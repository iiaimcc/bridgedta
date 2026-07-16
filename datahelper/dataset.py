import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class DTADataset(Dataset, ABC):
    def __init__(self, root='data', dataset='davis', mode='train', cold_start='', device='cpu'):
        assert mode in ['train','test']
        self.mode = mode
        self.device = torch.device(device)
        assert cold_start in ['','0','1','2','3','4']
        if cold_start:
            cold_start = '_'+cold_start
            
        # 加载基本数据
        if mode =='train':
            data = torch.tensor(np.loadtxt(f'{root}/{dataset}/processed/train{cold_start}.txt'))
        else:
            data = torch.tensor(np.loadtxt(f'{root}/{dataset}/processed/test{cold_start}.txt'))
        data = data.to(device)
        
        # 加载smiles和fasta序列
        self.compound_idx, self.protein_idx, self.Y = data[:,0].long(), data[:,1].long(), data[:,2]
        
        # 加载原始的smiles序列
        self.smiles = []
        with open(f'{root}/{dataset}/ligands_can.txt', 'r') as f:
            smiles_dict = json.load(f)
            # 将字典值（序列）转换为列表
            self.smiles = list(smiles_dict.values())

                
        # 加载蛋白质序列（JSON格式）
        self.fasta = []
        with open(f'{root}/{dataset}/proteins.txt', 'r') as f:
            protein_dict = json.load(f)
            # 将字典值（序列）转换为列表
            self.fasta = list(protein_dict.values())

    def __len__(self):
        return len(self.compound_idx)

    def __getitem__(self, idx):
        smiles = self.smiles[self.compound_idx[idx]]
        fasta = self.fasta[self.protein_idx[idx]]
        Y = self.Y[idx]
        return smiles, fasta, Y

    @abstractmethod
    def collate_fn(self, batch):
        pass
    