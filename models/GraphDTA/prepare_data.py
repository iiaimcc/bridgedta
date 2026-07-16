import os
import sys
sys.path.append("/home/zh/workspace/BridgeDTA")

import torch
import networkx as nx
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data as DATA
from torch_geometric.data import InMemoryDataset
from datahelper.dataset import DTADataset
import torch_geometric.data

class GraphDTADataset(DTADataset, InMemoryDataset):
    def __init__(self, root='data', dataset='davis', mode='train', cold_start='', device='cpu'):
        self.dataset = dataset
        self.mode = mode
        self.cold_start = cold_start
        self.device = device
        self.max_seq_len = 1000
        self.seq_dict = {v:(i+1) for i,v in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")}
        
        # 先初始化DTADataset以获取数据
        DTADataset.__init__(self, root, dataset, mode, cold_start, device)
        
        # Create smile_graph dictionary for all compounds
        self.smile_graph = {}
        for smile in self.smiles:
            if smile not in self.smile_graph:
                self.smile_graph[smile] = smile_to_graph(smile)
        
        # 初始化InMemoryDataset并加载数据
        InMemoryDataset.__init__(self, root, None, None)
        
        # 检查是否需要处理数据
        processed_file = self.processed_paths[0]
        if not os.path.isfile(processed_file):
            print(f'Pre-processed data {processed_file} not found, doing pre-processing...')
            self.process()
        
        # 加载处理好的数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.mode}_{self.cold_start}.pt']

    @property
    def processed_paths(self):
        model_processed_dir = os.path.join(os.path.dirname(__file__), self.dataset, 'processed')
        os.makedirs(model_processed_dir, exist_ok=True)
        return [os.path.join(model_processed_dir, name) 
                for name in self.processed_file_names]

    def seq_cat(self, prot):
        """Convert protein sequence to tensor using the original method"""
        x = torch.zeros(self.max_seq_len)
        for i, ch in enumerate(prot[:self.max_seq_len]): 
            x[i] = self.seq_dict.get(ch, 0)
        return x

    def process(self):
        """Convert SMILES to graphs and process protein sequences"""
        data_list = []
        
        for idx in range(len(self)):
            smiles = self.smiles[self.compound_idx[idx]]
            protein = self.fasta[self.protein_idx[idx]]  # 直接获取蛋白质序列
            label = self.Y[idx]
            
            # Get graph data
            c_size, features, edge_index = self.smile_graph[smiles]
            
            # Process protein sequence
            target = self.seq_cat(protein)
            
            # Create PyG data object
            data = DATA(
                x=torch.from_numpy(np.array(features, dtype=np.float32)),  # 先转换为numpy array
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([label]),
                target=target.long().view(1, -1),  # 改变维度为 [1, seq_length]
                c_size=torch.LongTensor([c_size])
            )
            
            # Add batch information
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            
            data_list.append(data)

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def collate_fn(self, data_list):
        """
        Custom collate function for GraphDTA data
        Args:
            data_list: List of PyG Data objects
        Returns:
            Batched data
        """
        # 首先获取所有的target数据并调整维度
        targets = torch.stack([data.target for data in data_list])  # [batch_size, 1, seq_length]
        targets = targets.squeeze(1)  # [batch_size, seq_length]
        
        # 使用PyG的默认batch方法处理其他数据
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        
        # 确保target数据被正确添加到batch中
        batch.target = targets
        
        return batch

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        data = super(InMemoryDataset, self).__getitem__(idx)
        return data.to(self.device)

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index
