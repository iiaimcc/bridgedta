import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.data
from datahelper.dataset import DTADataset

class MultiKD_DTADataset(DTADataset, InMemoryDataset):
    def __init__(self, root='data', dataset='davis', mode='train', cold_start='', device='cpu'):
        self.dataset = dataset
        self.mode = mode
        self.device = device
        self.cold_start = cold_start
        
        # First initialize DTADataset to get data
        DTADataset.__init__(self, root, dataset, mode, cold_start, device)
        
        # Create smile_graph dictionary for all compounds
        self.smile_graph = {}
        for smile in self.smiles:
            if smile not in self.smile_graph:
                self.smile_graph[smile] = smile_to_graph(smile)
        
        # Initialize InMemoryDataset and load data
        InMemoryDataset.__init__(self, root, None, None)
        
        # Check if data needs to be processed
        processed_file = self.processed_paths[0]
        if not os.path.isfile(processed_file):
            print(f'Pre-processed data {processed_file} not found, doing pre-processing...')
            self.process()
        
        # Load processed data
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


    def process(self):
        """Convert SMILES to graphs and process protein sequences"""
        data_list = []
        
        for idx in tqdm(range(len(self)), desc='Processing data'):
            smiles = self.smiles[self.compound_idx[idx]]
            protein = self.fasta[self.protein_idx[idx]]
            label = self.Y[idx]
            
            # Get graph data
            c_size, features, edge_index = self.smile_graph[smiles]
            
            # Process protein sequence
            target = None # Processed in get_item
            
            # Create PyG data object
            data = Data(
                x=torch.from_numpy(np.array(features, dtype=np.float32)),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([label]),
                c_size=torch.LongTensor([c_size])
            )

            data.pid = self.protein_idx[idx]

            # Add batch information
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            
            data_list.append(data)

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

    def collate_fn(self, data_list):
        pass

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
