"""
BridgeDTA standalone data processing module.
Integrates protein sequence encoding / MACCS fingerprint logic / molecular graph construction logic.
"""
import os
import json
import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch_geometric.data import Data, Batch

# ============================================================
# Constants
# ============================================================

# Protein amino acid character -> integer mapping (25 types)
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25
}
CHARPROTLEN = 25

# Atom type allowable set (44 types, including Unknown)
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
    'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
]

# ============================================================
# Helper functions -- Protein sequence encoding
# ============================================================

def encode_protein_sequence(fasta, max_len, char_set=CHARPROTSET):
    """Encode a FASTA protein sequence into a fixed-length integer tensor."""
    encoded = torch.zeros(max_len, dtype=torch.long)
    for i, amino_acid in enumerate(fasta[:max_len]):
        encoded[i] = char_set.get(amino_acid, 0)
    return encoded


# ============================================================
# Helper functions -- MACCS molecular fingerprint
# ============================================================

def compute_maccs_fingerprint(smiles):
    """Convert a SMILES string to a 166-dimensional MACCS molecular fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([int(bit) for bit in fingerprint.ToBitString()[1:]])


# ============================================================
# Helper functions -- Molecular graph construction
# ============================================================

def _one_of_k_encoding(value, allowable_set):
    """Strict one-hot encoding; value must be in the set."""
    if value not in allowable_set:
        raise ValueError(f"Input {value} not in allowable set {allowable_set}")
    return [value == item for item in allowable_set]


def _one_of_k_encoding_unk(value, allowable_set):
    """Tolerant one-hot encoding; unknown values are mapped to the last element."""
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def compute_atom_features(atom):
    """
    Encode an RDKit Atom object into a 78-dimensional feature vector.
    Composition: atom type (44d) + degree (11d) + num H atoms (11d) + implicit valence (11d) + aromaticity (1d)
    """
    return np.array(
        _one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + _one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + _one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a molecular graph representation.
    Returns: (num_atoms, node_features, edge_index)
    """
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    node_features = []
    for atom in mol.GetAtoms():
        feature = compute_atom_features(atom)
        feature_sum = feature.sum()
        if feature_sum > 0:
            feature = feature / feature_sum
        node_features.append(feature)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    directed_graph = nx.Graph(edges).to_directed()
    edge_index = [[source, target] for source, target in directed_graph.edges]

    return num_atoms, node_features, edge_index


# ============================================================
# BridgeDTADataset
# ============================================================

class BridgeDTADataset(Dataset):
    """
    BridgeDTA standalone dataset, loads and processes directly from raw data files:
    - Protein FASTA -> integer encoding (for FASTAModel)
    - SMILES -> molecular graph PyG Data (for DrugGNN)
    - SMILES -> KPGT pretrained features (loaded from external .npz)
    - Protein index -> ESM pretrained features (loaded from external .npz)
    """
    
    def __init__(self, root, dataset, mode, cold_start,
                 kpgt_features, esm_features,
                 max_fasta_len=1000, device='cpu'):
        """
        Args:
            root: Data root directory
            dataset: Dataset name (davis / kiba / Bind)
            mode: 'train' or 'test'
            cold_start: Cross-validation fold number ('' or '0'-'4')
            kpgt_features: dict, SMILES -> KPGT features (numpy array)
            esm_features: dict, protein index -> ESM features (numpy array)
            max_fasta_len: Maximum encoding length for protein sequences
            device: Compute device
        """
        super().__init__()
        assert mode in ['train', 'test']
        self.device = torch.device(device)
        self.max_fasta_len = max_fasta_len
        self.kpgt_features = kpgt_features
        self.esm_features = esm_features

        # Process cold_start suffix
        cold_suffix = f'_{cold_start}' if cold_start else ''

        # Load indices and labels
        data_file = f'{root}/{dataset}/processed/{mode}{cold_suffix}.txt'
        raw_data = torch.tensor(np.loadtxt(data_file))
        self.compound_idx = raw_data[:, 0].long()
        self.protein_idx = raw_data[:, 1].long()
        self.labels = raw_data[:, 2]

        # Load SMILES list
        with open(f'{root}/{dataset}/ligands_can.txt', 'r') as f:
            self.smiles = list(json.load(f).values())

        # Load protein sequence list
        with open(f'{root}/{dataset}/proteins.txt', 'r') as f:
            self.fasta = list(json.load(f).values())

        # Pre-build molecular graph cache (avoid redundant computation)
        self.smile_graph_cache = {}
        for smiles_str in self.smiles:
            if smiles_str not in self.smile_graph_cache:
                self.smile_graph_cache[smiles_str] = smiles_to_graph(smiles_str)

    def __len__(self):
        return len(self.compound_idx)

    def __getitem__(self, idx):
        compound_id = self.compound_idx[idx]
        protein_id = self.protein_idx[idx].item()
        smiles_str = self.smiles[compound_id]
        fasta_str = self.fasta[protein_id]
        label = self.labels[idx]

        # 1) Protein sequence -> integer encoding
        fasta_encoded = encode_protein_sequence(fasta_str, self.max_fasta_len)

        # 2) Molecular graph -> PyG Data
        num_atoms, node_features, edge_index = self.smile_graph_cache[smiles_str]
        graph_data = Data(
            x=torch.tensor(np.array(node_features, dtype=np.float32)),
            edge_index=torch.LongTensor(edge_index).t().contiguous() if edge_index else torch.zeros(2, 0, dtype=torch.long),
            y=torch.FloatTensor([label]),
            c_size=torch.LongTensor([num_atoms])
        )

        # 3) KPGT pretrained features -- supports two formats:
        #    a) graph-level: 1D vector [kpgt_dim]
        #    b) node-level:  2D matrix [n_atoms, kpgt_dim]
        kpgt_feat = torch.from_numpy(np.asarray(self.kpgt_features[smiles_str])).float()

        # 4) ESM pretrained features (truncated to max_fasta_len)
        esm_feat = torch.from_numpy(
            self.esm_features[protein_id].squeeze()[:self.max_fasta_len]
        ).float()

        # 5) Label
        label_tensor = torch.tensor(label).float()

        return fasta_encoded, graph_data, kpgt_feat, esm_feat, label_tensor


# ============================================================
# collate_fn & DataLoader
# ============================================================

def bridge_collate_fn(batch):
    """BridgeDTA custom batch collation function.

    KPGT supports two feature formats:
    - graph-level: each molecule is a [kpgt_dim] vector, directly stacked to [B, kpgt_dim], kpgt_mask=None
    - node-level: each molecule is a [n_atoms, kpgt_dim] matrix, padded to [B, max_atoms, kpgt_dim] + mask
    """
    fasta_list, graph_list, kpgt_list, esm_list, label_list = zip(*batch)

    # Protein encoding directly stacked (fixed length)
    fasta_batch = torch.stack(fasta_list)

    # Molecular graphs assembled with PyG Batch
    graph_batch = Batch.from_data_list(graph_list)

    # KPGT features -- distinguish graph-level from node-level
    if kpgt_list[0].dim() == 1:
        kpgt_batch = torch.stack(kpgt_list)            # [B, kpgt_dim]
        kpgt_masks = None
    else:
        kpgt_lengths = [k.size(0) for k in kpgt_list]
        kpgt_batch = pad_sequence(kpgt_list, batch_first=True, padding_value=0)  # [B, max_atoms, kpgt_dim]
        kpgt_masks = torch.zeros(kpgt_batch.shape[0], kpgt_batch.shape[1], dtype=torch.bool)
        for i, length in enumerate(kpgt_lengths):
            kpgt_masks[i, :length] = True

    # ESM features need padding (variable-length sequences)
    esm_lengths = [esm.size(0) for esm in esm_list]
    esm_padded = pad_sequence(esm_list, batch_first=True, padding_value=0)

    # ESM attention mask
    esm_masks = torch.zeros(esm_padded.shape[0], esm_padded.shape[1], dtype=torch.bool)
    for i, length in enumerate(esm_lengths):
        esm_masks[i, :length] = True

    # Label stack
    label_batch = torch.stack(label_list)

    return fasta_batch, graph_batch, kpgt_batch, kpgt_masks, esm_padded, esm_masks, label_batch


def get_bridge_dataloader(dataset, batch_size, shuffle=False):
    """Create a BridgeDTA-specific DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=bridge_collate_fn
    )


# ============================================================
# forward_step
# ============================================================

def bridge_forward_step(batch, model):
    """
    BridgeDTA-specific forward step called by Trainer.
    """
    fasta_batch, graph_batch, kpgt_batch, kpgt_masks, esm_padded, esm_masks, label_batch = batch

    device = next(model.parameters()).device
    graph_batch = graph_batch.to(device)
    fasta_batch = fasta_batch.to(device)
    kpgt_batch = kpgt_batch.to(device)
    if kpgt_masks is not None:
        kpgt_masks = kpgt_masks.to(device)
    esm_padded = esm_padded.to(device)
    esm_masks = esm_masks.to(device)
    label_batch = label_batch.to(device)

    output, ci_loss = model(
        graph_batch, fasta_batch, kpgt_batch, esm_padded, esm_masks,
        kpgt_mask=kpgt_masks,
        true_values=label_batch
    )

    return output, label_batch, ci_loss
