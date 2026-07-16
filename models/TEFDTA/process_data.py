import sys
sys.path.append("/home/zh/workspace/BridgeDTA")

from torch.utils.data import Dataset
import torch
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from datahelper.dataset import DTADataset

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
            "U": 19, "T": 20, "W": 21, 
            "V": 22, "Y": 23, "X": 24, 
            "Z": 25 }

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

MACCSLEN = 2

def label_chars(chars, max_len, char_set):
    X = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(chars[:max_len]):
        X[i] = char_set[ch]
    return X

def smiles_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([int(_) for _ in fp.ToBitString()[1:]])

class TEFDTADataset(DTADataset):
    def __init__(self, root, dataset, mode, cold_start, device, max_smiles_len=100, max_fasta_len=1000):
        super().__init__(root, dataset, mode, cold_start, device)
        self.max_smiles_len = max_smiles_len
        self.max_fasta_len = max_fasta_len

    def __getitem__(self, idx):
        smiles, fasta, Y = super().__getitem__(idx)
        return (smiles_fingerprint(smiles).to(self.device),
                label_chars(fasta, self.max_fasta_len, CHARPROTSET).to(self.device),
                torch.tensor(Y).float().to(self.device))

    # add for pred
    def get_ori_data(self, idx):
        smiles, fasta, Y = super().__getitem__(idx)
        return smiles, fasta, Y
    
    def collate_fn(self, batch):
        pass