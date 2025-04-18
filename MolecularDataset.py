#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:11:27 2025

@author: ambujsrivastava
"""
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, names_list, targets):
        self.smiles_list = smiles_list
        self.names = names_list  # Assuming names correspond to compound names
        self.targets = targets

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        # fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        # fingerprint = np.array(fingerprint)
        
        gen = GetMorganGenerator(radius=4, fpSize=1024)
        fp = gen.GetFingerprint(mol)
        arr = fp.ToBitString()
        return torch.tensor(fp, dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32), self.names[idx]
    