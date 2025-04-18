#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:20:23 2025

@author: ambujsrivastava
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from model_utils import OverfitVisualizer
from model_utils import PerformanceVisualizer
from MolecularDataset import MolecularDataset
from Model import NeuralNetwork
from ModelTrainer import ModelTrainer
import numpy as np


# Set seeds to ensure reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


dataset = pd.read_csv('dataset/dataset_EC50.csv')
# added_scores = pd.read_csv('./top36_dataset_filtered.csv')
# combined_scores = pd.concat([decoy_scores, added_scores])
dataset_shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
train_set, blind_set = train_test_split(dataset_shuffled, test_size=0.2, random_state=42)


# Separate features and target variables
smiles_list = train_set['SMILES'].tolist()
X_train = [Chem.CanonSmiles(smiles) if Chem.MolFromSmiles(smiles) is not None else None for smiles in smiles_list]
y_train = train_set['Activity'].tolist()
names_list = train_set['Title'].tolist()

blind_smiles = blind_set['SMILES'].tolist()
X_blind = [Chem.CanonSmiles(smiles) if Chem.MolFromSmiles(smiles) is not None else None for smiles in blind_smiles]
y_blind = blind_set['Activity'].tolist()
names_blind = blind_set['Title'].tolist()

blind_df = pd.DataFrame({
    'SMILES': X_blind,
    'Title': names_blind,
    'Activity': y_blind
})

blind_df.to_csv('blind_set.csv', index=False)



num_epochs = 50
batch_size1 = 16
n_splits=5
ctr = 1
# Create dataset and dataloader
dataset = MolecularDataset(X_train, names_list, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size1, shuffle=True)

# Define cross-validation
kf = KFold(n_splits, shuffle=True)
overfit_visualizer = OverfitVisualizer()
performance_visualizer = PerformanceVisualizer()
train_losses_list = []
val_losses_list = []
train_accuracies_list = []
val_accuracies_list = []
fpr_list = []
tpr_list = []
auc_list = []
for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    print(f'Fold {fold+1}')
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size1, shuffle=True)

    model = NeuralNetwork(input_size=1024, hidden_size=10)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    trainer = ModelTrainer(model, criterion, optimizer, visualizer=overfit_visualizer)

    train_losses, val_losses, train_accuracies, val_accuracies, fpr, tpr, auc1, f1 = trainer.train(train_dataloader, val_dataloader,num_epochs)
    train_losses_list.append(train_losses)
    val_losses_list.append(val_losses)
    train_accuracies_list.append(train_accuracies)
    val_accuracies_list.append(val_accuracies)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc1)
    

performance_visualizer.visualize_performance(train_losses_list, val_losses_list, train_accuracies_list, val_accuracies_list,fpr_list,tpr_list,auc_list)

blind_dataset = MolecularDataset(X_blind, names_blind, y_blind)
blind_dataloader = DataLoader(blind_dataset, batch_size=batch_size1, shuffle=False)
loss, accuracy, fpr, tpr, auc1, f1 = trainer.evaluate(blind_dataloader)
# print (loss, accuracy, fpr, tpr, auc1, f1)
print(f'Blind Loss: {loss:.4f}, Blind Accuracy: {accuracy:.4f}, '
      f'F1 Score: {f1:.4f}')

