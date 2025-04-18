#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:15:35 2025

@author: ambujsrivastava
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

import matplotlib.pyplot as plt

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, names_list):
        self.smiles_list = smiles_list
        self.names_list = names_list

    def __len__(self):
        return len(self.smiles_list)
        
    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        fingerprint = np.array(fingerprint)
        return torch.tensor(fingerprint, dtype=torch.float32), self.names_list[idx]
        
    def get_smiles_from_names(self, names_to_find):
        smiles_found = []
        for name in names_to_find:
            try:
                idx = self.names_list.index(name)
                smiles_found.append(self.smiles_list[idx])
            except ValueError:
                print(f"No SMILES found for compound name: {name}")
        return smiles_found
    
class OverfitVisualizer:
    def __init__(self):
        self.ctr1 = 1
        self.ctr2 = 1
        

    def plot_loss_curves(self, train_losses, val_losses):
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'r', label='Training loss')
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_fold'+str(self.ctr1)+'.png', dpi=300)
        self.ctr1 +=1
        plt.show()

    def plot_accuracy_curves(self, train_accuracies, val_accuracies):
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy_fold'+str(self.ctr2)+'.png', dpi=300)
        self.ctr2 +=1
        plt.show()

    def visualize_overfitting(self, train_losses, val_losses, train_accuracies, val_accuracies):
        self.plot_loss_curves(train_losses, val_losses)
        self.plot_accuracy_curves(train_accuracies, val_accuracies)
        
        
class PerformanceVisualizer:
    def __init__(self):
        pass
        
    def plot_losses(self,train_losses_list, val_losses_list):
        num_epochs = len(train_losses_list[0])
        epochs = range(1, num_epochs+1)
        plt.figure(figsize=(10, 6))
        for fold in range(len(train_losses_list)):
            plt.plot(epochs, train_losses_list[fold], label=f'Training Loss (Fold {fold+1})')
            plt.plot(epochs, val_losses_list[fold], label=f'Validation Loss (Fold {fold+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('training_validation_loss_plot.png')
        plt.show()
    def plot_accuracies(self,train_accuracies_list, val_accuracies_list):
        num_epochs = len(train_accuracies_list[0])
        epochs = range(1, num_epochs+1)
        plt.figure(figsize=(10, 6))
        for fold in range(len(train_accuracies_list)):
            plt.plot(epochs, train_accuracies_list[fold], label=f'Training accuracy (Fold {fold+1})')
            plt.plot(epochs, val_accuracies_list[fold], label=f'Validation accuracy (Fold {fold+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.legend()
        plt.savefig('training_validation_accuracy_plot.png')
        plt.show() 
    def plot_roc_curve(self, fpr_list, tpr_list, auc_list):
        num_epochs = len(fpr_list[0])
        epochs = range(1, num_epochs+1)
        plt.figure()
        for fold in range(len(fpr_list)):
            plt.plot(fpr_list[fold], tpr_list[fold], label=f'ROC Curve (AUC (Fold {fold+1}) = {auc_list[fold]:.2f})')
        #plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {self.auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('validation_roc_plot.png')
        plt.show()
    def visualize_performance(self, train_losses_list, val_losses_list, train_accuracies_list, val_accuracies_list,fpr_list,trp_list,auc_list):
        self.plot_losses(train_losses_list, val_losses_list)
        self.plot_accuracies(train_accuracies_list, val_accuracies_list)
        self.plot_roc_curve(fpr_list,trp_list,auc_list)
    