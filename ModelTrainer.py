#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:18:33 2025

@author: ambujsrivastava
"""

import torch
from sklearn.metrics import roc_curve, auc, f1_score

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, visualizer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.visualizer = visualizer

    def train(self, train_dataloader, val_dataloader, num_epochs=10):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            self.model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            for fingerprints, targets, _ in train_dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(fingerprints.float())
                loss = self.criterion(outputs, targets.float().view(-1, 1))
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * fingerprints.size(0)
                predicted = torch.round(outputs)
                correct_train += (predicted == targets.view(-1, 1)).sum().item()
                total_train += targets.size(0)
            train_loss = running_train_loss / len(train_dataloader.dataset)
            train_acc = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, val_acc, fpr, tpr, auc1, f1 = self.evaluate(val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            # fpr, tpr, auc1, f1 = self.calculate_roc(val_dataloader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, '
                  f'F1 Score: {f1:.4f}')
        
        if self.visualizer:
            self.visualizer.visualize_overfitting(train_losses, val_losses, train_accuracies, val_accuracies)

        
        return train_losses, val_losses, train_accuracies, val_accuracies, fpr, tpr, auc1, f1
        
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for fingerprints, targets, _ in dataloader:
                outputs = self.model(fingerprints.float())
                loss = self.criterion(outputs, targets.float().view(-1, 1))
                running_loss += loss.item() * fingerprints.size(0)
                
                predicted = torch.round(outputs)
                # print (targets,predicted)
                correct += (predicted == targets.view(-1, 1)).sum().item()
                total += targets.size(0)
        loss = running_loss / len(dataloader.dataset)
        accuracy = correct / total
        fpr, tpr, auc1, f1 = self.calculate_roc(dataloader)
        return loss, accuracy, fpr, tpr, auc1, f1
        
    def calculate_roc(self, val_loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                outputs = self.model(inputs)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        # print (all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        binary_preds = [(x >= 0.5).astype(int) for x in all_predictions]
        f1 = f1_score(all_labels, binary_preds)
        # f1 = f1_score(all_labels, all_predictions)
        fpr = fpr
        tpr = tpr
        auc1 = auc(fpr, tpr)
        return fpr, tpr, auc1, f1