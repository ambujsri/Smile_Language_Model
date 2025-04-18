# 🧪 SMILES Language Modeling Project

This repository contains code and utilities for building and evaluating a **SMILES-based Language Model** for molecular generation and analysis using deep learning. The model is trained using PyTorch and leverages RDKit for chemical informatics.

---

## 📦 Features

- SMILES tokenization and vocabulary handling
- Neural network models built with PyTorch
- Evaluation metrics including F1 score and ROC AUC
- Visualization of training and performance curves
- End-to-end pipeline: data preprocessing → training → testing

---

## 🏗 Dependencies

This project requires the following Python packages:

- `torch` (PyTorch for neural network training)
- `rdkit` (molecule handling, SMILES parsing)
- `scikit-learn` (evaluation metrics)
- `numpy`, `pandas` (data manipulation)
- `matplotlib` (visualization)

---

## ⚙️ Installation

### 🔁 Recommended via Conda (for RDKit)

```bash
# Create a new environment
conda create -n smiles-lm-env python=3.10
conda activate smiles-lm-env

# Install RDKit
conda install -c rdkit rdkit

# Install remaining packages
pip install -r requirements.txt
