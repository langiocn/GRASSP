import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    matthews_corrcoef, roc_auc_score, accuracy_score,
    confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_extraction.datasetnew import RNAGraphDatasetNew
from model.RNABP import HybridRNABindingSiteModel


def find_best_threshold(y_true, y_prob, metric='f1'):
    ts = np.linspace(0, 1, 1001)
    best_t, best_score = 0.5, -1
    for t in ts:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError("metric phải là 'f1' hoặc 'mcc'")
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score



@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    model.eval()
    probs_all, labels_all = [], []

    for data in loader:
        emb   = data.x.to(device)               # (1, L, 640)   # (L, F)
        edge  = data.edge_index.to(device)              # (2, E)
        y     = data.y.to(device) 
        batch = data.batch.to(device)

        ss_emb = data.ss_emb.to(device)



        logits = model(emb, ss_emb, edge, batch) 
    
        logits = logits.squeeze(0) if logits.dim() == 2 else logits
        probs  = torch.sigmoid(logits)

        probs_all.append(probs.cpu().numpy())
        labels_all.append(y.cpu().numpy())

    probs_all  = np.concatenate(probs_all,  axis=0)
    labels_all = np.concatenate(labels_all, axis=0).astype(int)


    threshold = 0.7
    preds = (probs_all > threshold).astype(int)

    metrics = {
        'accuracy' : accuracy_score(labels_all, preds),
        'precision': precision_score(labels_all, preds, zero_division=0),
        'recall'   : recall_score(labels_all, preds, zero_division=0),
        'f1'       : f1_score(labels_all, preds, zero_division=0),
        'mcc'      : matthews_corrcoef(labels_all, preds),
        'auc'      : roc_auc_score(labels_all, probs_all),
        'aupr'     : average_precision_score(labels_all, probs_all),
        'threshold': float(threshold)
    }
    return metrics

if __name__ == "__main__":
    device = 'cpu'
    
    # ========== 1. Load test dataset ==========
    print("Loading test dataset...")
    test_dataset = RNAGraphDatasetNew(root='data/processed/TE18NEW')
    num_rna = len(test_dataset)
    num_nucleotides = 0
    pos_labels = 0
    neg_labels = 0

    for data in test_dataset:
        # số nucleotide = số node
        N = data.x.shape[0]
        num_nucleotides += N

        # nhãn
        y = data.y.view(-1)
        pos_labels += int((y == 1).sum())
        neg_labels += int((y == 0).sum())

    print(f"RNA chains              : {num_rna}")
    print(f"Nucleotides             : {num_nucleotides}")
    print(f"Binding residues (pos)  : {pos_labels}")
    print(f"Non-binding residues    : {neg_labels}")
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    
    model = HybridRNABindingSiteModel(rna_dim=test_dataset[0].x.shape[1], ss_dim = test_dataset[0].ss_emb.shape[1], hidden=86, dropout=0.6)
    
    # Load checkpoint
    ckpt = torch.load('checkpoints/TR60_SEED39.pt', map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    saved_th = ckpt.get("threshold", 0.5)
    print(f"✓ Loaded model checkpoint")
    print(f"  Threshold: {saved_th:.4f}")
    
    # ========== 4. Evaluate ==========
    print("\nEvaluating on test set...")
    metrics = evaluate(
        model, 
        test_loader, 
        device, 
        saved_th,
    )
    
    # ========== 5. Print results ==========
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"AUPR:      {metrics['aupr']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print("="*60)
    
  