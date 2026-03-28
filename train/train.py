import torch
import torch.nn as nn
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import pickle
import numpy as np
from model.RNABP import HybridRNABindingSiteModel 
import random
import os


from feature_extraction.datasetnew import RNAGraphDatasetNew

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    model.eval()
    probs_all, labels_all = [], []

    for data in loader:
        emb   = data.x.to(device)               # (1, L, 640)       # (L, F)
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

    threshold = 0.5
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

def train(model, train_loader, val_loader, device,
          epochs=30, lr=1e-3, weight_decay=1e-2,
          use_pos_weight=False, save_path='best_model.pt', patience = 100):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_pos, num_neg = 0, 0
    for data in train_loader:  
        y = data.y.to(device) 
        m = (y >= 0)            
        num_pos += (y[m] == 1).sum().item()
        num_neg += (y[m] == 0).sum().item()
    pos_weight = torch.tensor([max(num_neg / max(1, num_pos), 3)], device=device)
    print(f"numpost: {num_pos}")
    print(f"num_neg: {num_neg}")
    print("posweight: ")
    print(pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_mcc, best_th = -1.0, 0.5
    patience_counter = 0
    best_aupr = 0

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
      

        for data in train_loader:
            emb   = data.x.to(device)               #torch.Size([1267, 640])
            edge  = data.edge_index.to(device)              # torch.Size([2, 10136])
            y     = data.y.to(device)               # torch.Size([1267])
            batch = data.batch.to(device)

            ss_emb = data.ss_emb.to(device)


            logits = model(emb, ss_emb, edge, batch) 
            logits = logits.squeeze(0) if logits.dim() == 2 else logits
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, threshold=None)

            

        if val_metrics['mcc'] > best_mcc:
    
            mcc, best_th = val_metrics['mcc'], val_metrics['threshold']
            best_mcc = mcc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mcc': mcc,
                'threshold': best_th
            }, save_path)        
        else:
            patience_counter += 1
            print(f"No improvement in MCC for {patience_counter} epochs")

        print(f"Epoch {epoch:03d} | train_loss {running/len(train_loader):.4f} "
              f"| val_AUPR {val_metrics['aupr']:.4f} | val_MCC {val_metrics['mcc']:.4f} | val_AUC {val_metrics['auc']:.4f} "
              f"| thres {val_metrics['threshold']:.2f}")

    print(f"\nDone. Best mcc={best_mcc:.4f} @ th={best_th:.2f} (saved -> {save_path})")



if __name__ == "__main__":
    
    set_seed(42)
    full_dataset = RNAGraphDatasetNew(
        root='data/processed/TR60NEW'
    )
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    print(f"Train size: {len(train_dataset)} ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"Test size:  {len(val_dataset)} ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(full_dataset[0].x.shape[1])
    print(full_dataset[0].ss_emb.shape[1])

    model = HybridRNABindingSiteModel(rna_dim=full_dataset[0].x.shape[1], ss_dim = full_dataset[0].ss_emb.shape[1], hidden=86, dropout=0.6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, train_loader, val_loader, device,
          epochs=100, lr=1e-4, weight_decay=1e-4, save_path='best_model.pt')
