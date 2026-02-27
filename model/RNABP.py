import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, GATConv

class HybridRNABindingSiteModel(nn.Module):

    def __init__(self, rna_dim=645, ss_dim = 6, hidden=128, dropout=0.5):
        super().__init__()
        self.rna_ln  = nn.LayerNorm(rna_dim)

        self.fuse = nn.Sequential(
            nn.Linear( rna_dim + ss_dim , hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


        self.gcn1 = GCNConv(hidden, hidden)
        self.gn1  = GraphNorm(hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gn2  = GraphNorm(hidden)
        self.gcn3 = GATConv(hidden, hidden)
        self.gn3  = GraphNorm(hidden)

        self.drop = nn.Dropout(dropout)

        self.gate = nn.Linear(2 * hidden, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        self.lnn = nn.LayerNorm(hidden)


    def forward(self, rna_embeddings, ss_emb, edge_index, batch):       
        h = self.fuse(torch.cat([rna_embeddings, ss_emb], dim=-1))  # (N, H)

        h_res = h
        h = self.gcn3(h, edge_index)
        h = self.gn3(h, batch)
        h = F.relu(h)
        h = self.drop(h)
        h1 = h

        h = self.gcn3(h, edge_index)
        h = self.gn3(h, batch)
        h = F.relu(h)
        h = self.drop(h)
        h = h + h_res
        h2 = h

        z = torch.cat([h1, h2], dim=-1)        
        gate = torch.sigmoid(self.gate(z))         
        h = gate * h1 + (1 - gate) *  h2  
        logits = self.head(h)
        logits = self.fc(logits).squeeze(-1)

        return logits
