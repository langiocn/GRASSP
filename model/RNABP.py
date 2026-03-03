import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm


class HybridRNABindingSiteModel(nn.Module):
    def __init__(self, rna_dim=645, ss_dim=6, hidden=128, dropout=0.5):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Linear(rna_dim + ss_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gnn = GATConv(hidden, hidden)
        self.gn = GraphNorm(hidden)
        self.drop = nn.Dropout(dropout)

        self.gate = nn.Linear(2 * hidden, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, rna_embeddings, ss_emb, edge_index, batch):
        h = self.fuse(torch.cat([rna_embeddings, ss_emb], dim=-1))

        h_res = h

        h = self.gnn(h, edge_index)
        h = self.gn(h, batch)
        h = F.relu(h)
        h = self.drop(h)
        h1 = h

        h = self.gnn(h, edge_index)
        h = self.gn(h, batch)
        h = F.relu(h)
        h = self.drop(h)
        h2 = h + h_res

        z = torch.cat([h1, h2], dim=-1)
        gate = torch.sigmoid(self.gate(z))
        h = gate * h1 + (1.0 - gate) * h2

        logits = self.fc(self.head(h)).squeeze(-1)
        return logits