import math
import argparse
from typing import Optional, Tuple
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool, GINEConv
from torch_geometric.nn import radius_graph, knn_graph

class SAScorer(nn.Module):
    """Graph neural network to predict quantilenormalized SA score (in [0,1]) for a linker-only graph.
    Expected PyG Data per graph:
      - x: [N, F] one-hot atom types (e.g., 9 types)
      - pos: [N, 3] (optional) 3D coordinates
      - edge_index: [2, E] (optional) if missing, will be built from pos
      - y: scalar in [0,1] when training
    """
    def __init__(
        self,
        atom_feat_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        use_distance: bool = True,
        rbf_k: int = 32,
        build_edges: str = "radius",   # "radius" | "knn" | "none"
        radius: float = 2.0,
        k: Optional[int] = None,
        pool: str = "mean",             # "mean" | "add" | "max"
        use_quantile_norm: bool = True,
        normalizer_path: Optional[str] = None
        ):
        super().__init__()
        self.use_distance = use_distance
        self.build_edges = build_edges
        self.radius = radius
        self.k = k
        self.pool = pool
        self.use_quantile_norm = use_quantile_norm

        in_dim = atom_feat_dim
        self.node_in = nn.Linear(in_dim, hidden_dim)

        if self.use_distance:
            self.rbf = RBF(num_kernels=rbf_k, r_min=0.0, r_max=5.0)
            edge_in = rbf_k
        else:
            edge_in = 1
        self.edge_mlp = MLP([edge_in, hidden_dim, hidden_dim], dropout=dropout)

        convs = []
        for _ in range(num_layers):
            nn_edge = MLP([hidden_dim, hidden_dim, hidden_dim], dropout=dropout)
            convs.append(GINEConv(nn_edge))
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            # nn.Sigmoid()  # Predict SA in [0,1]
        )

        # Load normalizer
        self.normalizer = None
        if normalizer_path is not None:
            self.load_normalizer(normalizer_path)
    def load_normalizer(self, normalizer_path: str):
        normalizer_path = Path(normalizer_path)
        if normalizer_path.is_dir():
            pkl_files = list(normalizer_path.glob("*.pkl"))
            if len(pkl_files) == 0:
                raise FileNotFoundError(f"No .pkl file found in {normalizer_path}")
            normalizer_path = pkl_files[0]
        self.normalizer = joblib.load(normalizer_path)
        print(f"Loaded quantile normalizer from {normalizer_path}")
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.use_quantile_norm = True

    @staticmethod
    def _center_pos(pos: torch.Tensor) -> torch.Tensor:
        # subtract centroid to get translation invariance
        centroid = pos.mean(dim=0, keepdim=True)
        return pos - centroid

    def _ensure_edges(self, data: 'Data') -> 'Data':
        if getattr(data, "edge_index", None) is not None:
            return data
        if getattr(data, "pos", None) is None:
            raise ValueError("No edge_index in Data and pos is None; cannot build edges.")
        pos = data.pos
        if self.build_edges == "radius":
            edge_index = radius_graph(pos, r=self.radius, loop=False)
        elif self.build_edges == "knn":
            if self.k is None:
                raise ValueError("build_edges='knn' requires k to be set.")
            edge_index = knn_graph(pos, k=self.k, loop=False)
        else:
            raise ValueError("build_edges is 'none' but edge_index missing.")
        data.edge_index = edge_index
        return data

    def _edge_attributes(self, data: 'Data') -> torch.Tensor:
        ei = data.edge_index
        if self.use_distance and getattr(data, "pos", None) is not None:
            src, dst = ei[0], ei[1]
            d = (data.pos[src] - data.pos[dst]).pow(2).sum(-1).sqrt()  # [E]
            rbf = self.rbf(d)  # [E, K]
            return self.edge_mlp(rbf)
        else:
            ones = torch.ones(ei.size(1), 1, device=ei.device)
            return self.edge_mlp(ones)

    def pool_global(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        elif self.pool == "add":
            return global_add_pool(x, batch)
        else:
            return global_max_pool(x, batch)

    def forward(self, data: 'Data', return_original_score: bool = False) -> torch.Tensor:
        """
        data can be a single Data or a Batch from PyG (has .batch attr)
        if return_original_score is True, use inverse_transform to get original score

        Returns:
        return_orginal_score=False, score in [0,1]
        return_orginal_score=True, score in [1.32, 5.34]
        """
        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        x = data.x
        pos = getattr(data, "pos", None)
        if pos is not None:
            data.pos = self._center_pos(pos)

        data = self._ensure_edges(data)
        edge_index = data.edge_index
        edge_attr = self._edge_attributes(data)

        h = self.node_in(x)
        for i, conv in enumerate(self.convs):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            #h = self.norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + h_res  # residual

        g = self.pool_global(h, data.batch)
        normalized_score = self.head(g).squeeze(-1)  # [B]

        if return_original_score and self.normalizer is not None:
            original_score = self.inverse_transform(normalized_score)
            return original_score
        return normalized_score

    def inverse_transform(self, normalized_score:torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            raise RuntimeError(f"Normalizer not loaded! Fail to inverse transform.")
        device = normalized_score.device
        normalized_np = normalized_score.detach().cpu().numpy().reshape(-1,1)
        original_np = self.normalizer.inverse_transform(normalized_np).flatten()
        original_score = torch.tensor(original_np, dtype=torch.float32).to(device)
        return original_score

    def predict_score(self, data:'Data') -> dict:
        with torch.no_grad():
            normalized = self.forward(data, return_original_score=False)
            if self.normalizer is not None:
                original = self.inverse_transform(normalized)
            else:
                original = normalized * (5.35 - 1.32) + 1.32

            categories = []
            for score in original.cpu().numpy():
                if score < 3.0: 
                    categories.append('Easy')
                elif score < 4.0:
                    categories.append("Medium")
                else:
                    categories.append("Hard")

            return {
               'normalized_score': normalized,
               'original_score': original,
               'category': categories
            }


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), act(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class RBF(nn.Module):
    """Radial basis function expansion for distances (rotation/translation invariant)."""
    def __init__(self, num_kernels=32, r_min=0.0, r_max=5.0, gamma=None):
        super().__init__()
        self.num_kernels = num_kernels
        centers = torch.linspace(r_min, r_max, num_kernels)
        self.register_buffer('centers', centers)
        if gamma is None:
            gamma = 1.0 / (2 * ((r_max - r_min) / num_kernels) ** 2)
        self.gamma = gamma
    def forward(self, d):
        # d: [E, 1]
        d = d.unsqueeze(-1) if d.dim()==1 else d
        return torch.exp(-self.gamma * (d - self.centers)**2)

