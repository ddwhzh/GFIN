import copy
from typing import Optional, List

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module
    Parameters:
    -----------
    appearance_size: int (hidden_state_size * 2, num_channels :512, 2048)
        Size of the appearance features
    spatial_size: int       (representation_size: 512)
        Size of the spatial features
    hidden_state_size: int (representation_size: 512)
        Size of the intermediate representations
    cardinality: int (cardinality: 16)
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        hidden_state_size: int, cardinality: int, activation='relu'
    ) -> None:
        super().__init__()
        self.cardinality = cardinality
        #   32
        sub_repr_size = int(hidden_state_size / cardinality)
        assert sub_repr_size * cardinality == hidden_state_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, hidden_state_size)
            for _ in range(cardinality)
        ])
        self.activation = _get_activation_fn(activation)
        
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        fc1 = [f(appearance)  for f in self.fc_1]
        fc2 =  [f(spatial)  for f in self.fc_2]
        out_list =  [self.activation(x1 + x2)  for x1, x2 in zip(fc1, fc2)]
        out_list = [f(x) for f, x in zip(self.fc_3, out_list)]
        out = torch.stack(out_list)
        out = out.sum(dim=0)
        out = self.activation(out)
        return out

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

