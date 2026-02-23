from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import pocket
from gfin_module.mbf import MultiBranchFusion
class HOFmodule(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, num_layers: int = 2,
        dropout_prob: float = .1, return_weights: bool = False,
        directional: bool = False, spatial_refine: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([HOFmoduleLayer(
            hidden_size=hidden_size, representation_size=representation_size,
            num_heads=num_heads, dropout_prob=dropout_prob, return_weights=return_weights,
            directional=directional, spatial_refine=spatial_refine,
        ) for _ in range(num_layers)])
    
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        for layer in self.mod_enc:
            appearance, attn = layer(appearance, spatial)
            attn_weights.append(attn)
        return appearance, attn_weights


class HOFmoduleLayer(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, dropout_prob: float = .1, return_weights: bool = False,
        activation="relu", directional: bool = False, spatial_refine: bool = False,
    ) -> None:
        super().__init__()
        if representation_size % num_heads != 0:
            raise ValueError(
                f"The given representation size {representation_size} "
                f"should be divisible by the number of attention heads {num_heads}."
            )
        self.sub_repr_size = int(representation_size / num_heads)

        self.hidden_size = hidden_size
        self.representation_size = representation_size

        self.num_heads = num_heads
        self.return_weights = return_weights

        self.unary = nn.Linear(hidden_size, representation_size)
        self.unary2 = nn.Linear(hidden_size, representation_size)
        self.pairwise = nn.Linear(representation_size, representation_size)

        self.attn1 = nn.ModuleList([nn.Sequential(
                                   nn.Linear(3 * self.sub_repr_size, 1),
                                   ) for _ in range(num_heads)])

        self.attn2 = nn.ModuleList([nn.Sequential(
                                   nn.Linear(3 * self.sub_repr_size, 1),
                                   ) for _ in range(num_heads)])

        self.attn3 = nn.ModuleList([nn.Sequential(
                                   nn.Linear(3 * self.sub_repr_size, 1),
                                   ) for _ in range(num_heads)])
  
        # self.attn4 = nn.ModuleList([nn.Sequential(
        #                            nn.Linear(3 * self.sub_repr_size, 1),
        #                            ) for _ in range(num_heads)])

        self.message = nn.ModuleList([nn.Sequential(
                                   nn.Linear(self.sub_repr_size, self.sub_repr_size),
                                   ) for _ in range(num_heads)])
        
        self.aggregate = nn.Linear(representation_size, hidden_size)
        self.norm  = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.ffn = pocket.models.FeedForwardNetwork(hidden_size, hidden_size * 4, dropout_prob)
        self.activation = _get_activation_fn(activation)

        self.directional = directional
        self.spatial_refine = spatial_refine

        if directional:
            self.direction_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 3),
            )

        if spatial_refine:
            self.spatial_gate = nn.Sequential(
                nn.Linear(hidden_size, representation_size),
                nn.Sigmoid(),
            )

    def build_ffn(self, representation_size, hidden_size, dropout_prob, activation='relu'):
        self.aggregate1 = nn.Linear(representation_size, hidden_size)
        self.aggregate2 = nn.Linear(hidden_size, hidden_size)
        self.norm1  = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.activation = _get_activation_fn(activation)

    def forward(self, appearance: Tensor, spatial: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        device = appearance.device
        n = len(appearance) 
        # Pairwise features (8, N, N, 64)
        Qspat = self.activation(self.pairwise(spatial))     
        Qspat = self.reshape(Qspat)

        # Direction D: modulate spatial features based on current appearance
        if self.spatial_refine:
            sg = self.spatial_gate(appearance)       # (N, repr_size)
            sg = self.reshape(sg)                    # (8, N, 64)
            Qspat = Qspat * sg.unsqueeze(2)          # (8, N, N, 64)

        # Unary features (8, N, 64)
        Kappe = self.activation(self.unary(appearance))    
        Kappe = self.reshape(Kappe)

        i, j = torch.meshgrid(torch.arange(n, device=device), 
                              torch.arange(n, device=device))

        if self.directional:
            # Direction A: three asymmetric attention branches
            fwd_features = torch.cat([Kappe[:, i], Kappe[:, j], Qspat], dim=-1)
            bwd_features = torch.cat([Kappe[:, j], Kappe[:, i], Qspat], dim=-1)
            int_features = torch.cat([
                Kappe[:, i] * Kappe[:, j],
                (Kappe[:, i] - Kappe[:, j]).abs(),
                Qspat,
            ], dim=-1)

            weights1 = [torch.sigmoid(f(m)) for f, m in zip(self.attn1, fwd_features)]
            weights2 = [torch.sigmoid(f(m)) for f, m in zip(self.attn2, bwd_features)]
            weights3 = [torch.sigmoid(f(m)) for f, m in zip(self.attn3, int_features)]

            gate = torch.softmax(self.direction_gate(appearance), dim=-1)  # (N, 3)
            g1 = gate[:, 0:1].unsqueeze(1)   # (N, 1, 1)
            g2 = gate[:, 1:2].unsqueeze(1)
            g3 = gate[:, 2:3].unsqueeze(1)
            weights = [g1 * w1 + g2 * w2 + g3 * w3
                       for w1, w2, w3 in zip(weights1, weights2, weights3)]
        else:
            attn_features = torch.cat([Kappe[:, i], Kappe[:, j], Qspat], dim=-1)
            weights1 = [torch.sigmoid(f(m)) for f, m in zip(self.attn1, attn_features)]
            weights2 = [torch.sigmoid(f(m)) for f, m in zip(self.attn2, attn_features)]
            weights3 = [torch.sigmoid(f(m)) for f, m in zip(self.attn3, attn_features)]
            weights = [(w1 + w2 + w3) / 3 for w1, w2, w3 in zip(weights1, weights2, weights3)]

        Vappe = self.activation(self.unary2(appearance))    
        Vappe = self.reshape(Vappe)
        mess = Vappe.unsqueeze(dim=2).repeat(1, 1, n, 1)
        message = [f(m) for f, m in zip(self.message, mess)]

        attn_map = [torch.sum(w * m, dim=0) for w, m in zip(weights, message)]                
        
        aggregated_messages = torch.cat(attn_map, dim=-1)           # 1,512
        aggregated_messages = self.activation(aggregated_messages)  # N, 512
        aggregated_messages = self.aggregate(aggregated_messages)
        aggregated_messages = self.dropout(aggregated_messages)
        appearance = self.norm(appearance + aggregated_messages)    # N, 256

        appearance = self.ffn(appearance) 
        if self.return_weights: 
            attn = weights
        else: 
            attn = None
        return appearance, attn

    def forward_ffn(self, tgt, v):
        tgt2 = self.dropout1(self.activation(self.aggregate1(tgt)))
        tgt2 = self.aggregate2(tgt2)
        tgt = v + self.dropout2(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_repr_size)
        x = x.view(*new_x_shape)
        #   value
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        #   pos_embed
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError("Incorrect tensor shape")
    
from gfin_module.activations import get_activation_fn as _get_activation_fn
