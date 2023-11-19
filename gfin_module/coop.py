
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict
import copy
from add_on.attention import *
import pocket

from  gfin_module.mbf import MultiBranchFusion
class HOFmodule(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, num_layers: int = 2,
        dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([HOFmoduleLayer(
            hidden_size=hidden_size, representation_size=representation_size,
            num_heads=num_heads, dropout_prob=dropout_prob, return_weights=return_weights
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
        activation="relu"
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

        self.message = nn.ModuleList([nn.Sequential(
                                   nn.Linear(self.sub_repr_size, self.sub_repr_size),
                                   ) for _ in range(num_heads)])
        
        self.aggregate = nn.Linear(representation_size, hidden_size)
        self.norm  = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.ffn = pocket.models.FeedForwardNetwork(hidden_size, hidden_size * 4, dropout_prob)
        self.activation = _get_activation_fn(activation)
        
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
        
        # Unary features (8, N, 64)
        Kappe = self.activation(self.unary(appearance))    
        Kappe = self.reshape(Kappe)

        #   乘法操作变为cat操作Q, K
        i, j = torch.meshgrid(torch.arange(n, device=device), 
                              torch.arange(n, device=device))
        attn_features = torch.cat([Kappe[:, i], Kappe[:, j], Qspat], dim=-1) #   ([8], N, N, 3*64)

        weights1 = [F.sigmoid(f(m)) for f, m in zip(self.attn1, attn_features)]
        weights2 = [F.sigmoid(f(m)) for f, m in zip(self.attn2, attn_features)]

        Vappe = self.activation(self.unary2(appearance))    
        Vappe = self.reshape(Vappe)
        mess = Vappe.unsqueeze(dim=2).repeat(1, 1, n, 1)  * Qspat        #   8, N, N, 64
        message = [f(m) for f, m in zip(self.message, mess)]
        

        weights = [(w1 + w2)/2 for w1, w2 in zip(weights1, weights2)]
        attn_map = [torch.sum(w * m, dim=0) for w, m in zip(weights,  message)]                
        
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




    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    # if activation=="prelu":
    #     return F.prelu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
