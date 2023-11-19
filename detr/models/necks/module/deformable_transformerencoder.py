import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
from .utils import _get_clones, _get_activation_fn



class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, 
                pos=None, padding_mask=None):
        output = src
#######################################  new  #################################################
        #   N, WH + WH , num_feature_levels, 2
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
#######################################  new  #################################################       
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # out-of-reference points have relative coords. larger than 1
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)     
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.build_ffn(dropout, activation, d_model, d_ffn)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def build_ffn(self, dropout, activation, d_model, dim_feedforward):
        # Implementation of Feedforward model  
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, tgt=None):
        if tgt is None:
            # self attention N, 35x35, 256 
            q = self.with_pos_embed(src, pos)
            k = reference_points    #change
            v = src
            src2, sampling_locations, attn_weights = self.self_attn(q, k, v, 
                                                    spatial_shapes, level_start_index, padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            # ffn
            src = self.forward_ffn(src)
            return src
        else:
            q = self.with_pos_embed(tgt, pos),
            k = reference_points    #change
            v = src
            # self attention
            tgt2, sampling_locations, attn_weights = self.self_attn(q, k, v, spatial_shapes,
                                                                    level_start_index, padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # ffn
            tgt = self.forward_ffn(tgt)

            return tgt
