import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from collections import OrderedDict
import copy

import pocket
import torchvision
from torch.jit.annotations import Optional, List, Dict, Tuple

from ops import norm_tensor
from add_on.net import *

from gfin_module.gce import TransformerEncoder, TransformerEncoderLayer
from gfin_module.pid import TransformerDecoder, TransformerDecoderLayer
from gfin_module.position_encoding import  build_1d_position_encoding
from gfin_module.hof import HOFmodule
from detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list)


class Base_InteractionHead(nn.Module):

    def backbone_forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        no_prop_srcs = []
        srcs = []
        masks = []
        poses = []
        for j, feat in enumerate(features):
            src, mask = feat.decompose()
            no_prop_srcs.append(src)
            if j== len(features)-1:#最后一层
                srcs.append(self.input_proj(src))
                masks.append(mask)
                poses.append(pos[j])
                assert mask is not None

        return no_prop_srcs, srcs, masks, poses
########################################################################################################################

    def build_encode(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before):
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        return encoder
    
    def encode_input_reshape(self, srcs, masks, poses):
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poses)):
            #   N, 256, 35, 35
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            #   35x35, N, 256
            src = src.flatten(2).permute(2, 0, 1)  
            #   N, 35x35
            mask = mask.flatten(1)
            #   35x35, N, 256
            if src.shape[-1]==self.hidden_state_size:
                pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  
                lvl_pos_embed = pos_embed 
            else:
                x = src.permute(1, 0, 2)
                pos_embed = [self.pos_2d_embed(s.view(w, h, 1, -1), m.view(1, w, h)) for s,m in zip(x, mask)]
                pos_embed = torch.cat(pos_embed).flatten(2).permute(2, 0, 1)  
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 0)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 0)
        return src_flatten, mask_flatten, pos_embed_flatten, spatial_shapes

    def encode_forward(self, src, mask, pos_embed, encoder):
        memory, memory_attn = encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory, memory_attn

########################################################################################################################

    def build_decode(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, normalize_before, return_intermediate_dec, 
                     num_channels, representation_size):
        decoder_layer = TransformerDecoderLayer(d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward, 
                                                num_channels = num_channels, dropout = dropout, activation = activation, 
                                                normalize_before = normalize_before, representation_size=representation_size)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        return decoder


    def decode_input_reshape(self, unary_tokens, hof_memory, mask, pos, boxes_h_collated,
                             query_embed=None):
        tgt = unary_tokens.permute(1, 0, 2)
        
        mask = mask.flatten(1)
        pos_embed = pos.flatten(-2).permute(2,0,1)
        pos_embed = torch.cat([pos_embed, pos_embed], dim=-1)
        
        bs = tgt.shape[1]
        L  = tgt.shape[0]
        S  = hof_memory.shape[0]
        attn_mask = torch.zeros((bs*self.nheads, L, S), device = self.device, dtype=torch.bool)
        for i, l in enumerate(boxes_h_collated):
            ni = i*self.nheads
            attn_mask[ni:ni+self.nheads, :len(l), mask[i]] = True
        return tgt, hof_memory, mask, query_embed, pos_embed, attn_mask
    

    def decode_input_reshape_idx(self, tgt, hof_memory, mask, pos, b_idx, 
                                 query_embed=None, box_pair_spatial_embed=None):

        memory = hof_memory[:,b_idx:b_idx+1,:]
        #   添加线性层
        pos_embed = pos[b_idx:b_idx+1].flatten(-2).permute(2,0,1)
        pos_embed = torch.cat([pos_embed, pos_embed], dim=-1)

        
        mask = mask[b_idx:b_idx+1].flatten(-2)

        if query_embed is None and box_pair_spatial_embed is not None:
            query_embed = box_pair_spatial_embed.unsqueeze(1)
        elif query_embed is not None:
            query_embed = query_embed[:tgt.shape[0]].unsqueeze(1)
                
        return tgt, memory, mask, query_embed, pos_embed


    def decode_forward(self, tgt, memory, mask, query_embed, pos_embed, attn_mask, decoder):
        hs, self_attn, cross_attn = decoder(tgt, memory, memory_key_padding_mask=mask, 
                                            query_pos=query_embed, pos=pos_embed, memory_mask = attn_mask)
        hs = hs.transpose(1, 2)
        return hs, self_attn, cross_attn
 

