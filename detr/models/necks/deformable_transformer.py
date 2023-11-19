# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import torch
from torch import nn

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from .module.ops.modules import MSDeformAttn
import math

from .module.deformable_transformerencoder import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from .module.deformable_transformerdecoder import DeformableTransformerDecoderLayer, DeformableTransformerDecoder

def build_transformer(args):
    '''
        建立一个transformer
    '''
    return Transformer(
        d_model = args.hidden_dim,                  #   256
        dropout = args.dropout,                     #   0.1
        nhead = args.nheads,                        #   多头8
        dim_feedforward = args.dim_feedforward,     #   transformer blocks中前馈层的中间大小2048
        num_encoder_layers = args.enc_layers,       #   encoder层的层数6
        num_decoder_layers = args.dec_layers,       #   decoder层的层数6
        normalize_before = args.pre_norm,           #   预先正则化
        return_intermediate_dec = True,
        num_feature_levels=args.num_feature_levels, 
        dec_n_points=args.dec_n_points,  
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage, 
        two_stage_num_proposals=args.num_queries)

class Transformer(nn.Module):
    '''和transformer论文结构一样'''
    def __init__(self, d_model = 512, 
                 nhead = 8, 
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 1024,
                 dropout = 0.1,
                 activation="relu", 
                 normalize_before = False,
                 return_intermediate_dec = False,
                 spatial_level=3,
                 num_feature_levels=4, 
                 dec_n_points=4,  
                 enc_n_points=4,
                 two_stage=False, 
                 two_stage_num_proposals=300):
        super(Transformer,self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.return_intermediate_dec = return_intermediate_dec
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals


        self.encoder = self.build_encode(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, num_feature_levels, enc_n_points)
        self.decoder = self.build_decode(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, return_intermediate_dec, num_feature_levels, dec_n_points)
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
            self.reference_points = None
        else:
            self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()
        


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
                
                

    
    def build_encode(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, num_feature_levels, enc_n_points):
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        return encoder

    def build_decode(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, return_intermediate_dec, num_feature_levels, dec_n_points):
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        return decoder



    def forward(self, srcs, masks, pos_embed, query_embed):
        assert self.two_stage or query_embed is not None
        # prepare input for encoder
    
        src, mask, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed = \
            self.encode_input_reshape(srcs, masks, pos_embed)
        # encoder  
        # memory : 1, WH + WH + , 256
        memory = self.encode_forward(src, mask, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed, self.encoder)
        # prepare input for decoder
        #   N, 35x35, 256 
        tgt, reference_points, query_embed, init_reference_out, enc_outputs_class, enc_outputs_coord_unact = \
            self.decode_input_reshape(query_embed, memory, mask, spatial_shapes, self.reference_points, self.decoder)
        # decoder
        hs, inter_references = self.decode_forward(tgt, reference_points, memory, 
                                                   spatial_shapes, level_start_index, valid_ratios, query_embed, mask, self.decoder)

        inter_references_out = inter_references
        outputs = {'src':srcs[-1],
                   'memory':memory, 'init_reference':init_reference_out, 'inter_references':inter_references_out,
                   'out_query':hs}
        if self.two_stage:
            outputs.update({'enc_outputs_class':enc_outputs_class, 
                            'enc_outputs_coord_unact':enc_outputs_coord_unact})
            return outputs
        return outputs


    def encode_input_reshape(self, srcs, masks, pos_embed):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embed)):
            #   N, 256, 35, 35
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            #   N, 35x35, 256
            src = src.flatten(2).transpose(1, 2)
            #   N, 35x35
            mask = mask.flatten(1)
            #   N, 35x35, 256
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask) 

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        return src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten



    def encode_forward(self, src, mask, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, encoder):
        memory = encoder(src, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask)
        return memory

    def decode_input_reshape(self, query_embed, memory, mask, spatial_shapes, reference_points_func = None,  decoder=None):
        bs, _, c = memory.shape

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask, spatial_shapes)
            # hack implementation for two-stage Deformable DETR  
            # 预测输出的score
            enc_outputs_class = decoder.class_embed[decoder.num_layers](output_memory)
            # 编码后的anchor+相对偏差
            enc_outputs_coord_unact = decoder.bbox_embed[decoder.num_layers](output_memory) + output_proposals         
        else:
            enc_outputs_class = None
            enc_outputs_coord_unact = None
      
        if self.two_stage:      
            topk = self.two_stage_num_proposals
            # 从二元（前景/背景）类中获取分数，尽管输出有91个输出点，但仅第一点将被用于损失计算
            # 选择最大的topk的proposal
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            # 选择对应topl score的编码后的框
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            # 相当于对proposal的微调
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            #   100, 256
            if query_embed.dim()==2:
                query_embed, tgt = torch.split(query_embed, c, dim=1) 
                 # bs x Lq x d_model     每个sample的query相同，参考位置也相同
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)                   # 初始的query
            else:#  N, 100, 256
                tgt = torch.zeros_like(query_embed)
            # 每个query是学习到不同的参考位置
            reference_points = reference_points_func(query_embed).sigmoid()
            init_reference_out = reference_points
        return tgt, reference_points, query_embed, init_reference_out, enc_outputs_class, enc_outputs_coord_unact

    def decode_forward(self, tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask, decoder):
        hs, inter_references = decoder(tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask)
        return hs, inter_references


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos


    def gen_encoder_output_proposals(self, memory, mask, spatial_shapes):
        #   考虑two_stage的情况，相当于先利用encoder进行proposals的粗选，
        # 即更具score筛选topk个候选位置。
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            #不同尺寸的anchor框，scale其实是对有效区域的处理，
            # 后续对output_proposals的处理是筛选掉边界附近的候选，
            mask_flatten_ = mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # 每个sample的有效尺寸
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2) 
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale  # 归一化
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)#  方形的候选框，其实等价于anchor
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            # 每个level的起始索引
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # 筛选有效的proposal，将靠近边界的点舍弃inf
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # 输出是对应位置的特征和编码后的proposal， 
        # 对应位置的特征用于映射proposal的类别score以及校正偏差。
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        
        valid_ratio = torch.stack([valid_ratio_w, 
                                   valid_ratio_h], -1)
        return valid_ratio