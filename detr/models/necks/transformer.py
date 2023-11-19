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

from .module.transformerencoder import TransformerEncoderLayer, TransformerEncoder
from .module.transformerdecoder import TransformerDecoderLayer, TransformerDecoder

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
        return_intermediate_dec = True)

class Transformer(nn.Module):
    '''和transformer论文结构一样'''
    def __init__(self, d_model = 512, 
                 nhead = 8, 
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 activation="relu", 
                 normalize_before = False,
                 return_intermediate_dec = False):
        super(Transformer,self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.return_intermediate_dec = return_intermediate_dec

        self.encoder = self.build_encode(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before)
        self.decoder = self.build_decode(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, normalize_before, return_intermediate_dec)
        
        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
                    
    def build_encode(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before):
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        return encoder

    def build_decode(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, normalize_before, return_intermediate_dec):
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        return decoder


    def encode_input_reshape(self, srcs, masks, poses):
        '''
            src:               N, 256, 35, 35 -> 1050, N, 256
            mask:              N, 35, 35      -> N, 35x35
            #nn.Embedding(num_queries, self.hidden_dim)   
            #   backbone, pos
            pos_embed:         N, 256, 35, 35 -> 35x35, N, 256     
        '''
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
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  
            lvl_pos_embed = pos_embed #+ self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 0)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 0)

        # src = src.flatten(2).permute(2, 0, 1)               
        # mask = mask.flatten(1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        return src_flatten, mask_flatten, pos_embed_flatten, spatial_shapes


    def encode_forward(self, src, mask, pos_embed, encoder):
        '''
            src:       35x35, N, 256  -> memory: 35x35, N, 256            
        '''
        memory, memory_attn = encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory, memory_attn

    def decode_input_reshape(self, query_embed, bs):
        '''
            query_embed:       100, 256       -> 100, N, 256  
            tgt:    
        '''
        if query_embed.dim()==2:               
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed) 
        return tgt, query_embed



    def decode_forward(self, tgt, memory, mask, query_embed, pos_embed, decoder):
        '''
            query_embed: 100, N, 256  -> tgt: 100, N, 256(一堆0)   
            tgt:    100, N, 256
            memory: 35x35, N, 256 
            mask:   N, 35x35   -> hs:  6, 100, N, 256      
        '''
        hs, self_attn, cross_attn = decoder(tgt, memory, memory_key_padding_mask=mask, query_pos=query_embed, pos=pos_embed)
        if self.return_intermediate_dec:
            hs = hs.transpose(1, 2)
        else:
            hs = hs.unsqueeze(0).transpose(1, 2)
            self_attn = self_attn.unsqueeze(0)
            cross_attn = cross_attn.unsqueeze(0)
        return hs, self_attn, cross_attn

    def forward(self, srcs, masks, query_embed, poses):
        bs, c, h, w = srcs[-1].shape 
        src, mask, pos_embed, spatial_shapes = self.encode_input_reshape(srcs, masks, poses)
        memory, memory_attn = self.encode_forward(src, mask, pos_embed, self.encoder)
        
        tgt, query_embed = self.decode_input_reshape(query_embed, bs)
        hs, self_attn, cross_attn = self.decode_forward(tgt, memory, mask, query_embed, pos_embed, self.decoder)
        
        memory = memory.permute(1, 2, 0)[:, :, -h*w:].view(bs, c, h, w) 
        outputs = {'src':srcs[-1],
                   'memory':memory,'memory_attn':memory_attn,
                   'out_query':hs,'self_attn':self_attn,'cross_attn':cross_attn}
        return outputs 



