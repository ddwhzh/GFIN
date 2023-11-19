import copy
from typing import Optional, List

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from  gfin_module.mbf import MultiBranchFusion
# from row_column_decoupled_attention import MultiheadRCDA

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,#None
                memory_mask: Optional[Tensor] = None,#None
                tgt_key_padding_mask: Optional[Tensor] = None,#None
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        #   tgt:    100, N, 256
        #   memory: 35*35, N, 256 
        output = tgt

        intermediate = []
        intermediate_self_attn = []
        intermediate_cross_attn = []

        for i,layer in enumerate(self.layers):
            output, self_attn, cross_attn = \
                    layer(output, memory, 
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, 
                        query_pos=query_pos)
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_self_attn.append(self_attn)
                intermediate_cross_attn.append(cross_attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:#True
            return torch.stack(intermediate), torch.stack(intermediate_self_attn), torch.stack(intermediate_cross_attn)
        return output, self_attn, cross_attn



class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 num_channels=2048,
                 representation_size=512,
                 activation="relu", 
                 normalize_before=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
    
        attention_module = nn.MultiheadAttention
        self.multihead_attn = attention_module(d_model, nhead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.normalize_before = normalize_before
        self.build_ffn(dropout, activation, d_model, dim_feedforward)
        # self.mbf = MultiBranchFusion(
        #     num_channels, representation_size,
        #     representation_size, cardinality=16
        # )
        #self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True) 

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    # def with_query_embed(self, tensor, pos: Optional[Tensor]):
    #     return self.mbf(tensor, pos)


    def build_ffn(self, dropout, activation, d_model, dim_feedforward):
        # Implementation of Feedforward model  
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        #self.norm4 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def forward_ffn(self, tgt, v):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        #tgt = tgt + torch.sigmoid(self.alpha)*v
        #tgt = self.norm4(tgt)
        return tgt

    def forward_post(self, 
                     tgt, memory, 
                     tgt_mask: Optional[Tensor] = None,#None
                     memory_mask: Optional[Tensor] = None,#None
                     tgt_key_padding_mask: Optional[Tensor] = None,#None
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        #   多头自注意力
        #   tgt:    100, N, 256
        q = k = self.with_pos_embed(tgt, query_pos)
        #q = k = self.with_query_embed(tgt, query_pos)
        v = tgt
        tgt2, self_attn = self.self_attn(query=q, key=k, value=v, 
                                        attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        #   多头全局注意力
        #   tgt:      100, N, 256 
        q2 = self.with_pos_embed(tgt, query_pos)
        #q2 = self.with_query_embed(tgt, query_pos)
        #   memory: 35*35, N, 256 
        k2 = self.with_pos_embed(memory, pos)
        v2 = memory
        #    100, N, 256 | N, 100, 35x35
        tgt2, cross_attn = self.multihead_attn(query=q2, key=k2, value=v2, 
                                               attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt = self.forward_ffn(tgt, v)
        return tgt, self_attn, cross_attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

