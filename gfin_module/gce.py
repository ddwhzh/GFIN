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


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        #   Encoder层，默认有6层
        #   _get_clones是对相同结构的模块进行复制，返回一个nn,ModuleList实例
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        #   归一化层
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        #   src对应backbone最后一层输出的特征图，并且维度映射到了hidden_dim，shape是（h*w, b, hidden_dim）;
        #   pos对应backbone最后一层输出的特征图对应的位置编码，shape是(h*w, b, c)
        #   src_key_padding_mask对应backbone最后一层输出的特征图对应的mask，shape是（b, h*w）
        #   35x35, N, 2048 
        output = src
        #   output: 35x35, N, 2048 
        #   -> 
        #   output: N, 2048, 35, 35
        for i,layer in enumerate(self.layers):
            output, self_attn = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        #   N, 2048, 35, 35
        return output, self_attn

class TransformerEncoderLayer(nn.Module):
    '''
        transformer嵌入层
    '''
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu", 
                 normalize_before=False):
        super().__init__()
        #   多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model   
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.build_ffn(dropout, activation, d_model, dim_feedforward)
        #self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True) 

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def build_ffn(self, dropout, activation, d_model, dim_feedforward):
        # Implementation of Feedforward model  
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
    

    def forward_ffn(self, src, v):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)   
        return src

    def forward_post(self,
                     src,   #   input embedding
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        #   多头自注意力
        #   将位置信息融合src + pos (35x35, N, 2048 -> 35x35, N, 2048)
        q = k = self.with_pos_embed(src, pos)
        v = src
        src2, self_attn = self.self_attn(query=q, key=k, value=v, 
                        attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        #   映射
        src = self.forward_ffn(src, v)  
        return src, self_attn           

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)  #   唯一区别正则化提前了
        q = k = self.with_pos_embed(src2, pos)
        v = src2
        src2, self_attn = self.self_attn(query=q, key=k, value=v, 
                              attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, self_attn

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        #   默认不使用
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        #   35x35, N, 2048 -> 
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

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
    # if activation=="prelu":
    #     return F.prelu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

