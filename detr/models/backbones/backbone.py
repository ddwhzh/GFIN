# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
    Backbone modules.
"""

import torchvision
from torch import nn
from typing import Dict, List

from ...util.misc import (NestedTensor, is_main_process)
from ..module.modules import FrozenBatchNorm2d

from .position_encoding import build_position_encoding
from .network.resnet import Resnet


def build_backbone(args):
    '''
        搭建backbone模型
    '''                                   
    #       是否需要训练backbone（即是否采用预训练backbone）,backbone的学习率0.00001
    train_backbone = args.lr_backbone > 0                                                   
    #       是否要记录backbone的每层输出 False  {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    return_interm_layers =  args.masks  or args.fpn                                                  
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation) 
    #       对backbone输出的特征图进行位置编码，用于后续transformer部分
    position_embedding = build_position_encoding(args)   
    #       将backbone和位置编码集合在一个model  
    #       2048, 35, 35 -> 2048, 35, 35
    model = Joiner(backbone, position_embedding)                                            
    model.num_channels = backbone.num_channels#2048                                              
    return model


class Backbone(Resnet):
    """
        ResNet backbone with frozen BatchNorm.
        ResNet骨干网与冻结的BatchNorm
    """
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        #   从torchvision中下载预训练模型
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation = [False, False, dilation],
            pretrained = is_main_process(), 
            norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        #   ddetr
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    '''
        joiner是nn.Sequential的子类，通过初始化，
        self[0]为backbone,
        self[1]是position encoding
        前向过程就是对backbone的每层输出都进行位置编码，
        最终返回backbone的输出及对应的位置编码结果
    '''
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        out: List[NestedTensor] = []    
        pos = []

        xs = self[0](tensor_list)       #backbone
        for name, value in xs.items():# f0,f1,f2,f3
            #   batch_size, 2048, 35, 35
            out.append(value)               #backbone
            #   batch_size, num_pos_feats*2, h, w
            pos.append(self[1](value).to(value.tensors.dtype))#position encoding
        return out, pos



