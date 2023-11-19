from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from ....util.misc import NestedTensor


class Resnet(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        #   layer1冻结
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        #   如果return_interm_layers为True，记录每一层的输
        if return_interm_layers:#False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        #   IntermediateLayerGetter接受nn.Module和一个dict作为初始化参数，返回函数结果
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)                  


    def forward(self, tensor_list: NestedTensor):
        # f1 3, 512, 512  -> 64, 128, 128
        # f2 64, 128, 128 -> 512, 64, 64
        # f3 512, 64, 64  -> 1024, 32, 32
        # f4 1024, 32, 32 -> 2048, 16, 16
        inp = tensor_list.tensors
        xs = self.body(inp)
        out: Dict[str, NestedTensor] = {}
        for name, value in xs.items():
            m = tensor_list.mask#哪里是padding补零的
            assert m is not None
            #   将mask插值到与输出特征图尺寸一致
            mask = F.interpolate(m[None].float(), size=value.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(value, mask)
        #    {'0': f1, '1': f2, '2': f3, '3': f4} 
        # or {'0': f4}
        return out