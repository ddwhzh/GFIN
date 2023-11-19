import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from detr.util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
from .modules import MHAttentionMap, MaskHeadSmallConv


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        self.hidden_dim, self.nheads = detr.transformer.d_model, detr.transformer.nhead
        self.build_head()

      
    def build_head(self):
        hidden_dim, nheads = self.hidden_dim, self.nheads#256, 8
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)


    def forward(self, samples: NestedTensor, targets=None):
        srcs, masks, poses, features = self.backbone_forward(samples)
        outputs = self.neck_forward(srcs, masks, poses, self.detr.query_embed.weight) 
        out = self.head_forward(outputs, features, masks)
        out.update(outputs)
        return out

    def backbone_forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        ###backbone###
        #   samples:    N, 3, 512, 512 
        #-> features:   N, 2048, 35, 35 (C4)*4
        #   pos:        N, 256, 35, 35 (C4)*4
        features, pos = self.detr.backbone(samples)

        srcs = []
        masks = []
        poses = []
        for j, feat in enumerate(features):
            if j== len(features)-1:
                src, mask = feat.decompose()
                srcs.append(self.detr.input_proj(src))
                masks.append(mask)
                poses.append(pos[j])
                assert mask is not None

        return srcs, masks, poses, features
    
    def neck_forward(self, srcs, masks, poses, query_embed):
        ###neck###
        #   src:               N, 2048, 35, 35 -> N, 256, 35, 35
        #   mask:              N, 35, 35 
        #   query_embed:       100, N, 256
        #   pos:               256, 35, 35
        #-> hs                 6, N, 100, 256
        #{'src':src,'out_query':hs,'memory':memory}
        outputs = self.detr.transformer(srcs, masks, query_embed, poses)
        return outputs

    def head_forward(self, outputs, features, mask):
        #   hs                 6, N, 100, 256
        #-> outputs_obj_class  N, 100, 80
        x = outputs['out_query']
        memory = outputs['memory']
        src = outputs['src']
        
        bs = x.shape[1]
        
        outputs_class = self.detr.class_embed(x)
        #   outputs_sub_coord  N, 100, 4 
        outputs_coord = self.detr.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1]} 
        
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # FIXME h_boxes takes the last one computed, keep this in mind
        # x:       N, 100, 256
        # memory:  N, 256, 35, 35
        # mask:    N, 35,  35
        bbox_mask = self.bbox_attention(x[-1], memory, mask=mask[-1])
        # src:        N, 100, 256
        # bbox_mask:  N, 100, 8, 35, 35
        # mask:       N, 35,  35
        seg_masks = self.mask_head(src, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        # seg_masks:    N, 100, w, h
        out["pred_masks"] = outputs_seg_masks
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个让 torchscript 满意的变通方法，
        # 因为 torchscript 不支持非同质值的 dictionary，
        # 比如一个 dict 同时拥有 Tensor 和 list。
        return [{'pred_logits': a, 
                 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]




