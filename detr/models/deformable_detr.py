# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)
from .module.modules import MLP


from .necks.module.utils import _get_clones

from .backbones.backbone import build_backbone
from .necks.deformable_transformer import build_transformer
from .matcher.matcher import build_matcher
from .losses.setcriterion import SetCriterion
from .postprocess.postprocess import PostProcess

import torch.distributed as dist

def build(args):

    device = torch.device(args.device)


    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=args.num_obj_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        args=args,
    )
    # if args.masks:
    #     model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args, 'focal_loss')
    weight_dict = get_weight_dict(args)
    losses = get_losses(args)

    criterion = SetCriterion(args, matcher, weight_dict, 
                             losses, loss_type='focal_loss')
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(trans_type='ddetr')}

    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm()
    #     if args.dataset_file == "coco_panoptic":
    #         is_thing_map = {i: i <= 90 for i in range(201)}
    #         postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, args=None, num_verb_classes=0):
        super().__init__()
        hidden_dim = transformer.d_model#256
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss
        self.num_queries = num_queries#100

        self.backbone = backbone
        self.transformer = transformer

        self.with_box_refine = args.with_box_refine
        self.two_stage = args.two_stage

        #neck decoder
        self.num_feature_levels = num_feature_levels
        if not self.two_stage:# query_embed, tgt (100, 256*2)
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

            
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            #   fpn输出层
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1), 
                                                     nn.GroupNorm(32, hidden_dim),))
            #   多余的层
            for _ in range(num_feature_levels - num_backbone_outs):

                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1), 
                                      nn.GroupNorm(32, hidden_dim),))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                                                           nn.GroupNorm(32, hidden_dim),)])

        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        self.build_head(num_classes, num_verb_classes)


    def build_head(self, num_classes, num_verb_classes):

        hidden_dim = self.hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes )
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (self.transformer.decoder.num_layers + 1) if self.two_stage else self.transformer.decoder.num_layers
        
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        
        if self.two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)



    def forward(self, samples: NestedTensor, targets=None):
        srcs, masks, pos = self.backbone_forward(samples)
        if self.two_stage:
            outputs = self.neck_forward(srcs, masks, pos, None)
        else: 
            outputs = self.neck_forward(srcs, masks, pos, self.query_embed.weight)
        out = self.head_forward(outputs)
        out.update(outputs)
        return out

    def backbone_forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        features, pos = self.backbone(samples)

        no_prop_srcs = []
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            no_prop_srcs.append(src)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        #   假设特征要更高维度，就最后一层映射
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        return no_prop_srcs, srcs, masks, pos


    def neck_forward(self, srcs, masks, pos, query_embeds=None):
        if self.two_stage:
            outputs = self.transformer(srcs, masks, pos, None)
        else: 
            outputs = self.transformer(srcs, masks, pos, query_embeds)
        
        return outputs

    def head_forward(self, outputs):
        
        x = outputs['out_query']
        init_reference = outputs['init_reference']
        inter_references = outputs['inter_references']
        enc_outputs_class = outputs['enc_outputs_class']
        enc_outputs_coord_unact = outputs['enc_outputs_coord_unact']
        
        outputs_classes = []
        outputs_coords = []
        #   6, 100, N, 256 
        for lvl in range(x.shape[0]):
            outputs_class = self.class_embed[lvl](x[lvl])
            outputs_classes.append(outputs_class)
            
            tmp = self.bbox_embed[lvl](x[lvl])
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits_cascade': outputs_class,
               'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1],
               'pred_boxes_cascade': outputs_coord} 
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 
                                  'pred_boxes': enc_outputs_coord}
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def get_weight_dict(args):
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    return weight_dict

def get_losses(args):
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    return losses


