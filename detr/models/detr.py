import torch
import torch.nn.functional as F
from torch import nn
import math

from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .module.modules import MLP

from .segmentations.segmentation import DETRsegm
from .postprocess.postprocesssegm import  PostProcessSegm
from .postprocess.postprocesspanoptic import  PostProcessPanoptic


from .backbones.backbone import build_backbone
from .necks.transformer import build_transformer
from .matcher.matcher import build_matcher
from .losses.setcriterion import SetCriterion
from .postprocess.postprocess import PostProcess

import torch.distributed as dist

def build(args):

    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_obj_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    matcher = build_matcher(args)
    weight_dict = get_weight_dict(args)
    losses = get_losses(args)

    criterion = SetCriterion(args, matcher=matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors



class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_verb_classes=0, aux_loss=False):
        """ 
            Initializes the model.
        Parameters:
            backbone:       要使用的主干网的torch模块。参见backbone.py
            transformer:    架构的torch模块。参见 transformer.py
            num_classes:    物体的种类
            num_queries:    对象查询的数量，即检测槽。这是DETR在一张图像中可以检测到的最大对象数量。对于COCO，我们建议100个查询。
            aux_loss:       如果要使用辅助decoder losses（每个decoder层的loss），则为真。
        """
        super().__init__()
        hidden_dim = transformer.d_model#256
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss
        self.num_queries = num_queries#100

        self.backbone = backbone

        # num_backbone_outs = len(backbone.strides)
        # input_proj_list = []
        # for i in range(num_backbone_outs):
        #     in_channels = backbone.num_channels[i]
        #     input_proj_list.append(nn.Sequential(
        #         nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
        #         #nn.GroupNorm(32, self.hidden_dim),))
        # self.input_proj = nn.ModuleList(input_proj_list)
        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)
        
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)#neck encoder
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)#neck  decoder
        self.transformer = transformer
        self.build_head(num_classes, num_verb_classes)
    
    
    def build_head(self, num_classes, num_verb_classes):
        hidden_dim = self.hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples, targets=None):
        #   bug
        no_prop_srcs, srcs, masks, poses = self.backbone_forward(samples)
        outputs = self.neck_forward(srcs, masks, poses, self.query_embed.weight) # hs, memory
        out = self.head_forward(outputs)
        out.update(outputs)
        return out

    def backbone_forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        ###backbone###
        #   samples:    N, 3, 512, 512 
        #-> features:   N, 2048, 35, 35 (C4)*4
        #   pos:        N, 256, 35, 35 (C4)*4
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
    
    def neck_forward(self, srcs, masks, poses, query_embed):
        ###neck###
        #   src:               N, 2048, 35, 35 -> N, 256, 35, 35
        #   mask:              N, 35, 35 
        #   query_embed:       100, N, 256
        #   pos:               256, 35, 35
        #-> hs                 6, N, 100, 256
        #{'src':src,'out_query':hs,'memory':memory}
        outputs = self.transformer(srcs, masks, query_embed, poses)
        return outputs

    def head_forward(self, outputs):
        #   hs                 6, N, 100, 256
        #-> outputs_obj_class  N, 100, 80
        x = outputs['out_query']

        outputs_class = self.class_embed(x)
        #   outputs_sub_coord  N, 100, 4 
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1],
               'pred_logits_cascade': outputs_class,
               'pred_boxes_cascade': outputs_coord} 
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个让 torchscript 满意的变通方法，
        # 因为 torchscript 不支持非同质值的 dictionary，
        # 比如一个 dict 同时拥有 Tensor 和 list。
        return [{'pred_logits': a, 
                 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



def get_weight_dict(args):
    weight_dict = {'loss_ce': 1, 
                   'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    return weight_dict

def get_losses(args):
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    return losses