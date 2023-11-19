"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import numpy as np

from torch import nn, Tensor
from typing import List
from collections import OrderedDict

import pocket
from ops import compute_spatial_encodings
from add_on.net import *
from gfin_module.base_interaction_head import Base_InteractionHead
from gfin_module.hof import HOFmodule

class InteractionHead(Base_InteractionHead):
    """
    Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_pair_predictor: nn.Module
        Module that classifies box pairs
    hidden_state_size: int      (256)
        Size of the object features
    representation_size: int    (512)
        Size of the human-object pair features
    num_channels: int           (2048)
        Number of channels in the global image features
    num_classes: int            (24)
        Number of target classes
    human_idx: int
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
    """
    def __init__(self,
        box_pair_predictor: nn.Module,
        num_channels, 
        object_class_to_target_class: List[list],
        args, num_query=500, 
        tb_writer = None
    ) -> None:
        super().__init__()
        self.num_query = num_query
        self.box_pair_predictor = box_pair_predictor
        
        hidden_state_size = args.hidden_dim 
        self.hidden_state_size = hidden_state_size
        
        representation_size = args.repr_dim
        self.representation_size = representation_size

        self.num_classes = args.num_classes
        self.human_idx = args.human_idx
        
        self.object_class_to_target_class = object_class_to_target_class

        self.args = args
        self.nheads = args.nheads
        self.menc_inter_layer = args.menc_inter_layer
        self.dec_inter_layers = args.dec_inter_layers
        self.num_heads = args.nheads
        self.inference = args.inference
        self.layer_num = args.dec_inter_layers
        self.device = args.device

        self.input_proj = nn.Conv2d(num_channels[-1], hidden_state_size, kernel_size=1)      
        if args.dataset =='hicodet':                                         
            nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
            nn.init.constant_(self.input_proj.bias, 0)     
        
        self.HOI_encoder = self.build_encode(d_model = hidden_state_size, nhead=args.nheads, num_encoder_layers = args.enc_inter_layers, 
                                             dim_feedforward = args.dim_feedforward, dropout = args.dropout, activation = "relu", normalize_before = False)
    
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, representation_size), nn.ReLU(),)
        
        self.pairwise_fusion_layer = HOFmodule(
            hidden_size = hidden_state_size,
            representation_size = representation_size,
            num_layers = args.menc_inter_layer,
            num_heads=args.nheads, return_weights = True)
        
        self.HOI_layer  = self.build_decode(d_model = hidden_state_size * 2, nhead = args.nheads, num_decoder_layers = args.dec_inter_layers, 
                                            dim_feedforward = args.dim_feedforward, dropout = args.dropout, activation = "relu", normalize_before = False, 
                                            return_intermediate_dec = True, num_channels = representation_size, representation_size = representation_size)

        self.detr_res_linear = nn.Linear(hidden_state_size, hidden_state_size)
        self.detr_memory_linear = nn.Linear(hidden_state_size, hidden_state_size)
          

    
    def forward(self, images, resnet_features: OrderedDict, srcs, 
                detr_memory, image_shapes: Tensor, sample_wh:Tensor, 
                region_props: List[dict], masks=None, poses=None):
        device = self.device
        ######################################   1.GCE   #####################################
        hof_memory = None
        hof_memory = self.GEC_forward(resnet_features, srcs, detr_memory, masks, poses)
        ######################################   GCE   #####################################               
        boxes_h_collated = []
        boxes_o_collated = []
        prior_collated = []
        object_class_collated = []
        attn_maps_collated = []
        HOI_tokens_collated = []     
        prior_score = []
        #   每一个sample
        for b_idx, props in enumerate(region_props):
            n = len(props['boxes'])
            box = props['boxes']  
            score = props['scores']
            label = props['labels']
            unary_token = props['hidden_states']    
            is_human = (label == self.human_idx)
            n_h = torch.sum(is_human)
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                prior_score.append(torch.zeros(0, device=device))
                
                unary_attn = [[torch.zeros((0, 0, 1), device=device, dtype=torch.float) for _ in range(self.num_heads)] for _ in range(self.menc_inter_layer)]
                HOI_token = torch.zeros((self.layer_num, 0, self.box_pair_predictor.weight.shape[-1]), dtype=torch.float, device=device)
                HOI_attn = torch.zeros((self.dec_inter_layers, 1, 0, 0), device=device, dtype=torch.float)
                HOI_tokens_collated.append(HOI_token)
                attn_maps_collated.append((unary_attn[-1], HOI_attn))
                continue  
   
            # Permute human instances to the top
            if not torch.all(label==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == False).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                box = box[perm]
                score = score[perm]
                label = label[perm]
                unary_token = unary_token[perm]
            # Get the pairwise indices  (N, N)
            x, y = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device))
            if self.args.dataset == 'hicodet':
                x_keep, y_keep = torch.nonzero(torch.logical_and(x!=y, x < n_h)).unbind(1)
            elif self.args.dataset == 'vcoco':
                x_keep, y_keep = torch.nonzero(x < n_h).unbind(1)
            prior = self.compute_prior_scores(x_keep, y_keep, score, label)
            x, y = x.flatten(), y.flatten()
            
            ######################################   2.HOF  #####################################   
            # Compute spatial features  (NxN, 36)
            box_pair_spatial_embed = None
            per_image_shapes = image_shapes[b_idx]
            box_pair_spatial = compute_spatial_encodings([box[x]], [box[y]], [per_image_shapes])
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            box_pair_spatial_embed = box_pair_spatial.reshape(n, n, -1)
            #   pair feature encoder
            unary_token, unary_attn = self.pairwise_fusion_layer(unary_token, box_pair_spatial_embed)#   N, 256
            unary_token = torch.cat([unary_token[x_keep], unary_token[y_keep]], -1)
            ######################################   HOF  #####################################
            
            ######################################   3.PID  #####################################
            box_pair_spatial_embed[x_keep, y_keep]
            tgt, memory, mask, query_embed, pos_embed = \
                self.decode_input_reshape_idx(unary_token.unsqueeze(1), hof_memory, masks[-1], poses[-1], b_idx)
            HOI_token, HOI_attn, _ = self.decode_forward(tgt, memory, mask, None, pos_embed, None, self.HOI_layer)
            HOI_token = HOI_token.squeeze(1)
            ######################################   PID  ##################################### 
            prior_score.append(score)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(label[y_keep])
            prior_collated.append(prior)   
            HOI_tokens_collated.append(HOI_token)
            attn_maps_collated.append((unary_attn,  HOI_attn))  
 
        HOI_tokens_collated = torch.cat(HOI_tokens_collated, dim=1)
        if not self.training:
            HOI_tokens_collated = HOI_tokens_collated[-1:] 
        logits = self.box_pair_predictor(HOI_tokens_collated) 
        
        return logits, prior_collated, prior_score,\
                boxes_h_collated, boxes_o_collated, \
                object_class_collated, attn_maps_collated

    def compute_prior_scores(self, x_keep: Tensor, y_keep: Tensor, scores: Tensor, object_class: Tensor) -> Tensor:
        prior_h = torch.zeros(len(x_keep), self.num_classes, device=self.device)
        prior_o = torch.zeros_like(prior_h)
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x_keep].pow(p)
        s_o = scores[y_keep].pow(p)
        # Map object class index to target class index(过滤矩阵滤掉不正确的pair)
        # Object class index to target class index is a one-to-many mapping
        #   Vcoco会自动不过滤没有宾语的动作
        target_cls_idx = [self.object_class_to_target_class[obj.item()] if x_keep[i] != y_keep[i]
                          else range(self.num_classes) for i, obj in enumerate(object_class[y_keep])]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]#pairN, 24
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]#pairN, 24

        return torch.stack([prior_h, prior_o])#2, pairN, 24
    

    def GEC_forward(self, resnet_features, srcs, detr_memory, masks, poses):
        global_features = [self.input_proj(resnet_features[-1])]  
        # _, global_features, _, _ = self.backbone_forward(images)
        w, h = detr_memory.shape[-2], detr_memory.shape[-1]    
        src, mask, pos_embed, spatial_shapes = self.encode_input_reshape(srcs = global_features[-1:], masks = masks[-1:], poses = poses[-1:])
        hoi_memory, _ = self.encode_forward(src, mask, pos_embed, self.HOI_encoder)

        hoi_memory = self.detr_res_linear(srcs[-1].flatten(-2).permute(2, 0, 1)) + hoi_memory         
        detr_memory = global_features[-1] + \
            self.detr_memory_linear(detr_memory.permute(0,2,3,1)).permute(0,3,1,2) 
            
        detr_memory = detr_memory.flatten(-2).permute(2,0,1)
        hof_memory = torch.cat([detr_memory, hoi_memory], dim=-1)     
        return hof_memory