# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
    Modules to compute the matching cost and solve the corresponding LSAP.计算匹配成本和解决相应LSAP的模块。
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ...util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def build_matcher(args, loss_type='ce_loss'):
    '''
        匈牙利算法做匹配
    '''
    #   N, 100, xxx
    return HungarianMatcher(cost_class=args.set_cost_class,  cost_bbox=args.set_cost_bbox,  cost_giou=args.set_cost_giou, loss_type=loss_type)



class HungarianMatcher(nn.Module):
    """ 匈牙利算法
    该类计算目标和网络预测之间的赋值
    为了提高效率，目标不包括no_object。
    正因为如此，一般情况下，预测比目标多。 prection
    在这种情况下，我们对最好的预测进行1对1的匹配，而其他的预测则不匹配（因此被视为非目标）。
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, loss_type='ce_loss'):
        """
        Creates the matcher
        Params:
            cost_class: 这就是分类loss在匹配cost中的相对权重。
            cost_bbox: 这是边界盒坐标的L1 loss在匹配cost中的相对权重。
            cost_giou: 这是在匹配cost中边界盒的giou loss的相对权重。
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_type = loss_type
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, loss_type='ce_loss'):
        """ Performs the matching

        Params:
            outputs: 这是一个至少包含这些条目的听写本。
                 "pred_logits" :  [batch_size=2, num_queries=100, num_classes] 
                "pred_boxes"  :  [batch_size=2, num_queries=100, 4] 

            targets  : 这是一个目标列表(len(target) = batch_size)，其中每个目标是一个dict，包含
                [targets for i in range(batchsize)]
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth        "labels" : [num_target_boxes]
                           objects in the target) containing the class labels                                            "boxes"  : [num_target_boxes, 4]
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            一个大小为batch_size的列表，包含(index_i, index_j)的元组，其中
                - index_i是所选预测的indices（按顺序）
                - index_j是对应的被选目标的indices（按顺序）
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        #   [batch_size , num_queries, num_obj]
        C = self.get_C(outputs, targets, loss_type)
        sizes = [len(v["boxes"]) for v in targets]#为每一个GT选择一个pred
        #   list[batch_size] tuple(2)  tensor[num_obj]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]   #   匈牙利算法计算，选cost最小的
        #    list[batch_size] tuple(2)  tensor[num_obj][num_obj]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def get_C(self, outputs, targets, loss_type=None):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        C = self.get_cost_class(outputs, targets) + \
            self.get_cost_bbox(outputs, targets) + \
            self.get_cost_giou(outputs, targets)
        C = C.view(bs, num_queries, -1).cpu()
        return C


    def get_cost_class(self, outputs, targets):
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) 
        tgt_ids = torch.cat([v["labels"] for v in targets]) 
        if self.loss_type=='ce_loss':
            cost_class = -out_prob[:, tgt_ids]#1-out_obj_prob[:, tgt_obj_labels]
        elif self.loss_type=='focal_loss':
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]            
        
        C = self.cost_class * cost_class
        return C


    def get_cost_bbox(self, outputs, targets):
        out_bbox = outputs["pred_boxes"].flatten(0, 1) 
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        C = self.cost_bbox * cost_bbox
        return C


    def get_cost_giou(self, outputs, targets):
        out_bbox = outputs["pred_boxes"].flatten(0, 1) 
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        bbox1 = box_cxcywh_to_xyxy(out_bbox)
        bbox2 = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(bbox1, bbox2)
        C = self.cost_giou * cost_giou
        return C       

