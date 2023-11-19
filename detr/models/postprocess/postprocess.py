import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...util.box_ops import box_cxcywh_to_xyxy
from ...util.misc import  interpolate
# try:
#     from panopticapi.utils import id2rgb, rgb2id
# except ImportError:
#     pass


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, trans_type='detr'):
        super().__init__()
        self.trans_type = trans_type

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        scores, labels, boxes  = self.get_outputs(outputs, target_sizes)
        results = [{'scores': s, 
                    'labels': l, 
                    'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

    @torch.no_grad()
    def get_outputs(self, outputs, target_sizes):
        assert target_sizes.shape[1] == 2
        scores, labels, topk_boxes = self.get_obj_cls(outputs, target_sizes)
        boxes = self.get_sub_bbox(outputs, target_sizes, topk_boxes)
        return scores, labels, boxes

    def get_obj_cls(self,outputs, target_sizes):
        out_logits = outputs['pred_logits']
        assert len(out_logits) == len(target_sizes)
        prob = F.softmax(out_logits, -1)
        if self.trans_type=='detr':
            scores, labels = prob[..., :-1].max(-1)#少最后一维度
            topk_boxes = None
        elif self.trans_type=='ddetr':
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
        return scores, labels, topk_boxes

    def get_sub_bbox(self,outputs, target_sizes, topk_boxes):
        out_bbox = outputs['pred_boxes']
        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        if self.trans_type=='ddetr':
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        return boxes
