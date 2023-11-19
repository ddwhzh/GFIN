"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
from sklearn.metrics import SCORERS
import torch
import torch.distributed as dist

import utils

from torch import nn, Tensor
import numpy as np
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import *
from interaction_head import InteractionHead

import sys
sys.path.append('detr')
from detr.models import build_model
from detr.util import box_ops


def build_detr(args): 
    net, criterion, postprocessors = build_model(args)
    num_channels = net.backbone[0].num_channels 
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')      
        net = utils.load_model(net, checkpoint['model_state_dict'])   
    return net, criterion, postprocessors, num_channels

def build_detector(args, class_corr, tb_writer=None):
    #   读取detr模型
    net, criterion, postprocessors, num_channels = build_detr(args)
    predictor = torch.nn.Linear(args.repr_dim, args.num_classes) 
    num_channels = net.backbone[0].num_channels    
    interaction_head = InteractionHead(
                        predictor, 
                        num_channels,       # (2048)
                        class_corr,
                        args=args,
                        tb_writer = tb_writer)
    #   UPT模型
    detector = GFIN(net, postprocessors, interaction_head, args = args)
    return detector


class GFIN(nn.Module):
    def __init__(self,
        detector: nn.Module,
        postprocessor: nn.Module,
        interaction_head: nn.Module,
        args,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.interaction_head = interaction_head
        self.human_idx=args.human_idx 
        self.num_classes=args.num_classes
        self.alpha=args.alpha 
        self.gamma=args.gamma
        self.box_score_thresh=args.box_score_thresh
        self.fg_iou_thresh=args.fg_iou_thresh
        self.min_instances=args.min_instances
        self.max_instances=args.max_instances
        self.dataset = args.dataset
        self.cascade_layer = args.cascade_layer
        self.dec_layers = args.dec_layers
        self.aux_attn_loss = args.aux_attn_loss,
        self.label_match = args.label_match
        self.Hungarian = args.Hungarian
        self.num_layer=args.num_layer
        self.no_nms = args.no_nms



    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        self.device = images[0].device
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        #   2048, 256(DETR部分，可以变成特征文件加速训练)
        original_image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=self.device)
        with torch.no_grad():
            self.detector.eval()
            no_prop_srcs, srcs, masks, poses, outputs = self.DETR_forward(images)
            detections, features, detr_memory = self.mutil_layer_process(outputs, original_image_sizes)
            region_props = self.prepare_region_proposals(detections, features)
            sample_wh = torch.tensor([max(original_image_sizes[:, 0]), max(original_image_sizes[:, 1])], device=self.device)
          
        logits, prior, prior_score, bh, bo, objects, attn_maps = \
            self.interaction_head(images, no_prop_srcs, srcs, 
                                  detr_memory, original_image_sizes, sample_wh, 
                                  region_props, masks, poses)
        boxes = [r['boxes'] for r in region_props]
        location = [r['location'] for r in region_props]
        od_labels = [r['labels'] for r in region_props]
        od_scores = [r['scores'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, 
                                                             od_labels, targets, 
                                                            attn_maps=attn_maps)
            loss_dict = dict(interaction_loss=interaction_loss)
            return loss_dict
        else:
            detections = self.postprocessing(boxes, bh, bo, logits, 
                                             prior, od_labels, od_scores,
                                             objects,  attn_maps, original_image_sizes, location)
            return detections
        
    def DETR_forward(self, images):    
        #   2048, 256(DETR部分，可以变成特征文件加速训练)
        #   同DETR的forward
        no_prop_srcs, srcs, masks, poses = self.detector.backbone_forward(images)
        out = self.detector.neck_forward(srcs, masks, poses, self.detector.query_embed.weight) # features, memory
        outputs = self.detector.head_forward(out)
        outputs.update(out) 
        return no_prop_srcs, srcs, masks, poses, outputs

    def mutil_layer_process(self, outputs, original_image_sizes):
        num_batch = len(original_image_sizes)
        #   取出每一个decoder的结果
        detections = [dict(scores=torch.zeros((0,), dtype=torch.float32, device=self.device), 
                            labels=torch.zeros((0,),  dtype=torch.int32, device=self.device), 
                            boxes=torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                            # pred_logits=torch.zeros((0, 81), dtype=torch.float32, device=outputs['out_query'].device),
                            location=torch.zeros((0, 2), dtype=torch.float32, device=self.device)) for _ in range(num_batch)]
        features = torch.zeros((num_batch, 0, 256), dtype=torch.float32, device=self.device)

        #   级联预测
        for i in range(self.cascade_layer, self.dec_layers):
            results_per_layer = {'pred_logits': outputs['pred_logits_cascade'][i], 
                                 'pred_boxes':  outputs['pred_boxes_cascade'][i]}
            results_per_layer = self.postprocessor['bbox'](results_per_layer, original_image_sizes) 
            num_query = results_per_layer[-1]['scores'].shape[-1]#100       
            location = torch.tensor( [(i, x) for x in range(num_query)], device=self.device)
            
            for bn in range(num_batch):
                detections[bn]['scores'] = torch.cat([detections[bn]['scores'], 
                                                    results_per_layer[bn]['scores']], dim=0)  
                detections[bn]['labels'] = torch.cat([detections[bn]['labels'], 
                                                    results_per_layer[bn]['labels']], dim=0) 
                detections[bn]['boxes'] = torch.cat([detections[bn]['boxes'], 
                                                    results_per_layer[bn]['boxes']], dim=0)
                detections[bn]['location'] = torch.cat([detections[bn]['location'],  location], dim=0)
            features = torch.cat([features, outputs['out_query'][i]], dim=1) 
        #   值的变化不影响结果
        detr_memory = outputs['memory']
        return detections, features, detr_memory  
    
    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            #   100
            sc, lb, bx, loc = res.values()
            if not self.no_nms:
                keep = batched_nms(bx, sc, lb, self.fg_iou_thresh)
                sc = sc[keep].view(-1)
                lb = lb[keep].view(-1)
                bx = bx[keep].view(-1, 4)
                hs = hs[keep].view(-1, 256)
                loc = loc[keep].view(-1, 2)
            #   置信度 > 0.2
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = (lb == self.human_idx)
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(~is_human).squeeze(1)
            n_human = is_human[keep].sum()
            n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval

            #   3-15
            if n_human < self.min_instances:#   如果置信度过小至少要三个人的bbox
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:#   只取前面几个
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]
            #   3-15
            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]                   

            keep = torch.cat([keep_h, keep_o])
            #obj_word_embed = self.obj_word_embedding[lb[keep]]
            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                location=loc[keep],
                hidden_states = hs[keep],
            ))

        return region_props


    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes
    
    def associate_with_ground_truth(self, boxes_h, boxes_o, od_labels, 
                                     targets, data_type='vcoco'):
        hoi_pair_num = boxes_h.shape[0]#==boxes_o.shape[0]
        labels = torch.zeros(hoi_pair_num, self.num_classes, device=self.device)
        
        gt_boxes_h = torch.zeros(hoi_pair_num, 4, device=self.device)
        gt_boxes_o = torch.zeros(hoi_pair_num, 4, device=self.device)
        # gt_label = torch.zeros(hoi_pair_num, dtype=torch.int, device=self.device)
        th = targets['boxes_h']
        to = targets['boxes_o']
        gt_bx_h = self.recover_boxes(th, targets['size'])
        gt_bx_o = self.recover_boxes(to, targets['size'])
        olb = targets['object']
        vlb = targets['labels']
         
        #   人和物的iou都要大于0.5
        iou = torch.min(
            box_iou(boxes_h, gt_bx_h),#pairN, 
            box_iou(boxes_o, gt_bx_o) #pairN, 
        )   
        iou_bool = (iou >= self.fg_iou_thresh) 
        x, y = torch.nonzero(iou_bool).unbind(1)
        xx, yy = x, y
        if self.label_match:
            od_labels = od_labels+1 if data_type=='vcoco' else od_labels
            xx = x[od_labels[x]==olb[y]]
            yy = y[olb[y]==od_labels[x]]

        verb_label = vlb[yy]
        labels[xx, verb_label] = 1
        gt_boxes_h[xx] =  gt_bx_h[yy]
        gt_boxes_o[xx] =  gt_bx_o[yy]
        return labels, gt_boxes_h, gt_boxes_o

    def compute_interaction_loss(self, boxes, bh, bo, 
                                logits_layer, priors,
                                od_labels, targets, 
                                attn_maps=None, topK=20, score_thre=0., alpha = 1., beta=1.): 
        label_list = []
        gt_boxes_h_list = []
        gt_boxes_o_list = []
        pred_boxes_h_list = []
        pred_boxes_o_list = []
     
       
        #   每一个sample
        start_i = 0
        for bx, h, o, lb, target in zip(boxes, bh, bo, od_labels, targets):
            start_i=start_i+h.shape[0]
            labels, gt_boxes_h, gt_boxes_o = \
                self.associate_with_ground_truth(bx[h], bx[o], lb[o],  target, self.dataset)
            label_list.append(labels)#  36, 24
                
            gt_boxes_h_list.append(gt_boxes_h)
            gt_boxes_o_list.append(gt_boxes_o)
            pred_boxes_h_list.append(bx[h])
            pred_boxes_o_list.append(bx[o])
                    
        labels = torch.cat(label_list)
        loss = 0   


        #  (2, N, 24) -> (N, 24)
        priors = torch.cat(priors, dim=1).prod(0)# human_scroe * obj_score       
        #   how many label
        x, y = torch.nonzero(priors).unbind(1)          
        prior = priors[x, y]
        label = labels[x, y]
        
        cls_loss = 0 
        num_layer = logits_layer.shape[0]  
        # num_loss_layer = self.num_layer 
        num_loss_layer = num_layer         
        for _, logits in enumerate(logits_layer[num_layer-num_loss_layer : num_layer]):
            logits_sigmoid = torch.sigmoid(logits)
            logit = logits_sigmoid[x, y]    #   [n,verb_num]     
            score = logit * prior 
            cls_loss += binary_focal_loss(
                score, label, reduction='sum',
                alpha=self.alpha, gamma=self.gamma 
            )                                    
        cls_loss = cls_loss/num_loss_layer
        n_p = len(torch.nonzero(label))
        if dist.is_initialized() and self.device.type!="cpu":
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device=self.device)
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        cls_loss = cls_loss / n_p   
        loss += beta * cls_loss 
        
        return loss

    def postprocessing(self, boxes, bh, bo, 
                    logits, prior, 
                    od_labels, od_scores,
                    objects, attn_maps, 
                    image_sizes, location):
        if logits.dim()==3:
            logits = logits[-1]     
        n = [len(b) for b in bh]
        logits = logits.split(n)

        detections = []
        #   every sample
        for bx, h, o,  lg, pr, odl, ods, obj, attn, size, loc in zip(
            boxes, bh, bo, logits, prior, od_labels, od_scores, objects, 
            attn_maps, image_sizes, location):
           #  (2, N, 24) -> (N, 24)
            priors = pr.prod(0)
            lgs = torch.sigmoid(lg)
            scores = lgs * priors

            x, y = torch.nonzero(priors).unbind(1)
       
            det = dict(boxes=bx, od_labels = odl, od_scores = ods, 
                    pairing=torch.stack([h[x], o[x]]), size=size,
                    scores= scores[x, y], labels=y, objects=obj[x], 
                    attn_maps=attn, hoi_dec_location=x, od_dec_location=loc,)
            detections.append(det)
        return detections
    
