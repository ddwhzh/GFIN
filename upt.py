"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import torch.distributed as dist

import utils

from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits
from upt_interaction_head import InteractionHead

import sys
sys.path.append('detr')
from detr.models import build_model
from detr.util import box_ops

def build_detector(args, class_corr):
    detr, _, postprocessors = build_model(args)
    
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        detr = utils.load_model(detr, checkpoint['model_state_dict'])
    else: 
        print("no args.pretrained model")
        # if detr.training:
        #     raise Exception("no args.pretrained model")
        
    predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)
    interaction_head = InteractionHead(
        predictor, args.hidden_dim, args.repr_dim,
        detr.backbone[0].num_channels[-1],
        args.num_classes, args.human_idx, class_corr, args
    )
    detector = UPT(
        detr, postprocessors['bbox'], interaction_head,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        dataset = args.dataset,
    )
    return detector


class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        detector: nn.Module,
        postprocessor: nn.Module,
        interaction_head: nn.Module,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15, dataset='vcoco',
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.interaction_head = interaction_head

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.dataset = dataset

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        with torch.no_grad():
            image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)
            sample_wh = torch.tensor([max(image_sizes[:, 0]), max(image_sizes[:,1])], device = images[0].device)
            
            no_prop_srcs, srcs, masks, poses = self.detector.backbone_forward(images)
            outputs = self.detector.neck_forward(srcs, masks, poses, self.detector.query_embed.weight) # hs, memory
            out = self.detector.head_forward(outputs)
            out.update(outputs)    

            num_batch = outputs['out_query'].shape[1]
            results = [dict(scores=torch.zeros((0,), dtype=torch.float32, device=images[0].device), 
                            labels=torch.zeros((0,),  dtype=torch.int32, device=images[0].device), 
                            boxes=torch.zeros((0, 4), dtype=torch.float32, device=images[0].device),
                            #pred_logits=torch.zeros((0, 81), dtype=torch.float32, device=outputs['out_query'].device),
                            location=torch.zeros((0, 2), dtype=torch.float32, device=images[0].device)) for _ in range(num_batch)]
            hs = torch.zeros((num_batch, 0, 256), dtype=torch.float32, device=images[0].device)
            
            
            #   级联预测
            for i in range(5, 6):
                results_per_layer = {'pred_logits': out['pred_logits_cascade'][i], 
                                     'pred_boxes':  out['pred_boxes_cascade'][i]}
                post_results_per_layer = self.postprocessor(results_per_layer, image_sizes) 
                num_query = post_results_per_layer[-1]['scores'].shape[-1]#100       
                location = torch.tensor( [(i, x) for x in range(num_query)], device=outputs['out_query'].device)
                
                for bn in range(num_batch):
                    results[bn]['scores'] = torch.cat([results[bn]['scores'], 
                                                      post_results_per_layer[bn]['scores']], dim=0)  
                    results[bn]['labels'] = torch.cat([results[bn]['labels'], 
                                                      post_results_per_layer[bn]['labels']], dim=0) 
                    results[bn]['boxes'] = torch.cat([results[bn]['boxes'], 
                                                     post_results_per_layer[bn]['boxes']], dim=0)
                    results[bn]['location'] = torch.cat([results[bn]['location'],  location], dim=0)
                hs = torch.cat([hs, outputs['out_query'][i]], dim=1) 
            region_props = self.prepare_region_proposals(results, hs)


        logits, prior, \
        bh, bo, objects, attn_maps \
            = self.interaction_head(no_prop_srcs[-1], image_sizes, region_props)
        boxes = [r['boxes'] for r in region_props]
        od_labels = [r['labels'] for r in region_props]
        od_scores = [r['scores'] for r in region_props]
        
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, 
                                                             od_labels, targets)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict

        detections = self.postprocessing(boxes, bh, bo, logits, 
                                         prior, od_labels, od_scores,
                                         objects, attn_maps, image_sizes, location)
        return detections


    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, od_labels, 
                                     targets, data_type='vcoco'):
        hoi_pair_num = boxes_h.shape[0]#==boxes_o.shape[0]
        labels = torch.zeros(hoi_pair_num, self.num_classes, device=boxes_h.device)
        
        gt_boxes_h = torch.zeros(hoi_pair_num, 4, device=boxes_h.device)
        gt_boxes_o = torch.zeros(hoi_pair_num, 4, device=boxes_h.device)
        # gt_label = torch.zeros(hoi_pair_num, dtype=torch.int, device=boxes_h.device)
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
        od_labels = od_labels+1 if data_type=='vcoco' else od_labels
        xx = x[od_labels[x]==olb[y]]
        yy = y[olb[y]==od_labels[x]]

        verb_label = vlb[yy]
        labels[xx, verb_label] = 1
        # if (od_labels[xx] != olb[yy]).any():
        #     raise ValueError("od_labels[xx]!=tlb[yy]")
        gt_boxes_h[xx] =  gt_bx_h[yy]
        gt_boxes_o[xx] =  gt_bx_o[yy]
        # gt_label[xx] =  olb[yy]
        return labels, gt_boxes_h, gt_boxes_o

    def compute_interaction_loss(self, boxes, bh, bo, 
                                 logits, priors, 
                                 od_labels, targets):

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


        priors = torch.cat(priors, dim=1).prod(0)
        x, y = torch.nonzero(priors).unbind(1)
        logit = logits[x, y]
        prior = priors[x, y]
        label = labels[x, y]


        loss = binary_focal_loss_with_logits(
            torch.log(prior / (1 + torch.exp(-logit) - prior) + 1e-8), 
            label, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        n_p = len(torch.nonzero(label))
        if dist.is_initialized() and label.device.type!="cpu":
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device=label.device)
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            #   100
            sc, lb, bx, loc = res.values()
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

    def postprocessing(self, boxes, bh, bo, 
                       logits, prior, 
                       od_labels, od_scores,
                       objects, attn_maps, 
                       image_sizes, location):
        n = [len(b) for b in bh]
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, odl, ods, obj, attn, size, loc in zip(
            boxes, bh, bo, logits, prior, od_labels, od_scores, objects, attn_maps, image_sizes, 
        location):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y, od_labels = odl, od_scores = ods, 
                objects=obj[x], attn_maps=attn, size=size,
                hoi_dec_location=x, od_dec_location=loc,
            ))

        return detections


