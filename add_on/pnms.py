import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

@torch.no_grad()
def triplet_nms_filter(boxes_h, boxes_o, od_label, scores, verbs, pairing):

    all_triplets = {}
    #   图里每个对
    hm = 0
    for index, (bh, bo, ol, sc, ac) in enumerate(zip(boxes_h, boxes_o, od_label, scores, verbs)):
        triplet =   str(hm) + '_' +  str(ol) + '_' +  str(ac)
        if triplet not in all_triplets:
            all_triplets[triplet] = {'subs':[], 
                                    'objs':[], 
                                    'scores':[], 
                                    'indexes':[]}
        all_triplets[triplet]['subs'].append(bh.cpu().numpy())
        all_triplets[triplet]['objs'].append(bo.cpu().numpy())
        all_triplets[triplet]['scores'].append(sc.cpu().numpy())
        all_triplets[triplet]['indexes'].append(index)

    all_keep_inds = []
    for triplet, values in all_triplets.items():
        subs, objs, scrs = values['subs'], values['objs'], values['scores']
        keep_inds = pairwise_nms(np.array(subs), np.array(objs), np.array(scrs))

        keep_inds = list(np.array(values['indexes'])[keep_inds])
        all_keep_inds.extend(keep_inds)

    return boxes_h[all_keep_inds], boxes_o[all_keep_inds], \
    od_label[all_keep_inds], scores[all_keep_inds], verbs[all_keep_inds], \
    pairing[:, all_keep_inds]

@torch.no_grad()
def pairwise_nms(subs, objs, scores, 
                 nms_alpha=0.5, nms_beta=1.0, 
                 thres_nms=0.5):
    sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
    ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

    sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
    obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)
    #   TODO: debug
    order = scores.argsort()[::-1]

    keep_inds = []
    while order.size > 0:
        i = order[0]
        keep_inds.append(i)

        sxx1 = np.maximum(sx1[i], sx1[order[1:]])
        syy1 = np.maximum(sy1[i], sy1[order[1:]])
        sxx2 = np.minimum(sx2[i], sx2[order[1:]])
        syy2 = np.minimum(sy2[i], sy2[order[1:]])

        sw = np.maximum(0.0, sxx2 - sxx1 + 1)
        sh = np.maximum(0.0, syy2 - syy1 + 1)
        sub_inter = sw * sh
        sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

        oxx1 = np.maximum(ox1[i], ox1[order[1:]])
        oyy1 = np.maximum(oy1[i], oy1[order[1:]])
        oxx2 = np.minimum(ox2[i], ox2[order[1:]])
        oyy2 = np.minimum(oy2[i], oy2[order[1:]])

        ow = np.maximum(0.0, oxx2 - oxx1 + 1)
        oh = np.maximum(0.0, oyy2 - oyy1 + 1)
        obj_inter = ow * oh
        obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

        ovr = np.power(sub_inter/sub_union, nms_alpha) * np.power(obj_inter / obj_union, nms_beta)
        inds = np.where(ovr <= thres_nms)[0]

        order = order[inds + 1]
    return keep_inds