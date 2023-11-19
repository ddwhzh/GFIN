import torch
import torch.nn.functional as F
from torch import nn
import copy
import numpy as np

from ...util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ...util.misc import (nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.                              
    The process happens in two steps:
        1) 我们计算GT box和模型输出之间的hungarian assignment。
        2) 我们对每一对匹配的GT/prediction 进行监督（监督类和box）
    """ 
    def __init__(self, args, matcher, weight_dict, losses, focal_alpha=0.25, loss_type='ce_loss'):
        """ 
        Create the criterion.
        Parameters:
            num_classes: 对象类别的数量，省略special no-object category
            matcher: 能够计算targets和proposals之间的matching的模块。
            weight_dict: dict，key为loss的名称，value为其relative weight
            eos_coef: no-object category 的relative classification weight
            losses: 所有要应用的list of losses。可用的损失列表见get_loss。
        """
        super().__init__()
        self.args = args
        self.matcher = matcher              #   对预测与GT进行匹配的算法
        self.losses = losses                #   指定需要计算哪些loss('labels','boxes','cardinality','masks')
        self.num_obj_classes = args.num_obj_classes  #   类别数，不包含背景
        self.num_queries = args.num_queries
        self.eos_coef = args.eos_coef        #   针对背景分类的loss权重
        self.focal_alpha = focal_alpha
        #   将这部分注册到buffer，能够被state_dict记录同时不会有梯度传播到此处
        self.loss_type = loss_type
        empty_weight = torch.ones(self.num_obj_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.weight_dict = weight_dict


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() 
                               if k not in ['aux_outputs', 'enc_outputs', 'backbone_outputs', 'mask_flatten']}
                               #if k != 'aux_outputs'}

        # 检索最后一层的输出与目标之间的匹配情况
        #    list[batch_size] tuple(2)  tensor[num_obj][num_obj]
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点的平均target boxes数，以实现标准化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # 计算所有requested的loss
        losses = {}
        for loss in self.losses:
            #   TODO: target有问题list -> torch
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # 在auxiliary loss的情况下，我们对每个中间层的输出重复这个过程。
        if 'aux_outputs' in outputs:
            losses = self.aux_outputs_loss(outputs, targets, num_boxes, losses)

        if 'backbone_outputs' in outputs:
            losses = self.backbone_outputs_loss(outputs, targets, num_boxes, losses)
                    
        if 'enc_outputs' in outputs:
            losses = self.enc_outputs_loss(outputs, targets, num_boxes, losses)
                
        if 'aux_outputs_enc' in outputs:
            losses = self.aux_outputs_enc_loss(outputs, targets, num_boxes, losses)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            "mask_prediction": self.loss_mask_prediction,
            "corr": self.corr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
            Classification loss (NLL)
            targets dicts必须包含一个包含dim [nb_target_boxes] 的张量的 "label "键
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o        
        if self.loss_type=='ce_loss':
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), 
                                      target_classes, self.empty_weight)
        elif self.loss_type=='focal_loss':
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ 
            计算基数cardinality error，即predicted non-empty boxes数量的绝对误差。
            这并不是真正的loss，它的目的只是为了记录。它不会传播梯度
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
           计算与边界框相关的损失，L1回归loss和GIoU loss 
           targets dicts必须包含[nb_target_boxes，4] "boxes"，
           目标框的预期格式为(center_x, center_y, w, h)，按图像大小归一化。
        """
        #   list[batch_size]   tuple(2)    tensor[num_obj]
        #   indices  
        assert 'pred_boxes' in outputs
        #   tuple(2)    [batchsize * num_objs]
        idx = self._get_src_permutation_idx(indices)
        #   [batchsize * num_objs]
        src_boxes = outputs['pred_boxes'][idx]
        #   [batchsize * num_objs]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        bbox1 = box_cxcywh_to_xyxy(src_boxes)
        bbox2 = box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - torch.diag(generalized_box_iou(bbox1, bbox2))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
            计算与masks有关的loss：focal loss和dice loss。
            targets dicts必须包含一个包含[nb_target_boxes, h, w]的张量的键 "masks"。 
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO 使用valid来掩盖由于损失中的padding而导致的无效区域                                          
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()

        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # 将预测值提高到目标尺寸
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
            }
    
        return losses

    #   sparse_detr
    def loss_mask_prediction(self, outputs, targets, indices, num_boxes, layer=None):
        assert "backbone_mask_prediction" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        mask_prediction = outputs["backbone_mask_prediction"] 
        loss_key = "loss_mask_prediction"

        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))

        losses = {}

        if 'mask_flatten' in outputs:
            flat_grid_attn_map_dec = flat_grid_attn_map_dec.masked_fill(
                outputs['mask_flatten'], flat_grid_attn_map_dec.min()-1)
                
        sparse_token_nums = outputs["sparse_token_nums"]
        num_topk = sparse_token_nums.max()

        topk_idx_tgt = torch.topk(flat_grid_attn_map_dec, num_topk)[1]
        target = torch.zeros_like(mask_prediction)
        for i in range(target.shape[0]):
            target[i].scatter_(0, topk_idx_tgt[i][:sparse_token_nums[i]], 1)

        losses.update({loss_key: F.multilabel_soft_margin_loss(mask_prediction, target)})

        return losses

    #   sparse_detr
    @torch.no_grad()
    def corr(self, outputs, targets, indices, num_boxes):
        if "backbone_topk_proposals" not in outputs.keys():
            return {}

        assert "backbone_topk_proposals" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        backbone_topk_proposals = outputs["backbone_topk_proposals"]
        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_topk = idx_to_flat_grid(spatial_shapes, backbone_topk_proposals)
        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))
        corr = compute_corr(flat_grid_topk, flat_grid_attn_map_dec, spatial_shapes)

        losses = {}
        losses["corr_mask_attn_map_dec_all"] = corr[0].mean()
        for i, _corr in enumerate(corr[1:]):
            losses[f"corr_mask_attn_map_dec_{i}"] = _corr.mean()
        return losses

    
    def _get_src_permutation_idx(self, indices):
        # 对以下指数进行移位预测
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 对以下指数进行排列组合
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def aux_outputs_loss(self, outputs, targets, num_boxes, losses):
        for i, aux_outputs in enumerate(outputs['aux_outputs']):
            indices = self.matcher(aux_outputs, targets)
            for loss in self.losses:
                if loss in ['masks', "mask_prediction", "corr"]:
                    # intermediate masks计算成本太高，我们忽略
                    continue
                kwargs = {}
                if loss == 'labels':
                    # 只对最后一层启用日志记录
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    #   sparse_detr
    def backbone_outputs_loss(self, outputs, targets, num_boxes, losses):
        backbone_outputs = outputs['backbone_outputs']
        bin_targets = copy.deepcopy(targets)
        if not self.eff_specific_head:
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])  # all labels are zero (meaning foreground)
        indices = self.matcher(backbone_outputs, bin_targets)
        for loss in self.losses:
            if loss in ['masks', "mask_prediction", "corr"]:
                # Intermediate masks losses are too costly to compute, we ignore them.
                continue
            kwargs = {}
            if loss == 'labels':
                # Logging is enabled only for the last layer
                kwargs['log'] = False
            l_dict = self.get_loss(loss, backbone_outputs, bin_targets, indices, num_boxes, **kwargs)
            l_dict = {k + f'_backbone': v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses

    def enc_outputs_loss(self, outputs, targets, num_boxes, losses):
        enc_outputs = outputs['enc_outputs']
        bin_targets = copy.deepcopy(targets)
        if not self.eff_specific_head:
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])  # all labels are zero (meaning foreground)
        indices = self.matcher(enc_outputs, bin_targets)
        for loss in self.losses:
            if loss in ['masks', "mask_prediction", "corr"]:
                # Intermediate masks losses are too costly to compute, we ignore them.
                continue
            kwargs = {}
            if loss == 'labels':
                # Logging is enabled only for the last layer
                kwargs['log'] = False
            l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
            l_dict = {k + f'_enc': v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses

    def aux_outputs_enc_loss(self, outputs, targets, num_boxes, losses):
        for i, aux_outputs in enumerate(outputs['aux_outputs_enc']):
            indices = self.matcher(aux_outputs, targets)
            for loss in self.losses:
                if loss in ['masks', "mask_prediction", "corr"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
    




def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


#   sparse_detr
def idx_to_flat_grid(spatial_shapes, idx):
    flat_grid_shape = (idx.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, device=idx.device, dtype=torch.float32)
    flat_grid.scatter_(1, idx.to(torch.int64), 1)

    return flat_grid


def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, _, n_heads, *_ = sampling_locations.shape
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels, 2]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels]

    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # hw -> wh (xy)
    col_row_float = sampling_locations * rev_spatial_shapes

    col_row_ll = col_row_float.floor().to(torch.int64)
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
    col_row_hh = col_row_ll + 1

    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        idx = (idx * valid_mask).flatten(1, 2)
        weights = (attention_weights * valid_mask * margin).flatten(1)
        flat_grid.scatter_add_(1, idx, weights)

    return flat_grid.reshape(N, n_layers, n_heads, -1)


def compute_corr(flat_grid_topk, flat_grid_attn_map, spatial_shapes):
    if len(flat_grid_topk.shape) == 1:
        flat_grid_topk = flat_grid_topk.unsqueeze(0)
        flat_grid_attn_map = flat_grid_attn_map.unsqueeze(0)
        
    tot = flat_grid_attn_map.sum(-1)
    hit = (flat_grid_topk * flat_grid_attn_map).sum(-1)

    corr = [hit / tot]
    flat_grid_idx = 0

    for shape in spatial_shapes:
        level_range = np.arange(int(flat_grid_idx), int(flat_grid_idx + shape[0] * shape[1]))
        tot = (flat_grid_attn_map[:, level_range]).sum(-1)
        hit = (flat_grid_topk[:, level_range] * flat_grid_attn_map[:, level_range]).sum(-1)
        flat_grid_idx += shape[0] * shape[1]
        corr.append(hit / tot)
    return corr

