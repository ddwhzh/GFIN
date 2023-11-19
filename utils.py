"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
from regex import S
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

import time
import torch
import multiprocessing

from torch import Tensor
from collections import deque
from typing import Optional, Iterable, Any, List, Union, Tuple



from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from vcoco.vsrl_eval import VCOCOeval
import time
import itertools
import torch.distributed as dist

import sys
sys.path.append('detr')
import detr.datasets.transforms as T
from add_on.pnms import *
def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(os.getcwd(), data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(os.getcwd(), data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                istest = 'test' in partition
            )
        elif name == 'vcoco':
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(os.getcwd(), data_root, image_dir[partition]),
                #anno_file=os.path.join(data_root,'origin_anno', 'instances_vcoco_{}.json'.format(partition)), 
                anno_file=os.path.join(os.getcwd(), data_root, 'instances_vcoco_{}.json'.format(partition)), 
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                istest = 'test' in partition
            )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
        ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name
        self.istest = 'test' in partition
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if target!=None:
            if self.name == 'hicodet':
                target['labels'] = target['verb']
                # Convert ground truth boxes to zero-based index and the
                # representation from pixel indices to coordinates
                target['boxes_h'][:, :2] -= 1
                target['boxes_o'][:, :2] -= 1
            else:
                target['labels'] = target['actions']
                target['object'] = target.pop('objects')

        image, target = self.transforms(image, target)

        return image, target

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]


class NewDetectionAPMeter(DetectionAPMeter):
    def __init__(self, num_cls: int, num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC', nproc: int = 20,
            precision: int = 64,
            output: Optional[List[Tensor]] = None,
            labels: Optional[List[Tensor]] = None,
            ko_mode=False) -> None:
        super().__init__(num_cls, num_gt, algorithm, nproc, precision,output, labels)
        self.file_name_to_obj_cat = json.load(open(os.path.join('hicodet/file_name_to_obj_cat.json'), "r"))
        hoi_pair_list = json.load(open(os.path.join('hicodet/hoi_list_new.json'), "r"))#相当于过滤矩阵
        self.hoi_pair_dict = dict()
        for hoi in hoi_pair_list:
            self.hoi_pair_dict[int(hoi['id'])] = hoi
        self.ko_mode = ko_mode

    def append(self, output: Tensor, prediction: Tensor, labels: Tensor, filenames) -> None:
        if isinstance(output, torch.Tensor) and \
                isinstance(prediction, torch.Tensor) and \
                isinstance(labels, torch.Tensor):
            prediction = prediction.long()
            unique_cls = prediction.unique()

            for cls_idx in unique_cls:
                if self.ko_mode:
                    if filenames not in self.file_name_to_obj_cat:#只统计有图片的
                        continue
                    obj_cats = self.file_name_to_obj_cat[filenames]
                    if self.hoi_pair_dict[int(cls_idx)]['object_cat'] not in obj_cats:#只统计交互对象正确的
                        continue
                sample_idx = torch.nonzero(prediction == cls_idx).squeeze(1)
                self._output_temp[cls_idx] += output[sample_idx].tolist()
                self._labels_temp[cls_idx] += labels[sample_idx].tolist()
        else:
            raise TypeError("Arguments should be torch.Tensor")


    def eval(self):
        self._output = [torch.cat([out1, torch.as_tensor(out2, dtype=self._dtype)]) 
                        for out1, out2 in zip(self._output, self._output_temp)]
        self._labels = [torch.cat([tar1, torch.as_tensor(tar2, dtype=self._dtype)]) 
                        for tar1, tar2 in zip(self._labels, self._labels_temp)]
        self.reset(keep_old=True)
                
        self.ap, self.max_rec = self.compute_ap(self._output, self._labels, self.num_gt,
                                                nproc=self._nproc, algorithm=self.algorithm)

        return self.ap, self.max_rec


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, test_loader=None, trainset=None, dataset=None, tb_writer=None,
                 max_norm=0, num_classes=117, start_epoch=0, iteration=0, pnms=False, world_size=4, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        
        self._test_loader =test_loader
        self._trainset = trainset
        self.dataset =  dataset
        self.tb_writer = tb_writer
        self._state.epoch = start_epoch
        self._state.iteration = iteration
        self.pnms = pnms
        self.world_size = world_size
        
    def __call__(self, startn, n: int) -> None:
        self.start_epochs = startn
        self.epochs = n
        # Train for a specified number of epochs
        self._on_start()
        for _ in range(startn, n):
            self._on_start_epoch()
            timestamp = time.time()
            for batch in self._train_loader:
                self._state.inputs = batch[:-1]
                # for target in batch[-1]:
                #     if 'file_name' in target:target.pop('file_name')
                #     if 'filenames' in target:target.pop('filenames')
                self._state.targets = batch[-1]
                self._on_start_iteration()
                self._state.t_data.append(time.time() - timestamp)

                self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                self._state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()
                
            self._on_end_epoch()
        self._on_end()

    def _on_each_iteration(self):
        loss_dict = self._state.net(*self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")
        #   实际上只做了interaction_loss的loss，没有匈牙利匹配计算目标检测loss
        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()


    def save_checkpoint(self) -> None:
        """Save a checkpoint of the model state"""
        checkpoint = {
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': self._state.net.module.state_dict(),
            'optim_state_dict': self._state.optimizer.state_dict(),
            'scaler_state_dict': self._state.scaler.state_dict()
        }
        if self._state.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
        # Cache the checkpoint
        print('ckpt_{:02d}.pt'.format(self._state.epoch))
        torch.save(checkpoint, os.path.join(
            self._cache_dir,
            'ckpt_{:02d}.pt'.format(self._state.epoch)
        ))


    def _on_end_epoch(self):
        # Save checkpoint in the master process
        if self._rank == 0:
            self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
        #torch.cuda.empty_cache()
        return
    
        if self._state.epoch % 4 !=0 and self._state.epoch != 1 and self._state.epoch != self.epochs-1:
            return
        print('test start')
        if self.dataset == 'vcoco':
            self.cache_vcoco(self._test_loader, self._cache_dir)
            if self._rank == 0:
                vcocoeval = VCOCOeval(
                                    vsrl_annot_file='/WHZH/dataset/v-coco/data/vcoco/vcoco_test.json', 
                                    coco_annot_file='/WHZH/dataset/v-coco/data/instances_vcoco_all_2014.json', 
                                    split_file='/WHZH/dataset/v-coco/data/splits/vcoco_test.ids',
                                    #vsrl_annot_file='/WHZH/dataset/v-coco/data/vcoco/vcoco_trainval.json', 
                                    #coco_annot_file='/WHZH/dataset/v-coco/data/instances_vcoco_all_2014.json', 
                                    #split_file='/WHZH/dataset/v-coco/data/splits/vcoco_trainval.ids',
                                    )
                vcocoeval._do_eval(self._cache_dir+"/cache.pkl", ovr_thresh=0.5)
            torch.distributed.barrier()
        elif self.dataset == 'hicodet':
            ap, recall = self.test_hico(self._test_loader)
            # Fetch indices for rare and non-rare classes
            num_anno = torch.as_tensor(self._trainset.dataset.anno_interaction)
            rare = torch.nonzero(num_anno < 10).squeeze(1)
            non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
            print(
                f"The mAP is {ap.mean():.4f},"
                f" rare: {ap[rare].mean():.4f},"
                f" none-rare: {ap[non_rare].mean():.4f},"
                f" recall: {recall.mean():.4f}",
                f" rare recall: {recall[rare].mean():.4f},"
                f" none-rare recall: {recall[non_rare].mean():.4f}"
            )
            if self._rank == 0:
                self.tb_writer.add_scalar(f'mAP', ap.mean(), global_step=self._state.epoch, walltime=None)  
                self.tb_writer.add_scalar(f'rare', ap[rare].mean(), global_step=self._state.epoch, walltime=None)  
                self.tb_writer.add_scalar(f'none-rare', ap[non_rare].mean(), global_step=self._state.epoch, walltime=None)   
                self.tb_writer.add_scalar(f'recall', recall.mean(), global_step=self._state.epoch, walltime=None)  
                self.tb_writer.add_scalar(f'rare recall', recall[rare].mean(), global_step=self._state.epoch, walltime=None)  
                self.tb_writer.add_scalar(f'none-rare recall', recall[non_rare].mean(), global_step=self._state.epoch, walltime=None)               
            torch.distributed.barrier()


    @torch.no_grad()
    def test_hico(self, dataloader, ko_mode = False, device='cuda'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(dataset.object_n_verb_to_interaction, dtype=float))

        meter = NewDetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P',
            ko_mode = ko_mode)
        all_results = []
        preds = []
        for batch in dataloader:
            target = batch[-1][0]
            inputs = pocket.ops.relocate_to_cuda(batch[0])  if device=='cuda' else pocket.ops.relocate_to_cpu(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)

            # Format detections
            boxes = output['boxes']
            pairing = output['pairing']
            boxes_h, boxes_o = boxes[pairing].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            ##########################################
            #
            #   PNMS
            #
            ##########################################
            if self.pnms:
                boxes_h, boxes_o, objects, scores, verbs, pairing = \
                triplet_nms_filter(boxes_h, boxes_o, objects, scores, verbs, pairing)
            
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate((gt_bx_h[gt_idx].view(-1, 4), gt_bx_o[gt_idx].view(-1, 4)),
                                        (boxes_h[det_idx].view(-1, 4), boxes_o[det_idx].view(-1, 4)), scores[det_idx].view(-1))               
            result = dict(scores=scores, interactions=interactions, labels = labels, filenames = target['filenames'])
            all_results.append(result)
            # meter.append(scores, interactions, labels)
            
        torch.distributed.barrier()        
        preds.extend(list(itertools.chain.from_iterable(all_gather(all_results))))
        if self._rank==0:
            for pred in preds:
                scores = pred['scores']
                interactions = pred['interactions']
                labels = pred['labels']
                filenames = pred['filenames']
                meter.append(scores, interactions, labels, filenames)
        return meter.eval()

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache', device='cuda'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        preds = []
        for _, batch in enumerate(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])if device=='cuda' else pocket.ops.relocate_to_cpu(batch[0])
            output = net(inputs)
            #   TODO: 增加可视化
            #   boxes       [num_obj, 4]
            #   pairing:    [num_pair, 4]
            #   scores :    [num_pair]
            #   labels :    [num_pair]
            #   objects:    [num_pair]
            #   attn_maps:
            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            #image_id = dataset.image_id(i)
            target = batch[-1]
            file_name = target[0]['file_name']
            image_id = int(file_name.split('_')[-1].split('.')[0])
            # Format detections
            boxes = output['boxes']
            pairing = output['pairing']
            boxes_h, boxes_o = boxes[pairing].unbind(0)
            for j, _ in enumerate(boxes_o):
                if (boxes_o[j] == boxes_h[j]).all():
                    boxes_o[j] *= 0
            scores = output['scores']
            objects = output['objects']
            verbs = output['labels']
            ##########################################
            #
            #   PNMS
            #
            ##########################################
            if self.pnms:
                boxes_h, boxes_o, objects, scores, verbs, pairing = \
                    triplet_nms_filter(boxes_h, boxes_o, objects, scores, verbs, pairing)
            # Rescale the boxes to original image size
            #ow, oh = dataset.image_size(i)
            ow, oh = dataset.load_image(os.path.join(dataset._root, file_name)).size
            h, w = output['size']
            scale_fct = torch.as_tensor([ow / w, oh / h, ow / w, oh / h]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            #每一个交互
            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, verbs):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)
                

        torch.distributed.barrier()        
        preds.extend(list(itertools.chain.from_iterable(all_gather(all_results))))
        if self._rank==0:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
                # Use protocol 2 for compatibility with Python2
                pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)


    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab', device ='cuda'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)
        
        pred = []
        preds = []
        for i, batch in enumerate(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0]) if device=='cuda' else pocket.ops.relocate_to_cpu(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            #image_idx = dataset._idx[i]
            target = batch[-1]
            file_name = target[0]['filenames']
            image_idx = int(file_name.split('_')[-1].split('.')[0])
            # Format detections
            boxes = output['boxes']
            pairing = output['pairing']
            boxes_h, boxes_o = boxes[pairing].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            ##########################################
            #
            #   PNMS
            #
            ##########################################
            if self.pnms:
                boxes_h, boxes_o, objects, scores, verbs, pairing = \
                triplet_nms_filter(boxes_h, boxes_o, objects, scores, verbs, pairing)
            
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            #ow, oh = dataset.image_size(i)
            ow, oh = dataset.load_image(os.path.join(dataset._root, file_name)).size
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, 
                ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
            pred.append(all_results)

        torch.distributed.barrier()        
        preds.extend(list(itertools.chain.from_iterable(all_gather(pred))))
        if self._rank==0:
            all_results = np.concatenate(preds)
            # Replace None with size (0,0) arrays
            for i in range(600):
                for j in range(all_results.shape[-1]):
                    if all_results[i, j] is None:
                        all_results[i, j] = np.zeros((0, 0))


            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # Cache results
            for object_idx in range(80):
                interaction_idx = object2int[object_idx]
                sio.savemat(
                    os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                    dict(all_boxes=all_results[interaction_idx])
                )

def load_model(model, state_dict):
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
            # else:
            #    print('loading parameter {}.'.format(k))
        else:
            if 'total_params' not in k and 'total_ops' not in k:
                #pass
                print('Drop parameter {}.'.format(k))
    print('#####    No param session  #####')
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False) 
    return model


def load_optim(optim, state_dict):
    optim_state_dict = optim.state_dict()
    for k in optim_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = optim_state_dict[k]
    optim.load_state_dict(state_dict) 
    return optim


def load_encoder_model(model, state_dict):
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    t_index = []
    for k in state_dict:
        k_r = k.replace('transformer.encoder.','')
        if k_r in model_state_dict:   
            if state_dict[k].shape != model_state_dict[k_r].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(k, model_state_dict[k_r].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k_r]
            else:
               print('loading parameter {}.'.format(k))
               t_index.append(k)
        else:
            if 'total_params' not in k and 'total_ops' not in k:
                pass
                #print('Drop parameter {}.'.format(k))
    for k in t_index:
        k_r = k.replace('transformer.encoder.','')
        state_dict[k_r] = state_dict[k]                
    # print('#####    No param session  #####')
    # for k in model_state_dict:
    #     k_r = "transformer.encoder."+k
    #     if not (k_r in state_dict):
    #         # print('No param {}.'.format(k))
    #         state_dict[k_r] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False) 
    return model


def load_decoder_model(model, state_dict):
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    t_index = []
    t_temp = dict()
    for k in state_dict:
        k_r = k.replace('transformer.decoder.','')
        if k_r in model_state_dict:   
            if state_dict[k].shape != model_state_dict[k_r].shape:
                temp = state_dict[k]
                if model_state_dict[k_r].shape[0] == state_dict[k].shape[0]*2:
                    temp = torch.cat([temp, temp], 0)
                if len(model_state_dict[k_r].shape)>1 and model_state_dict[k_r].shape[-1] == state_dict[k].shape[-1]*2:
                    temp = torch.cat([temp, temp], -1)
                    
                if temp.shape != model_state_dict[k_r].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(k, model_state_dict[k_r].shape, temp.shape))
                    state_dict[k] = model_state_dict[k_r]
                else:
                    print('loading parameter {}.'.format(k))
                    t_index.append(k)
                    t_temp[k] = temp
            else:
               print('loading parameter {}.'.format(k))
        else:
            if 'total_params' not in k and 'total_ops' not in k:
                pass
                #print('Drop parameter {}.'.format(k))

    for k in t_index:
        k_r = k.replace('transformer.decoder.','')
        state_dict[k_r] = t_temp[k]             

    # print('#####    No param session  #####')
    # for k in model_state_dict:
    #     k_r = "transformer.decoder."+k
    #     if not (k_r in state_dict):
    #         # print('No param {}.'.format(k))
    #         state_dict[k_r] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False) 
    return model


def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    device = dist.get_rank() 
    
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
