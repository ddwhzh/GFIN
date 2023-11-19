"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from gfin import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory
import utils

import time
import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter
from vcoco.vsrl_eval import VCOCOeval

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)#20
    parser.add_argument('--epochs', default=20, type=int)#20
    parser.add_argument('--lr-drop', default=10, type=int)#10
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--no_fpn', dest='fpn', action='store_false')
    #parser.add_argument('--fpn', action='store_true')

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    #   new(enc more important)
    #################################
    parser.add_argument('--cascade_layer', default=0, type=int)#cascade
    parser.add_argument('--menc-inter-layer', default=4, type=int)#HOF
    parser.add_argument('--enc-inter-layers', default=4, type=int)#GCE
    parser.add_argument('--dec-inter-layers', default=4, type=int)#PID
    ################################
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    #parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=30, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=4, type=int)#3
    parser.add_argument('--max-instances', default=15, type=int)#15
    #   train
    parser.add_argument('--no_label_match', dest='label_match', action='store_false')
    parser.add_argument('--Hungarian', action='store_true')
    parser.add_argument('--aux_attn_loss', action='store_true')
    parser.add_argument('--num_layer', default=3, type=int)#aux_loss
    
    #   eval
    parser.add_argument('--pnms', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--mode', default='scenario_1', type=str)
    parser.add_argument('--ko_mode',  action='store_true')
    parser.add_argument('--ovr_thresh', default=0.5, type=float)
    parser.add_argument('--ignore_point', action='store_true')
    parser.add_argument('--no_nms', action='store_true')

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--master_addr', default="127.0.0.1", type=str, help='number of distributed processes')
    parser.add_argument('--master_port', default="1333", type=str)

    return parser


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank)

    setup_for_distributed(rank == 0)

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)   
    torch.cuda.manual_seed(seed)
    
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False 
    # torch.backends.cudnn.enabled = True           
    # torch.backends.cudnn.deterministic = True    


    torch.cuda.set_device(rank)

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank))
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        # sampler=torch.utils.data.SequentialSampler(testset),
        sampler=DistributedSampler(
            testset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    
    args.human_idx = 0
    args.num_obj_classes = 81
    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_classes = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        # args.num_classes = 24
        args.num_classes = 29
        
    gfin = build_detector(args, object_to_target)
    
    iteration = 0
    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        gfin = utils.load_model(gfin, checkpoint['model_state_dict'])
        args.start_epoch=checkpoint['epoch'] if 'epoch' in checkpoint else 0
        iteration = checkpoint['iteration'] if 'iteration' in checkpoint else 0
    else:
        # checkpoint =  torch.load(args.pretrained, map_location='cpu')
        # gfin.interaction_head.encoder = utils.load_encoder_model(gfin.interaction_head.encoder, checkpoint['model_state_dict'])
        # print("______________________________________gap________________________________________")
        # gfin.interaction_head.comp_layer = utils.load_decoder_model(gfin.interaction_head.comp_layer, checkpoint['model_state_dict'])
        print(f"=> Rank {rank}: start from a randomly initialised model")

    if rank==0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tb_writer = SummaryWriter(args.output_dir)

    engine = CustomisedDLE(
        gfin, train_loader, 
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        trainset = trainset,
        dataset = args.dataset,
        tb_writer = tb_writer,
        device = args.device if args.device=="cpu" else None,
        start_epoch= args.start_epoch,
        iteration = iteration,
        pnms = args.pnms,
        world_size = args.world_size,
    )
    

    

    print("create engine finished")
    #   测试阶段
    if args.cache:
        print("cache start")
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir, args.device)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir, args.device)
        print("cache finished")
        if rank == 0:
            tb_writer.close()
        return

    if args.eval:
        print("eval start")
        if args.dataset == 'vcoco':
            #raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
            engine.cache_vcoco(test_loader, args.output_dir, args.device)
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            if rank==0:
                vcocoeval = VCOCOeval(vsrl_annot_file='/WHZH/dataset/v-coco/data/vcoco/vcoco_test.json', 
                                      coco_annot_file='/WHZH/dataset/v-coco/data/instances_vcoco_all_2014.json', 
                                      split_file='/WHZH/dataset/v-coco/data/splits/vcoco_test.ids')
                vcocoeval._do_eval(args.output_dir+"/cache.pkl", mode=args.mode, 
                                   ovr_thresh=args.ovr_thresh, ignore_point = args.ignore_point)
        elif args.dataset == 'hicodet':
            ap, recall = engine.test_hico(test_loader, args.ko_mode, args.device)
            # Fetch indices for rare and non-rare classes
            num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
            rare = torch.nonzero(num_anno < 10).squeeze(1)
            non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
            st = "" if not args.ko_mode else "KO"
            print(
                "The "+st+f" mAP is {ap.mean():.4f},"
                f" rare: {ap[rare].mean():.4f},"
                f" none-rare: {ap[non_rare].mean():.4f},"
                f" recall: {recall.mean():.4f},"
                f" rare recall: {recall[rare].mean():.4f},"
                f" none-rare recall: {recall[non_rare].mean():.4f}"
            )
            if rank == 0:
                tb_writer.add_scalar(f'mAP', ap.mean(), global_step=args.epochs, walltime=None)  
                tb_writer.add_scalar(f'rare', ap[rare].mean(), global_step=args.epochs, walltime=None)  
                tb_writer.add_scalar(f'none-rare', ap[non_rare].mean(), global_step=args.epochs, walltime=None)  
                tb_writer.add_scalar(f'recall', recall.mean(), global_step=args.epochs, walltime=None)  
                tb_writer.add_scalar(f'rare recall', recall[rare].mean(), global_step=args.epochs, walltime=None)  
                tb_writer.add_scalar(f'none-rare recall', recall[non_rare].mean(), global_step=args.epochs, walltime=None)  
                tb_writer.close()
        print("eval finished")
        if rank == 0:
            tb_writer.close()
        return

    for p in gfin.detector.parameters():
        p.requires_grad = False
    param_dicts = [{
        "params": [p for n, p in gfin.named_parameters()
        if "interaction_head" in n and p.requires_grad]
    }]
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    if os.path.exists(args.resume):
        #args.lr_head = args.lr_head* 0.1 if args.start_epoch>=args.lr_drop else args.lr_head
        optim = utils.load_optim(optim, checkpoint['optim_state_dict'])
        lr_scheduler = utils.load_optim(lr_scheduler, checkpoint['optim_state_dict'])
    
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    print("start training")
    engine(args.start_epoch, args.epochs)
    # if rank == 0:
    #     tb_writer.close()
    print("end training")

    print("eval start")
    if args.dataset == 'vcoco':
        #raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        engine.cache_vcoco(test_loader, args.output_dir, args.device)
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        if rank==0:
            vcocoeval = VCOCOeval(
                                vsrl_annot_file='/WHZH/dataset/v-coco/data/vcoco/vcoco_test.json', 
                                coco_annot_file='/WHZH/dataset/v-coco/data/instances_vcoco_all_2014.json', 
                                split_file='/WHZH/dataset/v-coco/data/splits/vcoco_test.ids', )
            vcocoeval._do_eval(args.output_dir+"/cache.pkl", mode=args.mode, 
                               ovr_thresh=args.ovr_thresh, ignore_point = args.ignore_point)
    elif args.dataset == 'hicodet':
        ap, recall = engine.test_hico(test_loader, args.ko_mode)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        st = "" if not args.ko_mode else "KO"
        print(
            "The "+st+f" mAP is {ap.mean():.4f},"
            f" rare: {ap[rare].mean():.4f},"
            f" none-rare: {ap[non_rare].mean():.4f},"
            f" recall: {recall.mean():.4f},"
            f" rare recall: {recall[rare].mean():.4f},"
            f" none-rare recall: {recall[non_rare].mean():.4f}"
        )
        if rank == 0:
            tb_writer.add_scalar(f'mAP', ap.mean(), global_step=args.epochs, walltime=None)  
            tb_writer.add_scalar(f'rare', ap[rare].mean(), global_step=args.epochs, walltime=None)  
            tb_writer.add_scalar(f'none-rare', ap[non_rare].mean(), global_step=args.epochs, walltime=None)  
            tb_writer.add_scalar(f'recall', recall.mean(), global_step=args.epochs, walltime=None)  
            tb_writer.add_scalar(f'rare recall', recall[rare].mean(), global_step=args.epochs, walltime=None)  
            tb_writer.add_scalar(f'none-rare recall', recall[non_rare].mean(), global_step=args.epochs, walltime=None)  
            tb_writer.close()
    print("eval finished")
    if rank == 0:
        tb_writer.close()

@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    gfin = build_detector(args, object_to_target)
    if args.eval:
        gfin.eval()

    image, target = dataset[0]
    outputs = gfin([image], [target])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GFIN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.num_layer = args.dec_inter_layers
    print(args)
    args.partitions
    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    start_time = time.time()

    if args.device=='cuda':
        mp.spawn(main, nprocs=args.world_size, args=(args,))
    elif args.device=='cpu':
        main(0, args)    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('running time {}'.format(total_time_str))
