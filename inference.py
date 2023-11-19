"""
    Visualise detected human-object interactions in an image

    Fred Zhang <frederic.zhang@anu.edu.au>

    The Australian National University
    Australian Centre for Robotic Vision
"""

import os
import torch
import pocket
import warnings
import argparse
import numpy as np



import utils
from utils import  DataFactory
from gfin import build_detector
from add_on.infer_util import *
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--start_epoch', default=0, type=int)#20
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--no_fpn', dest='fpn', action='store_false')
    parser.add_argument('--num_obj_classes', default=81, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    #   new(enc more important)
    parser.add_argument('--menc-inter-layer', default=4, type=int)
    parser.add_argument('--enc-inter-layers', default=4, type=int)
    parser.add_argument('--dec-inter-layers', default=4, type=int)
  
    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
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
    #   loss parameter
    ##################################
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)#0.2
    parser.add_argument('--lambda_para', default=2.8, type=float)#2.8
    ##################################
    parser.add_argument('--dataset', default='hicodet', type=str)
    # parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--human-idx', type=int, default=0)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=2, type=int)
    parser.add_argument('--max-instances', default=5, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--index_start', default=0, type=int)
    parser.add_argument('--index_list', default=-1, type=int)
    parser.add_argument('--action', default=None, type=int,
        help="Index of the action class to visualise.")
    
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
        help="Threshold on action classes.")
    
    parser.add_argument('--image-path', default=None, type=str,
        help="Path to an image file.")
    parser.add_argument('--output_img', default="./outputs_img", type=str,
        help="Path to an image file.")        

    #   train
    parser.add_argument('--cascade_layer', default=0, type=int)
    parser.add_argument('--HOF', action='store_true')
    # parser.add_argument('--no_HOF', dest='HOF', action='store_false')
    parser.add_argument('--no_label_match', dest='label_match', action='store_false')
    parser.add_argument('--no_Hungarian', dest='Hungarian', action='store_false')
    parser.add_argument('--aux_attn_loss', action='store_true')
    parser.add_argument('--num_layer', default=3, type=int)
    parser.add_argument('--no_nms', action='store_true')

    parser.add_argument('--draw_pic', action='store_true')
    parser.add_argument('--pnms', action='store_true')
    parser.add_argument('--no_inference', dest='inference', action='store_false')

    parser.add_argument('--video', action='store_true')
    # parser.add_argument('--video_dir', type=int, default=0,
    #                     help='the video path')
    parser.add_argument('--video_dir', type=str,
                        help='the video path')
    parser.add_argument('--save_video', type=str, default='',
                        help='the video save path')

    parser.add_argument('--compute_para', action='store_true')

    return parser


@torch.no_grad()
def main(args):
    dataset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root)
    #actions = dataset.dataset.verbs if args.dataset == 'hicodet' else dataset.dataset.actions
    if args.dataset == 'hicodet':
        object_to_target = dataset.dataset.object_to_verb
        args.num_classes = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(dataset.dataset.object_to_action.values())
        #args.num_classes = 24
        args.num_classes = 29

    

    gfin = build_detector(args, object_to_target).to(args.device)
    gfin.eval()

    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        gfin = utils.load_model(gfin, checkpoint['model_state_dict'])
        args.start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
    # else:
    #     raise Exception("no pretrain model")

    if args.video:# test video
        video_run(args, gfin, dataset)
    else:
        image_run(args, gfin, dataset)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('gfin training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_img = os.path.join(args.output_img, args.dataset)
    if not os.path.exists(args.output_img):
        os.makedirs(args.output_img)
    print(args)
    main(args)
