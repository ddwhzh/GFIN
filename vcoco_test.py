import os
import time
import argparse
import datetime
from vcoco.vsrl_eval import VCOCOeval
import utils
from collections import defaultdict

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--det_file', default="./logs/vcoco/debug/cache.pkl", type=str)
  parser.add_argument('--mode', default='scenario_1', type=str)
  parser.add_argument('--ovr_thresh', default=0.5, type=float)
  parser.add_argument('--ignore_point', action='store_true')
  args = parser.parse_args()
  
  print('start eval')
  start_time = time.time()
  if not os.path.exists(args.det_file):
    raise NotImplementedError(f"No det file, please check.")
    
  vcocoeval = VCOCOeval(vsrl_annot_file='/WHZH/dataset/v-coco/data/vcoco/vcoco_test.json', 
                        coco_annot_file='/WHZH/dataset/v-coco/data/instances_vcoco_all_2014.json', 
                        split_file='/WHZH/dataset/v-coco/data/splits/vcoco_test.ids')
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  print('data load time {}'.format(total_time_str))
    # e.g. vsrl_annot_file: data/vcoco/vcoco_val.json
    #      coco_file:       data/instances_vcoco_all_2014.json
    #      split_file:      data/splits/vcoco_val.ids
  vcocoeval._do_eval(args.det_file, mode=args.mode, 
                     ovr_thresh=args.ovr_thresh, ignore_point = args.ignore_point)


