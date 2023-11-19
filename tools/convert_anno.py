import json
import numpy as np
import argparse

def convert_anno(version):
    anno_file = "/root/whzh/codes/HOI/upt/origin_anno/vcoco/instances_vcoco_{}.json".format(version)
    with open(anno_file, 'r') as f:
        hicos = json.load(f)
    anno_file = "/root/whzh/codes/HOI/upt/vcoco/v-coco/v-coco/annotations/cdn_annotations/{}_vcoco.json".format(version)
    with open(anno_file, 'r') as f:
        hicos_2 = json.load(f)


    save_hico = dict(annotations=None, classes=None, objects=None, images=None, action_to_object=None)
    save_hico['classes'] = ['hold obj', 'stand', 'sit instr', 'ride instr', 'walk', 'look obj', 'hit instr', 'hit obj',
                            'eat obj', 'eat instr', 'jump instr', 'lay instr', 'talk_on_phone instr', 'carry obj',
                            'throw obj', 'catch obj', 'cut instr', 'cut obj', 'run', 'work_on_computer instr',
                            'ski instr', 'surf instr', 'skateboard instr', 'smile', 'drink instr', 'kick obj',
                            'point instr', 'read obj', 'snowboard instr']
    save_hico['objects'] = hicos['objects']
    save_hico['images'] = [int(hico['file_name'].split('_')[-1].split('.')[0])   for hico in hicos_2]
    corre_vcoco = np.load('/root/whzh/codes/HOI/upt/vcoco/v-coco/v-coco/annotations/cdn_annotations/corre_vcoco.npy')
    save_hico['action_to_object'] = [(np.where(cor==1)[0]+1).tolist() for cor in corre_vcoco]

    save_hico['annotations'] = list()
    temp = dict()
    for hico in hicos_2:
        temp[int(hico['file_name'].split('_')[-1].split('.')[0])] = hico
        
    for img in save_hico['images']:
        x = dict()
        t = temp[img]
        x['file_name'] = t['file_name']
        x['boxes_h'] = [list(np.array(t['annotations'])[i['subject_id']]['bbox']) for i in t['hoi_annotation']]
        x['boxes_o'] = [list(np.array(t['annotations'])[i['object_id']]['bbox']) if i['object_id']!=-1 else 
                        list(np.array(t['annotations'])[i['subject_id']]['bbox'])
                        for i in t['hoi_annotation']]
        x['actions'] = [[i['category_id']] for i in t['hoi_annotation']]

        x['objects'] = [int(np.array(t['annotations'])[i['object_id']]['category_id']) if i['object_id']!=-1 else 
                        int(np.array(t['annotations'])[i['subject_id']]['category_id'])
                        for i in t['hoi_annotation']]
        cov = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
        x['objects'] = [int(np.where([i==j for i in cov])[0]+1) for j in x['objects']]
        save_hico['annotations'].append(x)

    anno_file = "/root/whzh/codes/HOI/upt/vcoco/instances_vcoco_{}_plus.json".format(version)
    with open(anno_file, 'w') as f:
        json.dump(save_hico, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='test', type=str)
    args = parser.parse_args()
  
    convert_anno(args.version)