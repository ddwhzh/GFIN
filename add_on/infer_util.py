from cv2 import getTickCount, getTickFrequency
import copy
import time
import os
import torch
import pocket
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
from PIL import Image, ImageDraw
from torchvision import transforms

from mpl_toolkits.axes_grid1 import make_axes_locatable
from add_on.draw_pic import *
from add_on.pnms import *
from add_on.hook_util import *
from thop import profile

warnings.filterwarnings("ignore")




def image_run(args, gfin, dataset):
    hooks, conv_features, od_dec_attn_weights, hoi_dec_attn_weights = define_hook(args, gfin)
    verb_dec_attn_weights, pairwise_dec_attn_weights, enc_attn_weights = [], [], []
    
    if args.index_list!=-1:#    跑数据集中的几张图
        output = []        
        for index in range(args.index_start, args.index_list):
            ind = args.index_start
            img_list = [dataset[index][0].to(args.device)] 
            torch.cuda.synchronize()
            start = time.time()
            if args.compute_para:
                flops, params = profile(gfin, (img_list, None), verbose=False)
                print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
            out = gfin(img_list, None,)  
            torch.cuda.synchronize()
            end = time.time()
            print('infer_time:', end-start)    
            output.extend(out)  
    elif args.image_path is None:#  跑数据集一张图  
        ind = args.index
        image, _= dataset[args.index]
        image = image.to(args.device)
        torch.cuda.synchronize()
        start = time.time()
        if args.compute_para:
            flops, params = profile(gfin, ([image], None), verbose=False)
            print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        output = gfin([image], None)
        torch.cuda.synchronize()
        end = time.time()
        print('infer_time:', end-start)  
    else:#  跑一张图
        ind = 0
        image = dataset.dataset.load_image(args.image_path)
        
        image_tensor, _ = dataset.transforms(image, None)
        image_tensor = image_tensor.to(args.device)
        torch.cuda.synchronize()
        start = time.time()
        output = gfin([image_tensor])
        torch.cuda.synchronize()
        end = time.time()
        print('infer_time:', end-start) 
        
    conv_features_list, od_dec_attn_weights_list, hoi_dec_attn_weights_list, \
    verb_dec_attn_weights_list, pairwise_dec_attn_weights_list = \
    hook_list_process(conv_features, od_dec_attn_weights, hoi_dec_attn_weights, \
                verb_dec_attn_weights, pairwise_dec_attn_weights, enc_attn_weights)

    if len(hooks)!=0:
        for hook in hooks:
            hook.remove()   
    print("draw start")
    for index in range(0, len(output)):   
        out = output[index]
        out = pocket.ops.relocate_to_cpu(out, ignore=True)   
        
        if args.pnms:
            boxes = out['boxes']
            pairing = out['pairing']
            boxes_h, boxes_o = boxes[pairing].unbind(0)
            boxes_h, boxes_o, out['objects'], out['scores'], out['labels'], out['pairing'] = \
            triplet_nms_filter(boxes_h, boxes_o, out['objects'], out['scores'],  out['labels'],  out['pairing'])  
        
        
        if args.index_list!=-1 or args.image_path is None:
            per_index = ind if args.index_list==-1 and args.index is not None else args.index_start + index 
            image = dataset.dataset.load_image(
                os.path.join(dataset.dataset._root,
                    dataset.dataset.filename(per_index)))
            img_name = "{}".format(per_index)
            args.draw_pic = False
        else:
            img_name = args.image_path.split('/')[-1].split('.')[0]
            
        od_dec_attn_weight, hoi_dec_attn_weight, verb_dec_attn_weights, pairwise_dec_attn_weights = \
                    hook_process_dataset(args, index, 
                    conv_features_list, od_dec_attn_weights_list, hoi_dec_attn_weights_list, \
                    verb_dec_attn_weights_list, pairwise_dec_attn_weights_list)
            

                
        visualise_entire_image(args, image, out,  
                                od_dec_attn_weight, hoi_dec_attn_weight, #verb_dec_attn_weight,
                                args.output_img, img_name, args.action,  args.action_score_thresh,
                                dataset=args.dataset, draw_pic= args.draw_pic)
         


##########################################################################################################################

def visualise_entire_image(args, image, out,
                           od_dec_attn_weight=None, hoi_dec_attn_weight=None, #verb_dec_attn_weight=None,
                           output_img=None, img_name=None, action=None, thresh=0.2, dataset='hicodet',  draw_pic=False):
    """Visualise bounding box pairs in the whole image by classes"""
    ori_image = copy.deepcopy(image)
    path = os.path.join(output_img, img_name)
    if not os.path.exists(path):
        os.makedirs(path)      

    plt.figure()
    plt.imshow(image)
    plt.axis('off')            
    if output_img is not None:
        plt.savefig(path+"/origin_img.png")
    plt.show()

    # Rescale the boxes to original image size
    verb_name_dict = hico_verb_name_dict if dataset == 'hicodet' else vcoco_verb_name_dict
    ow, oh = image.size
    h, w = out['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = out['boxes'] * scale_fct
    # Find the number of human and object instances
    pairing = out['pairing']
    nh = len(pairing[0].unique())
    no = len(boxes)

    scores = out['scores']
    pred = out['labels']
    
    bx_h, bx_o = boxes[pairing].unbind(0)
    bx_o_keep = []
    for j, _ in enumerate(bx_o):
        if (bx_o[j] == bx_h[j]).all():
            bx_o[j] *= 0
            bx_o_keep.append(False)
        else:
            bx_o_keep.append(True)
    
    #######################################################################################
    #   将检测到的人-物对与所附分数可视化
    # 1.Visualise detected human-object pairs with attached scores
    #######################################################################################

    plt.figure()
 
    bp_list = []
    map_box = dict()
    for action_i in verb_name_dict.keys():
        if action is not None and action_i!=action:
            continue
        x = torch.logical_and(scores >= thresh, pred == action_i)
        keeps = torch.nonzero(x * torch.tensor(bx_o_keep)).squeeze(1)
        pocket.utils.draw_box_pairs(image, bx_h[keeps], bx_o[keeps], width=5)
        
        if len(bx_o_keep)!=0:
            keeps = torch.nonzero(x * ~torch.tensor(bx_o_keep)).squeeze(1)
            bx_h_1 = bx_h[keeps].reshape(-1, 4)
            canvas = ImageDraw.Draw(image)
            for b in bx_h_1:
                canvas.rectangle(b.tolist(), outline='#007CFF', width=5)
                b_h_centre = (b[:2]+b[2:])/2
                canvas.ellipse((b_h_centre - 5).tolist() + (b_h_centre + 5).tolist(), fill='#FF4444')

        keep = torch.nonzero(x).squeeze(1)            
        for _, kp in enumerate(keep):
            if len(keep)==0:break
            bp = bx_h[kp, :2].tolist()
            if str(bp) not in map_box.keys():
                map_box[str(bp)] =f"{scores[kp]:.2f} "+f"{verb_name_dict[action_i]}\n"
                bp_list.append(bp)
            else: 
                map_box[str(bp)] += f"{scores[kp]:.2f} "+f"{verb_name_dict[action_i]}\n" 
    
    #   打印文字  
    for bp in bp_list:
        txt = plt.text(bp[0], bp[1], map_box[str(bp)][:-1], fontsize=10, fontweight='semibold', color='red')
        txt.set_path_effects([peff.withStroke(foreground='#000000')])
        plt.draw()
    plt.imshow(image)
    plt.axis('off')     
    ax = plt.gca()     
    if output_img is not None:
        plt.savefig(path+"/HOI.png")
    plt.show()

    #######################################################################################
    # 2.Heatmap_HOI
    #######################################################################################

    img = image
    cow, coh = img.size[0], img.size[1]
    if hoi_dec_attn_weight is not None:
        plt.figure() 
        hoi_dec_loc = out['hoi_dec_location']   
        if action is not None:# 单一的动作
            keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
            feature_map_sum = np.zeros((coh, cow), dtype=np.float)
            for i in keep:
                feature_map = hoi_dec_attn_weight[0, hoi_dec_loc[i]]
                if feature_map.sum()!= 0:
                    feature_map =(feature_map - feature_map.min()) /(feature_map.max() - feature_map.min()) * 255
                feature_map = cv2.resize(feature_map, (cow, coh), interpolation=cv2.INTER_CUBIC)
                feature_map_sum += feature_map

            feature_map = feature_map_sum
            
            if feature_map.sum()!= 0:
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) 
            feature_map = colorize(feature_map)
            img_copy = (np.array(copy.deepcopy(img)) * 0.5  + feature_map * 0.5)
        else: # 全部动作
            feature_map_sum = np.zeros((coh, cow), dtype=np.float)
            for action_i in verb_name_dict.keys():
                keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action_i)).squeeze(1)
                for i in keep:
                    feature_map = hoi_dec_attn_weight[0, hoi_dec_loc[i]]
                    if feature_map.sum()!= 0:
                        feature_map =(feature_map - feature_map.min()) /(feature_map.max() - feature_map.min()) * 255
                    feature_map = cv2.resize(feature_map, (cow, coh), interpolation=cv2.INTER_CUBIC)
                    feature_map_sum += feature_map
            feature_map = feature_map_sum
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) 
            feature_map = colorize(feature_map)
            img_copy = (np.array(copy.deepcopy(img)) * 0.5  + feature_map * 0.5)
        
        img_copy = img_copy.astype(int)
        plt.imshow(img_copy)
        plt.axis('off')            
        if output_img is not None:
            plt.savefig(path+"/Heatmap_HOI.png")
        plt.show()
    #######################################################################################
    # 3.Draw the bounding boxes
    #######################################################################################
    od_labels = out['od_labels']
    od_scores = out['od_scores']
    
    if draw_pic:
        keep = [sc>thresh for sc in od_scores]
        boxes = torch.tensor(np.array(boxes)[keep])
        od_labels = torch.tensor(np.array(od_labels)[keep])
        od_scores = torch.tensor(np.array(od_scores)[keep])
    
    plt.figure()
    plt.imshow(ori_image)
    plt.axis('off')
    ax = plt.gca()
    draw_boxes(ax, boxes, od_labels, od_scores)
    if output_img is not None:
        plt.savefig(path+"/bounding_boxes.png")
    plt.show()
    #######################################################################################
    # 4.Heatmap_OD
    #######################################################################################
    
    if od_dec_attn_weight is not None and 'od_dec_location' in out:
        od_dec_loc = out['od_dec_location']
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        ax = plt.gca()
        draw_boxes_HM(ax, boxes, image, od_dec_attn_weight, od_dec_loc)
        if output_img is not None:
            plt.savefig(path+"/Heatmap_OD.png")
        plt.show()

    #######################################################################################
    # 5.Visualise attention from the cooperative layer
    #######################################################################################
    
    coop_attn = out['attn_maps'][0]
    if coop_attn is not None and len(coop_attn)!=0:
        #fig, axe = plt.subplots(2, 4)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"Attention in coop. layer")
        #axe = np.concatenate(axe)
        try:ticks = list(range(coop_attn[0][0].shape[0]))
        except:ticks = list()
        labels = [v + 1 for v in ticks]
        attn_coop_list = []
        for i, attn_1 in enumerate(coop_attn):#2
            attn_list = []
            # for ax, attn in zip(axe, attn_1):#8
            #     attn_list.append(attn.squeeze(-1))
            for attn in attn_1:#8
                attn_list.append(attn.squeeze(-1))
            attn_coop_list.append(torch.stack(attn_list, -1))
            
        attn = torch.mean(torch.cat(attn_coop_list, -1) ,-1, keepdim=True).cpu().numpy()
        im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        for i in range(len(ticks)):
            for j in range(len(ticks)):
                text = ax.text(j, i, f"{float(attn.squeeze().T[i, j]):.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, cax=cax)
        if output_img is not None:
            plt.savefig(path+"/attention_from_the_cooperative_layer.png")


    #######################################################################################
    # 6.Visualise attention from the competitive layer
    #######################################################################################
    
    x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    x, y = torch.nonzero(x != y).unbind(1)
    pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]
    comp_attn = out['attn_maps'][1]
    if comp_attn is not None and len(comp_attn)!=0:
        #fig, axe = plt.subplots(1, 3)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Attention in comp. layer")
        #axe = np.concatenate(axe)
        try:ticks = list(range(len(pairs)))
        except:ticks = list()
        attn_list = []
        # for ax, attn in zip(axe, comp_attn):
        #     im = ax.imshow(attn[0].cpu().numpy(), vmin=0, vmax=1)
        for attn in comp_attn:#8
            attn_list.append(attn[0])
        attn = torch.mean(torch.stack(attn_list, -1) ,-1, keepdim=True).cpu().numpy()
        if attn.squeeze().size != 1:
            im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(pairs, rotation=45)
            ax.set_yticks(ticks)
            ax.set_yticklabels(pairs)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            # for i in range(len(ticks)):
            #     for j in range(len(ticks)):
            #         text = ax.text(j, i, f"{float(attn.squeeze().T[i, j]):.2f}", ha="center", va="center", color="black")
            fig.colorbar(im, cax=cax)
        
        if output_img is not None:
            plt.savefig(path+"/attention_from_the_competitive_layer.png")

    
    # # 打印预测的行动和相应的分数
    # # Print predicted actions and corresponding scores
    # pairing = output['pairing']
    # unique_actions = torch.unique(pred)
    # for verb in unique_actions:
    #     print(f"\n=> Action: {actions[verb]}")
    #     sample_idx = torch.nonzero(pred == verb).squeeze(1)
    #     for idx in sample_idx:
    #         idxh, idxo = pairing[:, idx] + 1
    #         print(
    #             f"({idxh.item():<2}, {idxo.item():<2}),",
    #             f"score: {scores[idx]:.4f}"
    #         )

##########################################################################################################################\
##########################################################################################################################\
     
def draw_heat_map(harvest, xaxis, yaxis, output_img=None):

    # 这里是创建一个画布
    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xaxis)))
    ax.set_yticks(np.arange(len(yaxis)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xaxis)
    ax.set_yticklabels(yaxis)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(yaxis)):
        for j in range(len(xaxis)):
            text = ax.text(j, i, harvest[i, j],ha="center", va="center", color="w")
    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.colorbar(im)
    if output_img is not None:
        plt.savefig(output_img +"/demo02.png")
    plt.show()

def draw_boxes(ax, boxes, od_labels = None, 
               od_scores = None, color='w', width=2):
    '''
        Draw the bounding boxes
    '''
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)      
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w', linewidth = width)
        ax.add_patch(patch)
        st = ""
        if od_labels is not None:
            st +=" {}".format(OBJECTS[od_labels.tolist()[i]]) 
        if od_scores is not None:
            st +=" {:.2f}".format(od_scores.tolist()[i])
        #   打印文字  
        txt = plt.text(*a.tolist(), str(i+1) +st, fontsize=12, fontweight='semibold', color=color)
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()
 


def draw_boxes_HM(ax, boxes, image,
               od_dec_attn_weight, od_dec_loc):
    '''
        Heatmap_OD
    '''
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    
    ow, oh = image.size
    feature_map_sum = np.zeros((oh, ow), dtype=np.float)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        
        # if i ==0 or i==1 or i==3:
        feature_map = od_dec_attn_weight[int(od_dec_loc[i,0]), int(od_dec_loc[i,1])]
        if feature_map.sum()!= 0:
            feature_map =(feature_map - feature_map.min()) /(feature_map.max() - feature_map.min()) * 255
        feature_map = cv2.resize(feature_map, (ow, oh), interpolation=cv2.INTER_CUBIC)
        feature_map_sum += feature_map
        
    feature_map = feature_map_sum
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) 
    feature_map = colorize(feature_map)
    img_copy = (np.array(copy.deepcopy(image)) * 0.5  + feature_map * 0.5)
    img_copy = img_copy.astype(int)
    plt.imshow(img_copy)

############################################################################
#
#      video run
#
############################################################################
def video_run(args, gfin, dataset):
    cap = cv2.VideoCapture(args.video_dir)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWrite = cv2.VideoWriter("./last_video"+'.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while(cap.isOpened()):
        index = 0
        start = time.time()
        img_ret, frame = cap.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        image_tensor, _ = dataset.transforms(image, None)
        image_tensor = image_tensor.to(args.device)
        output = gfin([image_tensor])
        out = output[index]
        out = pocket.ops.relocate_to_cpu(out, ignore=True)
        paint_frame = visualise_video(image, out, args.action,  
                                        args.action_score_thresh, dataset=args.dataset)
        # paint_frame = np.array(image)
        end = time.time()
        seconds = end - start
        FPS = int(1 / seconds)
        paint_frame=cv2.putText(paint_frame, str(FPS), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, colors_action_hp[1], 1)
        paint_frame = cv2.cvtColor(paint_frame, cv2.COLOR_RGB2BGR)  

        # cv2.imshow("video", paint_frame)
        videoWrite.write(paint_frame)
        c= cv2.waitKey(1) & 0xff 
        if c==27:
            cap.release()
            break
    print("end")
    return

def visualise_video(image, output, action=None, thresh=0.2, dataset='hicodet', draw_pic=True):
    # Rescale the boxes to original image size
    verb_name_dict = hico_verb_name_dict if dataset == 'hicodet' else vcoco_verb_name_dict
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    pairing = output['pairing']
    scores = output['scores']
    pred = output['labels']
    bx_h, bx_o = boxes[pairing].unbind(0)
    bx_o_keep = []
    for j, _ in enumerate(bx_o):
        if (bx_o[j] == bx_h[j]).all():
            bx_o[j] *= 0
            bx_o_keep.append(False)
        else:
            bx_o_keep.append(True)

    od_labels = output['od_labels']
    od_scores = output['od_scores']
    
    if draw_pic:
        keep = [sc>thresh for sc in od_scores]
        boxes = torch.tensor(np.array(boxes)[keep])
        od_labels = torch.tensor(np.array(od_labels)[keep])
        od_scores = torch.tensor(np.array(od_scores)[keep])

    # xy = boxes[:, :2].unbind(0)
    # h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)      
    # for i, b in enumerate(boxes):
    #     canvas = ImageDraw.Draw(image)
    #     canvas.rectangle(b.tolist(), outline='white', width=1)
    #     b_h_centre = (b[:2]+b[2:])/2
    #     # canvas.ellipse((b_h_centre - 5).tolist() + (b_h_centre + 5).tolist(), fill='#FF4444')


    bp_list = []
    map_box = dict()
    for action_i in verb_name_dict.keys():
        if action is not None and action_i!=action:
            continue
        if action_i ==57:
            continue
        x = torch.logical_and(scores >= thresh, pred == action_i)
        keeps = torch.nonzero(x * torch.tensor(bx_o_keep)).squeeze(1)
        pocket.utils.draw_box_pairs(image, bx_h[keeps], bx_o[keeps], width=5)
        
        if len(bx_o_keep)!=0:
            keeps = torch.nonzero(x * ~torch.tensor(bx_o_keep)).squeeze(1)
            bx_h_1 = bx_h[keeps].reshape(-1, 4)
            canvas = ImageDraw.Draw(image)
            for b in bx_h_1:
                canvas.rectangle(b.tolist(), outline='#007CFF', width=5)
                b_h_centre = (b[:2]+b[2:])/2
                canvas.ellipse((b_h_centre - 5).tolist() + (b_h_centre + 5).tolist(), fill='#FF4444')
            
        keep = torch.nonzero(x).squeeze(1)            
        for _, kp in enumerate(keep):
            if len(keep)==0:break
            bp = bx_h[kp, :2].tolist()
            if str(bp) not in map_box.keys():
                map_box[str(bp)] =""
                #map_box[str(bp)]  +=f"{scores[kp]:.2f} "
                map_box[str(bp)]  += f"{verb_name_dict[action_i]}\n"
                bp_list.append(bp)
            else: 
                #map_box[str(bp)] += f"{scores[kp]:.2f} "
                map_box[str(bp)] += f"{verb_name_dict[action_i]}\n"
    
    image = np.array(image)


    # for i, b in enumerate(boxes):
    #     st = ""
    #     st +=" {}".format(OBJECTS[od_labels.tolist()[i]]) 
    #     #st +=" {:.2f}".format(od_scores.tolist()[i])
    #     image=cv2.putText(image, st, (int(b[0]), int(b[1])), font, 0.8, colors_action_hp[2], 2)

    for bp in bp_list:
        image=cv2.putText(image, map_box[str(bp)][:-1], (int(bp[0]), int(bp[1]+20)),
                        font, 0.8, colors_action_hp[0], 2)

    return image


