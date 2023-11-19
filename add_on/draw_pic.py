import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T



# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])





colors_hp = [(255, 0, 255), (0, 255, 0), (0, 255, 255),
             (255, 0, 0),   (0, 0, 255), (255, 0, 0), 
             (0, 0, 255),   (255, 0, 0), (0, 0, 255)]
colors_action_hp = [(255, 0, 0), (0, 0, 255),(255, 255, 0),
                    (255, 0, 0), (0, 0, 255), (255, 0, 0), 
                    (0, 0, 255),(255, 0, 0), (0, 0, 255)]
#   hoi标签
OBJECTS = [
    'person', 
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear','hair drier', 'toothbrush']


hico_verb_name_dict = {0: 'adjust', 1: 'assemble', 2: 'block', 3: 'blow', 4: 'board', 5: 'break', 6: 'brush_with', 7: 'buy', 8: 'carry', 9: 'catch', 
                    10: 'chase', 11: 'check', 12: 'clean', 13: 'control', 14: 'cook', 15: 'cut', 16: 'cut_with', 17: 'direct', 18: 'drag', 19: 'dribble', 
                    20: 'drink_with', 21: 'drive', 22: 'dry', 23: 'eat', 24: 'eat_at', 25: 'exit', 26: 'feed', 27: 'fill', 28: 'flip', 29: 'flush', 
                    30: 'fly', 31: 'greet', 32: 'grind', 33: 'groom', 34: 'herd', 35: 'hit', 36: 'hold', 37: 'hop_on', 38: 'hose', 39: 'hug', 
                    40: 'hunt', 41: 'inspect', 42: 'install', 43: 'jump', 44: 'kick', 45: 'kiss', 46: 'lasso', 47: 'launch', 48: 'lick', 49: 'lie_on', 
                    50: 'lift', 51: 'light', 52: 'load', 53: 'lose', 54: 'make', 55: 'milk', 56: 'move', 57: 'no_interaction', 58: 'open', 59: 'operate', 
                    60: 'pack', 61: 'paint', 62: 'park', 63: 'pay', 64: 'peel', 65: 'pet', 66: 'pick', 67: 'pick_up', 68: 'point', 69: 'pour', 
                    70: 'pull', 71: 'push', 72: 'race', 73: 'read', 74: 'release', 75: 'repair', 76: 'ride', 77: 'row', 78: 'run', 79: 'sail', 
                    80: 'scratch', 81: 'serve', 82: 'set', 83: 'shear', 84: 'sign', 85: 'sip', 86: 'sit_at', 87: 'sit_on', 88: 'slide', 89: 'smell', 
                    90: 'spin', 91: 'squeeze', 92: 'stab', 93: 'stand_on', 94: 'stand_under', 95: 'stick', 96: 'stir', 97: 'stop_at', 98: 'straddle', 99: 'swing', 
                    100: 'tag', 101: 'talk_on', 102: 'teach', 103: 'text_on', 104: 'throw', 105: 'tie', 106: 'toast', 107: 'train', 108: 'turn', 109: 'type_on', 
                    110: 'walk', 111: 'wash', 112: 'watch', 113: 'wave', 114: 'wear', 115: 'wield', 116: 'zip'}

# vcoco_verb_name_dict = { 0:'hold obj', 1:'sit instr', 2:'ride instr', 3:'look obj', 4:'hit instr', 5:'hit obj', 6:'eat obj',
#                         7:'eat instr', 8:'jump instr', 9:'lay instr', 10:'talk_on_phone instr', 11:'carry obj', 12:'throw obj',
#                         13:'catch obj', 14:'cut instr', 15:'cut obj', 16:'work_on_computer instr', 17:'ski instr', 18:'surf instr',
#                         19:'skateboard instr', 20:'drink instr', 21:'kick obj', 22:'read obj', 23:'snowboard instr'}

vcoco_verb_name_dict = {0:'hold obj', 1:'stand', 2:'sit instr', 3:'ride instr', 4:'walk', 5:'look obj', 6:'hit instr', 7:'hit obj',
                        8:'eat obj', 9:'eat instr', 10:'jump instr', 11:'lay instr', 12:'talk_on_phone instr', 13:'carry obj',
                        14:'throw obj', 15:'catch obj', 16:'cut instr', 17:'cut obj', 18:'run', 19:'work_on_computer instr',
                        20:'ski instr', 21:'surf instr', 22:'skateboard instr', 23:'smile', 24:'drink instr', 25:'kick obj',
                        26:'point instr', 27:'read obj', 28:'snowboard instr'}

font=cv2.FONT_HERSHEY_SIMPLEX
thick = 1            
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def folat2uint8(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = np.asarray(255.0 * img, np.uint8)
    return img

def bgr2rgb(img):
    img = img.detach().cpu().numpy()
    img = np.asarray(img, np.uint8)
    b,g,r = cv2.split(img)   # 分解Opencv里的标准格式B、G、R
    img = cv2.merge([r,g,b]) # 将BGR格式转化为常用的RGB格式
    return img


def rgb2bgr(img):
    img = img.detach().cpu().numpy()
    img = np.asarray(img, np.uint8)
    r,g,b = cv2.split(img)   # 分解Opencv里的标准格式B、G、R
    img = cv2.merge([b,g,r]) # 将BGR格式转化为常用的RGB格式
    return img

def paint_heatmap(img, heatmap, alpha = 0.5):
    height, weight = img.shape[0], img.shape[1]
    overlay = img.copy()
    
    heatmap = heatmap.view(1,1,heatmap.shape[0],heatmap.shape[1])
    heatmap =  torch.nn.functional.interpolate(heatmap, size=(height, weight), scale_factor=None, mode='bilinear', align_corners=True)
    heatmap = heatmap.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    heatmap = folat2uint8(heatmap).reshape(height, weight)
    #heatmap = (heatmap - heatmap.min()) // (heatmap.max() - heatmap.min())
    #heatmap = np.asarray(colorize(heatmap), np.uint8)
    heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    cv2.rectangle(overlay, (0, 0), (height, weight), (255, 0, 0), -1) # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap, alpha, img, 1-alpha, 0, img) # 将热度图覆盖到原图
    
    # heatmap = ( (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) ) * 255
    # heatmap = heatmap[0].detach().cpu().numpy()
    # heatmap = cv2.resize(heatmap, (weight, height), interpolation=cv2.INTER_CUBIC)
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = colorize(heatmap)
    # img = img * alpha  + heatmap * (1-alpha)
    return img


def img_add(img, heatmap, alpha = 0.5):
    cv2.addWeighted(heatmap, alpha, img, 1-alpha, 0, img) # 将热度图覆盖到原图   
    return img 



def draw_img_gt(img, output, top_k, rel_threshold=0, dataset_verb='vcoco', thick=1):
    list_action = []
    #   
    for i in range(output['hois'].shape[0]):
        subject_id  = int(output['hois'][i, 0].numpy())
        object_id   = int(output['hois'][i, 1].numpy())
        category_id = int(output['hois'][i, 2].numpy())
        score       = float(np.array(1.0))

        single_out  = [subject_id, object_id, category_id, score]
        list_action.append(single_out)
    list_action = sorted(list_action, key=lambda x:x[-1], reverse=True)
    action_dict, action_cate = [], []
    #   topK的动作29
    for output_i in list_action[:top_k]:
        subject_id, object_id, category_id, score = output_i

        if score<=rel_threshold:
            continue
        subject_obj = output['labels'][subject_id]
        subject_box = output['boxes'][subject_id]
        object_obj = output['labels'][object_id]
        object_box = output['boxes'][object_id]
        img = draw_box_on_img(subject_box, img, subject_obj, thick)
        img = draw_box_on_img(object_box, img, object_obj, thick)
        
        point_1 = [int((subject_box[0]+subject_box[2])*1.0/2),int((subject_box[1]+subject_box[3])*1.0/2)]
        point_2 = [int((object_box[0]+object_box[2])*1.0/2),int((object_box[1]+object_box[3])*1.0/2)]
        if [point_1,point_2] not in action_dict:
            action_dict.append([point_1,point_2])
            action_cate.append([])
            action_cate[action_dict.index([point_1,point_2])].append(category_id)
    #print(action_dict)
    #   画交互线
    for action_item in action_dict:
        img = draw_line_on_img(action_item,img, action_cate[action_dict.index(action_item)], dataset_verb, thick)
    return img


def draw_img_pred(img, results, top_k, rel_threshold=0, dataset_verb='vcoco', thick=2):
    list_action = []
    for i in range(results['verb_scores'].shape[0]):
        subject_id = int(results['sub_ids'][i].numpy())#diff
        object_id = int(results['obj_ids'][i].numpy())#diff
        #   100,
        category_id = int(results['verb_scores'].max(-1)[1][i].numpy())#diff
        score = float(results['verb_scores'].max(-1)[0][i].numpy())#diff
        single_out = [subject_id, object_id, category_id, score]
        list_action.append(single_out)
        
    list_action = sorted(list_action, key=lambda x:x[-1], reverse=True)
    
    action_dict, action_cate = [], []
    bbox_len = len(results['sub_ids'])
    for output_i in list_action[:top_k]:
        subject_id, object_id, category_id, score = output_i

        if score<=rel_threshold:
            continue
        subject_label = results['labels'][subject_id]
        subject_box = results['boxes'][subject_id]
        subject_score = results['score'][subject_id]
        
        object_label = results['labels'][object_id]
        object_box = results['boxes'][object_id]
        object_score = results['score'][object_id]
        
        img = draw_box_on_img(subject_box, img, subject_label, thick)
        img = draw_box_on_img(object_box, img, object_label, thick)
        
        point_1 = [int((subject_box[0]+subject_box[2])*1.0/2),int((subject_box[1]+subject_box[3])*1.0/2)]
        point_2 = [int((object_box[0]+object_box[2])*1.0/2),int((object_box[1]+object_box[3])*1.0/2)]
        if [point_1,point_2] not in action_dict:
            action_dict.append([point_1,point_2])
            action_cate.append([])
            action_cate[action_dict.index([point_1,point_2])].append(category_id)
    #print(action_dict)
    #   画交互线
    for action_item in action_dict:
        img = draw_line_on_img(action_item,img, action_cate[action_dict.index(action_item)], dataset_verb, thick)
    return img


def draw_point_on_img(xy, img, class_index, thick=1):
    '''
        在图片上画边框
    '''
    if class_index >= 1:
        if class_index == 80:
            class_index = 2
        else:
            class_index = 1
    vis_img = img.copy()#   image
    draw_point=[int(xy[0]),int(xy[1])]
    cv2.circle(vis_img,(draw_point[0],draw_point[1]), thick+1 , colors_hp[class_index], -1)
    return vis_img

def draw_box_on_img(box, img, class_index, thick=1):
    '''
        在图片上画边框
    '''
    if class_index >= 1:
        if class_index == 80:
            class_index = 2
        else:
            class_index = 1
    vis_img = img.copy()#   image
    box = [int(x) for x in box]
    cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), colors_hp[class_index], thick)
    draw_point=[int((box[0]+box[2])*1.0/2),int((box[1]+box[3])*1.0/2)]
    cv2.circle(vis_img,(draw_point[0],draw_point[1]), thick+1, colors_hp[class_index], -1)
    return vis_img

def draw_line_on_img(line, img, class_index, dataset_verb, thick=1):
    #print(class_index)
    vis_img = img.copy()
    cv2.line(vis_img, (line[0][0],line[0][1]), (line[1][0], line[1][1]), colors_action_hp[1], thick) #5
    if dataset_verb=='vcoco':
        verb = vcoco_verb_name_dict
    elif dataset_verb=='hico':
        verb = hico_verb_name_dict
    for i in range(len(class_index)):
        if i == 0:
            action_str = verb[class_index[i]]
        else:
            action_str= action_str + '/' + verb[class_index[i]]
    #action_str = 'jump'
    font=cv2.FONT_HERSHEY_SIMPLEX
    img=cv2.putText(vis_img, action_str, 
                    (int((line[0][0]+line[1][0])/2), int((line[1][1]+line[1][1])/2+20)), 
                    font, 1.2, colors_action_hp[1], thick)
    #  img=cv2.putText(vis_img, str(score) ,(int((line[0][0]+line[1][0])/2),int((line[1][1]+line[1][1])/2+50)),font,1.2,colors_action_hp[class_index],3)
    return vis_img

def rescale_bboxes(out_bbox, size):
    #把比例坐标乘以图像的宽和高，变成真实坐标
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def resize(img,size):
    img = Image.fromarray(img) 
    img = F.resize(img, size)
    img = np.array(img)
    return img

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor_RGB(gray_img[y,x], 0, 1)
    return out

def getJetColor_RGB(v, vmin, vmax):#BGR
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[2] = 256 * (0.5 + (v * 4)) #R: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[2] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[2] = 256 * (-4 * v + 2.5)  #R: 1 ~ 0
        c[1] = 255
        c[0] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[0] = 255
    else:
        c[0] = 256 * (-4 * v + 4.5) #B: 1 ~ 0.5                      
    return c

def paint_heatmap(heatmap, img, mask, hs, type='mean'):
    if type == 'max':
        heatmap = heatmap.max(0, keepdim=True)[0]
    elif type == 'mean':
        heatmap = heatmap.mean(0, keepdim=True)

    for i in range(heatmap.shape[0]):
        img = paint_heatmap(img, heatmap[i])
    
    img = img[mask].reshape(hs,-1,3) 
    return img
