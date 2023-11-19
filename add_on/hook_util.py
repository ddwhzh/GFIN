import torch
import numpy as np 

def define_hook(args, gfin):
    hooks = []
    #backbone
    conv_features = []
    hooks += [
        gfin.detector.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output['3'].tensors))
    ]
    #   DETR decoder
    od_dec_attn_weights = []
    hooks += [
        gfin.detector.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
        lambda self, input, output: od_dec_attn_weights.append(output[1])
        ) for i in range(args.dec_layers)
    ]
    
    #   decoder attention
    hoi_dec_attn_weights = []
    hooks += [
        gfin.interaction_head.HOI_layer.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: hoi_dec_attn_weights.append(output[1]))
    ]
    return hooks, \
        conv_features, od_dec_attn_weights, hoi_dec_attn_weights

def hook_process_per_image(index, 
                           conv_features, od_dec_attn_weights, hoi_dec_attn_weights,
                           verb_dec_attn_weights, pairwise_dec_attn_weights, enc_attn_weights):
    conv_feature  = conv_features[index].cpu().numpy()
    w, h = conv_feature.shape[-2], conv_feature.shape[-1]
   
    od_dec_attn_weight = None
    od_dec_attn_weight = torch.cat(od_dec_attn_weights[index:index+6]).view(6, -1, w, h).cpu().numpy()
    
    hoi_dec_attn_weight = None
    hoi_dec_attn_weight = hoi_dec_attn_weights[index].view(1, -1, w, h).cpu().numpy()
    
    verb_dec_attn_weight = None
    verb_dec_attn_weight = verb_dec_attn_weights[index].cpu().numpy()
    
    pairwise_dec_attn_weight = None
    pairwise_dec_attn_weight = pairwise_dec_attn_weights[index].view(1, -1, w, h).cpu().numpy()
    
    enc_attn_weight = enc_attn_weights[index]
    
    return od_dec_attn_weight, hoi_dec_attn_weight, verb_dec_attn_weight, \
        pairwise_dec_attn_weight, enc_attn_weight

def hook_list_process(conv_features, od_dec_attn_weights, hoi_dec_attn_weights,
                 verb_dec_attn_weights, pairwise_dec_attn_weights, enc_attn_weights):
    '''
        对所有特征统一处理
    '''
    conv_features_list = [] 
    for i in range(len(conv_features)):
        conv_features_list += list(conv_features[i].cpu().numpy())
        
    od_dec_attn_weights_list = []
    for i in range(len(od_dec_attn_weights)):
        od_dec_attn_weights_list += list(od_dec_attn_weights[i].cpu().numpy())
        # for j in range(len(od_dec_attn_weights[i])):
        #     od_dec_attn_weights_list += list(od_dec_attn_weights[i][j].cpu().numpy())
    
    hoi_dec_attn_weights_list = []
    for i in range(len(hoi_dec_attn_weights)):
        hoi_dec_attn_weights_list += list(hoi_dec_attn_weights[i].cpu().numpy())
    
    verb_dec_attn_weights_list = []        
    for i in range(len(verb_dec_attn_weights)):
        verb_dec_attn_weights_list += list(verb_dec_attn_weights[i].cpu().numpy())    
    
    pairwise_dec_attn_weights_list = []
    for i in range(len(pairwise_dec_attn_weights)):
        pairwise_dec_attn_weights_list += list(pairwise_dec_attn_weights[i].cpu().numpy())
    
    return conv_features_list, od_dec_attn_weights_list, hoi_dec_attn_weights_list, \
        verb_dec_attn_weights_list, pairwise_dec_attn_weights_list

def hook_process_dataset(args, index, 
                        conv_features_list, od_dec_attn_weights_list, hoi_dec_attn_weights_list, \
                        verb_dec_attn_weights_list, pairwise_dec_attn_weights_list):
    conv_feature  = conv_features_list[index]
    w, h = conv_feature.shape[-2], conv_feature.shape[-1]

    # od_dec_attn_weight = np.concatenate(od_dec_attn_weights_list[6*index : 6*(index+1)]).reshape(6, -1, w, h) \
    #     if len(od_dec_attn_weights_list)!=0 else None

    od_dec_attn_weight = np.concatenate(od_dec_attn_weights_list[6*index : 6*(index+1)]).reshape(6, -1, w, h) \
        if len(od_dec_attn_weights_list)!=0 else None

    hoi_dec_attn_weight = hoi_dec_attn_weights_list[index].reshape(1, -1, w, h) \
        if len(hoi_dec_attn_weights_list)!=0 else None
        
    verb_dec_attn_weights = verb_dec_attn_weights_list[index] \
        if len(verb_dec_attn_weights_list)!=0 else None
        
    pairwise_dec_attn_weights = pairwise_dec_attn_weights_list[index].reshape(1, -1, w, h) \
        if len(pairwise_dec_attn_weights_list)!=0 else None
        
    return od_dec_attn_weight, hoi_dec_attn_weight, verb_dec_attn_weights, pairwise_dec_attn_weights