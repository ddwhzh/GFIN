import math
import torch
from torch import nn

from ...util.misc import NestedTensor

def build_position_encoding(opt):
    '''
        进行位置编码
    '''
    N_steps = opt.hidden_dim // 2   #  128
    #   余弦位置编码
    if opt.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments    
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)    
    #   可学习的绝对编码 
    elif opt.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)                 
    else:
        raise ValueError(f"not supported {opt.position_embedding}")

    return position_embedding

class PositionEmbeddingLearned(nn.Module):
    """   
        绝对的pos嵌入, learned 
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)                               #   默认需要编码的行列位置不超过50个， 嵌入层（50，256）
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors                                                         #   图像，形状为[batch_size x 3 x H x W]
        h, w = x.shape[-2:]             
        i = torch.arange(w, device=x.device)                                            #   一行中的每一个位置   
        j = torch.arange(h, device=x.device)                                            #   一列中的每个位置
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)                          #   （w, num_pos_feats）
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)                          #   (h, num_pos_feats)
        pos = torch.cat([x_emb, y_emb], dim=-1)        
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) 
        #   (batch_size, num_pos_feats*2, h, w)
        return pos


class PositionEmbeddingSine(nn.Module):
    """
        这是一个比较标准的位置嵌入版本，
        非常类似于Attention is all you need paper所使用的版本，通用于图像工作。
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")     #   如果通过了scale，normalize应该为True
        if scale is None:
            scale = 2 * math.pi                                                 #   角度范围在0到2Pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors                                                 #   batched图像，形状为[batch_size x 3 x H x W]（实际上未使用）
        mask = tensor_list.mask                                                 #   [batch_size x H x W]的二进制掩码（单通道黑白），在填充像素为1
        assert mask is not None
        not_mask = ~mask                                                        #   图像中不是padding的部分
        y_embed = not_mask.cumsum(1, dtype=torch.float32)                       #   在第一维（列方向）累加（b, h, w），bool -> float
        x_embed = not_mask.cumsum(2, dtype=torch.float32)                       #   在第二维（行方向）累加（b, h, w）
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale                                 #   在行方向做归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale                                 #   在列方向做归一化

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)                             #   pow(10000, 2i/d), 2i需要在num_pos_feats范围内，因此i为dim_t//2           

        pos_x = x_embed[:, :, :, None] / dim_t                                                          #   (b, h, w, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t                                                          #   (b, h, w, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), 
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)                             #   (b,h,w,2 * (num_pos_feats//2))
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), 
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) 
         #   (batch_size, num_pos_feats*2, h, w)
        return pos

