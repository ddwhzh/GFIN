import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import sys
from .mask import RTLMask_layer1, RTLMask_layer2, LTRMask_layer1, LTRMask_layer2
from .modules import conv3x3, Conv1x1
import math

#SA, CCA, YLA, Non-Local, SE, CBAM
def Attention_Layer(in_c, attention="SA"):
    if attention == "SA":
        return Self_Attention(in_c)
    elif attention == "CCA":
        return CCA(in_c)
    elif attention == "YLA":
        return Your_Local_Attention(in_c)
    else:
        return None



class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x







class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1./8, rd_channels=None, rd_divisor=1):
        super(GlobalContext, self).__init__()
        act_layer = nn.ReLU

        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None

        # if rd_channels is None:
        #     rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None

        self.gate = nn.Sigmoid()
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x





#######################################  Self_Attention Attention Modules (SA) ##################################################################
class Self_Attention_1d(nn.Module):
    def __init__(self, in_c, d_feature):
        super(Self_Attention_1d, self).__init__()
        self.in_c = in_c
        self.d_feature = d_feature
       
        self.query = nn.Linear(self.in_c, self.d_feature)
        self.key = nn.Linear(self.in_c, self.d_feature)
        self.value = nn.Linear(self.in_c, self.d_feature)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, query, key, value):
        #   100, N, 256
        Q = self.query(query).transpose(0,1)
        K = self.key(key).permute(1,2,0).contiguous()
        V = self.value(value).transpose(0,1)
        #   N, 100, 100
        scores = torch.matmul(Q, K) #/ math.sqrt(self.d_feature)
        attn_map = torch.tanh(scores)
        #   N, 100, 256
        out = torch.matmul(attn_map, V).transpose(0,1)
        out = self.gamma * out + value
        return out

class Self_Attention(nn.Module):
    def __init__(self, in_c):
        super(Self_Attention, self).__init__()
        self.in_c = in_c

        self.query = Conv1x1(self.in_c, self.in_c // 8)
        self.key = Conv1x1(self.in_c, self.in_c // 8)
        self.value = Conv1x1(self.in_c, self.in_c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        #   16, 256, 64, 64
        B, C, W, H = x.size()
        #   16, 64*64, 32
        query = self.query(x).view(B, C // 8, W * H).permute(0, 2, 1)
        #   16, 32, 64*64
        key = self.key(x).view(B, C // 8, W * H)
        #   16, 256, 64*64
        value = self.value(x).view(B, C, W * H)
        #   16, 64*64, 64*64
        attention_map = F.softmax(torch.bmm(query, key), -1)
        #   16, 256, 64*64
        out = torch.bmm(value, attention_map.permute(0, 2, 1))
        #   16, 256, 64, 64
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out

####################################### Criss-cross Attention Modules (CCAM) ##################################################################
class Criss_Cross_Attention(nn.Module):
    def __init__(self, in_c):
        super(Criss_Cross_Attention, self).__init__()
        self.in_c = in_c

        self.query = Conv1x1(self.in_c, self.in_c // 8)
        self.key = Conv1x1(self.in_c, self.in_c // 8)
        self.value = Conv1x1(self.in_c, self.in_c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        #   16, 256, 64, 64
        B, C, W, H = x.size()
        #   16, 32, 64, 64
        query = self.query(x)
        #   16, 32, 64, 64
        key = self.key(x)
        #   16, 256, 64, 64
        value = self.value(x)
        #   16*64, 64, 32
        query_h = query.permute(0, 3, 2, 1).contiguous().view(B * W, H, C // 8)
        query_w = query.permute(0, 2, 3, 1).contiguous().view(B * H, W, C // 8)
        #   16*64, 32, 64
        key_h = key.permute(0, 3, 1, 2).contiguous().view(B * W, C // 8, H)
        key_w = key.permute(0, 2, 1, 3).contiguous().view(B * H, C // 8, W)
        #   16*64, 256, 64
        value_h = value.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        value_w = value.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        
        #   16, 64, 64, 64
        energy_h = (
            torch.bmm(query_h, key_h).contiguous().view(B, W, H, H)
            - (torch.eye(W, W).unsqueeze(0).repeat(B * W, 1, 1) * sys.maxsize)
            .contiguous()
            .view(B, W, H, H)
            .cuda()
        ).permute(0, 2, 1, 3)
        energy_w = torch.bmm(query_w, key_w).contiguous().view(B, H, W, W)
        #   16, 64, 64, 128
        attention_map = F.softmax(torch.cat([energy_h, energy_w], 3), 3)
        #   16*64, 64, 64
        attention_map_h = (
            attention_map[:, :, :, 0:H]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B * W, H, H)
            .permute(0, 2, 1)
        )
        attention_map_w = (
            attention_map[:, :, :, H : H + W]
            .contiguous()
            .view(B * H, W, W)
            .permute(0, 2, 1)
        )
        #   16, 256, 64, 64
        out_h = (
            torch.bmm(value_h, attention_map_h)
            .contiguous()
            .view(B, W, -1, H)
            .permute(0, 2, 3, 1)
        )
        out_w = (
            torch.bmm(value_w, attention_map_w)
            .contiguous()
            .view(B, H, -1, W)
            .permute(0, 2, 1, 3)
        )
        out = out_h + out_w
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out

class CCA(nn.Module):
    def __init__(self, in_c):
        super(CCA, self).__init__()
        self.Attention = Criss_Cross_Attention(in_c)

    def forward(self, x):
        out = self.Attention(x)
        out = self.Attention(out)
        return out



####################################### Your_Local_Attention ##################################################################
class Your_Local_Attention(nn.Module):
    def __init__(self, in_c):
        super(Your_Local_Attention, self).__init__()
        self.in_c = in_c

        self.query = Conv1x1(self.in_c, self.in_c // 8)
        self.key = Conv1x1(self.in_c, self.in_c // 8)
        self.value = Conv1x1(self.in_c, self.in_c // 2)
        self.after_attention = Conv1x1(self.in_c // 2, self.in_c)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.mask_rtl1 = RTLMask_layer1
        self.mask_rtl2 = RTLMask_layer2
        self.mask_ltr1 = LTRMask_layer1
        self.mask_ltr2 = LTRMask_layer2

    def forward(self, x):
        #   16, 256, 64, 64
        B, C, W, H = x.size()

        head_num = 8
        head_size = C // (8 * head_num)
        #   16, 256, 64*64
        query = self.query(x).view(-1, head_num, head_size, H * W)
        #   16, 256, 64*64
        key = F.max_pool2d(self.key(x), kernel_size=2, stride=2, padding=0).view(
            -1, head_num, head_size, H * W // 4
        )
        attention_logits = torch.einsum("abcd, abce -> abde", query, key)

        masks = self.get_grid_masks((H, W), (H // 2, W // 2))
        attention_adder = (1.0 - masks) * (-1000.0)
        attention_adder = torch.from_numpy(attention_adder).cuda()

        attention_logits += attention_adder
        attention_map = F.softmax(attention_logits, dim=-1)

        value = F.max_pool2d(self.value(x), kernel_size=2, stride=2, padding=0)
        value_head_size = C // (2 * head_num)
        value = value.view(-1, head_num, value_head_size, H * W // 4)

        out = torch.einsum("abcd, abed -> abec", attention_map, value)
        out = out.contiguous().view(-1, C // 2, W, H)
        out = self.after_attention(out)
        out = self.gamma * out + x

        return out

    def get_grid_masks(self, gridO, gridI):
        masks = []

        # RTL
        masks.append(self.mask_rtl1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_rtl2.get_mask(gridI, nO=gridO))

        masks.append(self.mask_rtl1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_rtl2.get_mask(gridI, nO=gridO))

        # LTR
        masks.append(self.mask_ltr1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_ltr2.get_mask(gridI, nO=gridO))

        masks.append(self.mask_ltr1.get_mask(gridI, nO=gridO))
        masks.append(self.mask_ltr2.get_mask(gridI, nO=gridO))

        return np.array(masks)

####################################### SENET #########################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#attention
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #   N, 256, 64, 64
        b, c, _, _ = x.size()
        #   N, 256
        y = self.avg_pool(x).view(b, c)
        #   N, 256, 1, 1
        y = self.fc(y).view(b, c, 1, 1)
        #   N, 256, 64, 64
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#attention
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #   N, 256, 64, 64
        b, c, _, _ = x.size()
        #   N, 256
        y = self.avg_pool(x).view(b, c)
        #   N, 256, 1, 1
        y = self.fc(y).view(b, c, 1, 1)
        #   N, 256, 64, 64
        return x * y.expand_as(x)



class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)#加在第二层激活函数之前

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


####################################### Convolutional BlockAttention Modules (CBAM) ##################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #   N, 256, 64, 64
        avg_out = self.fc(self.avg_pool(x))
        #   N, 256, 1, 1
        max_out = self.fc(self.max_pool(x))
        #   N, 256, 1, 1
        out = avg_out + max_out
        #   N, 256, 64, 64
        y = self.sigmoid(out) * x
        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #   N, 256, 64, 64
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #   N, 1, 64, 64
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #   N, 1, 64, 64
        t = torch.cat([avg_out, max_out], dim=1)
        #   N, 2, 64, 64
        t = self.conv1(t)
        #   N, 1, 64, 64
        y = self.sigmoid(t) * x
        return y

class CBAM_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out)
        out = self.sa(out) 

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
####################################### Zhou_et_attention ##################################################################
class AdaptivestdPool2d(nn.Module):
    def __init__(self, in_planes):
        self.in_planes = in_planes
        super(AdaptivestdPool2d, self).__init__()
    def forward(self, x):
        b,c,w,h = x.size()
        x = x.view(b,c,-1)
        x = torch.std(x,2).view(b,self.in_plane,1,1)
        return x


class Zhou_et_Attention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.std_pool = AdaptivestdPool2d(1)
           
        self.fc_1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                  nn.BatchNorm2d(in_planes // ratio),
                                  nn.Sigmoid(),
                                  nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
                                  nn.BatchNorm2d(in_planes))
        self.fc_2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                  nn.BatchNorm2d(in_planes // ratio),
                                  nn.Sigmoid(),
                                  nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
                                  nn.BatchNorm2d(in_planes))
        self.fc_3 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                  nn.BatchNorm2d(in_planes // ratio),
                                  nn.Sigmoid(),
                                  nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
                                  nn.BatchNorm2d(in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #   N, 256, 64, 64
        avg_out = self.fc_1(self.avg_pool(x))
        #   N, 256, 1, 1
        max_out = self.fc_2(self.max_pool(x))
        
        std_out = self.fc_3(self.std_pool(x))
        #   N, 256, 1, 1
        out = avg_out + max_out + std_out
        #   N, 256, 64, 64
        y = self.sigmoid(out) * x
        return y

####################################### Non-Local ##################################################################
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, types='concatenation'):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        self.types = types
        
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if types=='concatenation':
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                nn.ReLU()
            )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        #   16, 256, 64, 64
        batch_size = x.size(0)
        #   16, 128, 64x64
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        #   16, 64x64, 128
        g_x = g_x.permute(0, 2, 1)#value


        if self.types=='concatenation':
            #   16, 128, 64*64, 1
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            h = theta_x.size(2)
            w = phi_x.size(3)
            #   16, 128, 64*64, 64*64
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            #   16, 256, 64*64, 64*64
            concat_feature = torch.cat([theta_x, phi_x], dim=1)
            #   16, 1, 64*64, 64*64
            f = self.concat_project(concat_feature)
            b, _, h, w = f.size()
            #   16, 64*64, 64*64
            f = f.view(b, h, w)
            N = f.size(-1)
            f_div_C = f / N
        elif self.types=='dot_product':
            #   16, 64*64, 128
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            #   16, 128, 64*64
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            #   16, 64*64, 64*64
            f = torch.matmul(theta_x, phi_x)
            N = f.size(-1)
            f_div_C = f / N
        elif self.types=='embedded_gaussian':#self-attention
            #   16, 64*64, 128
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)                              #query
            #   16, 128, 64*64
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)   #key
            #   16, 64*64, 64*64
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)                                  #attention_map
        elif self.types=='gaussian':
            #   16, 64*64, 128
            theta_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            #   16, 128, 64*64
            if self.sub_sample:
                phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
            else:
                phi_x = x.view(batch_size, self.in_channels, -1)
            #   16, 64*64, 64*64
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
        
        #   16, 64*64, 128
        y = torch.matmul(f_div_C, g_x)                                  #out
        y = y.permute(0, 2, 1).contiguous()
        #   16, 128, 64, 64
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
        
        

