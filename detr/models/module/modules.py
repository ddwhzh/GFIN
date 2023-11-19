import torch
import torch.nn as nn
import torch.nn.functional as F
from  detr.models.necks.module.utils import _get_activation_fn
import sys

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d，其中batch statistics和affine参数是固定的。
    从 torchvision.misc.ops复制粘贴，在 rqsrt 之前添加了 eps，
    除了 torchvision.models.resnet[18,34,50,101] 之外的任何其他模型都会产生 nans。
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        #   4个量注册到buffer，阻止梯度反向传播而更新它们
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly   
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias



BN_MOMENTUM = 0.1

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def Conv1x1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, 1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#conv+bn+relu
#conv+relu
class convolution(nn.Module):
    def __init__(self, kernel, inp_dim, out_dim, stride=1, with_bn=True, act='relu'):
        super(convolution, self).__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (kernel, kernel), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = get_activation(name=act, inplace=True)


    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        out = self.relu(bn)
        return out

class residual(nn.Module):
    def __init__(self, kernel, inp_dim, out_dim, stride=1, with_bn=True, act='relu'):
        super(residual, self).__init__()
        pad = (kernel - 1) // 2
        
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (kernel, kernel), padding=(pad, pad), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu1 = get_activation(name=act, inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (kernel, kernel), padding=(pad, pad), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = get_activation(name=act, inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,  act='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = get_activation(name=act, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = get_activation(name=act, inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1, act='relu'):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = get_activation(name=act, inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True, act='relu'):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = get_activation(name=act, inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = convolution(ksize, in_channels * 4, out_channels,  stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = convolution(1,in_channels, hidden_channels, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = convolution(1, conv2_channels, out_channels, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

#TCN
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channel, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out
    
class MultiStageTCN(nn.Module):
    def __init__(self, in_channel, n_classes, stages,
                 n_features=64, dilated_n_layers=10, kernel_size=15):
        super().__init__()
        if stages[0] == 'dilated':
            self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, dilated_n_layers)
        elif stages[0] == 'etspnet':
            self.stage1 = Single_ETSPNet(in_channel, n_features, n_classes, dilated_n_layers) 
        # elif stages[0] == 'ed':
        #     self.stage1 = EDTCN(in_channel, n_classes)
        else:
            print("Invalid values as stages in Mixed Multi Stage TCN")
            sys.exit(1)

        if len(stages) == 1:
            self.stages = None
        else:
            self.stages = []
            for stage in stages[1:]:
                if stage == 'dilated':
                    self.stages.append(SingleStageTCN(n_classes, n_features, n_classes, dilated_n_layers))
                # elif stage == 'ed':
                #     self.stages.append(EDTCN(n_classes, n_classes, kernel_size=kernel_size))
                else:
                    print("Invalid values as stages in Mixed Multi Stage TCN")
                    sys.exit(1)
            self.stages = nn.ModuleList(self.stages)

    def forward(self, x):
        if self.training:
            # # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            if self.stages is not None:
                for stage in self.stages:
                    out = stage(F.softmax(out, dim=1))
                    outputs.append(out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            if self.stages is not None:
                for stage in self.stages:
                    out = stage(F.softmax(out, dim=1))
            return out

class Single_ETSPNet(nn.Module):
    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)])
        self.act = nn.PReLU(n_features*10)  
        self.conv1_1 = nn.Conv1d(n_features*10, n_features, 1)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
        out = []
        out.append(self.conv_in(x))
        for index, layer in enumerate(self.layers):
            if index == 0: 
                out.append(layer(out[0]))
            else:
                out.append(layer(out[0] + out[index]))
        out_cat = torch.cat(out[1:], dim=1)
        out_cat = self.act(out_cat)
        output = self.conv1_1(out_cat)
        output = self.conv_out(output+out[0])
        return output

class SingleStageTCN(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py
    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)])
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

class DilatedResidualLayer(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py
    def __init__(self, dilation, in_channel, out_channels):
        super().__init__()
        #   
        self.conv_dilated = nn.Conv1d(in_channel, out_channels, 3, padding=dilation, dilation=dilation)
        #   
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out
    


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = Conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = Conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
