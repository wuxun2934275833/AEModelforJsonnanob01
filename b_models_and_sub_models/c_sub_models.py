import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


# SSPCAB implementation
class SSPCAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2*dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)#在x周围加上padding
        #按照后两个维度，分割成4块
        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x


import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class conv3BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, stride, padding):
        super(conv3BnRelu, self).__init__()
        self.conv = conv3x3(in_chan, out_chan, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x



class conv1BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, stride, padding):
        super().__init__()
        self.conv = conv1x1(in_chan, out_chan, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)#同样的问题

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.softmax(x3)

        return x3 * x1

class ChannelAttention(nn.Module):
    #CBAM模块用来替换attention
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_att(x)
        # Apply spatial attention
        x = x * self.spatial_att(x)
        return x


class Attention2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)#这里可能有个错误，softmax实际是作用在了特征图的行上，而不是常见的加载了通道维度上。

    def forward(self, x):
        x1 = self.conv1(x)#x1是对x作卷积
        x1 = F.relu(x1)

        x2 = self.conv2(x1)#x2对x1作卷积
        x2 = self.softmax(x2)

        return x2 * x1




class BasicBlockDe(nn.Module):
    expansion: int = 1
    '''用于解码器上采样的BasicBlock'''
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:#这里的参数groups是控制是否要进行分组卷积的。
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1_output = None
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        self.conv1_output = out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out


class SCSA_Block(nn.Module):
    """
    Structured Channel-Spatial Aggregation Block
    结构感知 + 通道注意力 + 空间注意力
    """

    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=16, spatial_kernel=7):
        super(SCSA_Block, self).__init__()

        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_dim, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_dim, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_dim, bias=False)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_dim, bias=False)

        self.bn_struct = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        hidden_channels = max(channels // reduction_ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )

        self.channel_sigmoid = nn.Sigmoid()

        assert spatial_kernel in (3, 7), "spatial_kernel must be 3 or 7"
        padding = 3 if spatial_kernel == 7 else 1
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x_pad[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x_pad[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x_pad[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x_pad[:, :, self.border_input:, self.border_input:])

        x_struct = x1 + x2 + x3 + x4
        x_struct = self.bn_struct(x_struct)
        x_struct = self.relu(x_struct)

        avg_out = self.channel_mlp(self.avg_pool(x_struct))
        max_out = self.channel_mlp(self.max_pool(x_struct))
        channel_att = self.channel_sigmoid(avg_out + max_out)

        x_channel = x_struct * channel_att

        avg_map = torch.mean(x_channel, dim=1, keepdim=True)
        max_map, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))

        out = x_channel * spatial_att

        out = out + identity

        return out





