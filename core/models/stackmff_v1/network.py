"""Vendored upstream network definition (MIT).

Source: https://github.com/Xinzhe99/StackMFF
Only the architecture is kept; inference, tiling and bit-depth
handling live in OpenFocus (`fusion_methods/`).
"""
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import platform

import torch
from torch import nn
from torch.nn import Module, Sequential, BatchNorm3d, ReLU
import torch.nn.functional as F


_MACOS_26 = platform.system() == 'Darwin' and platform.mac_ver()[0].split('.')[0] == '26'


def _mps_26(x):
    return _MACOS_26 and x.device.type == 'mps'


def _planes(x):
    """Fold depth into the batch for spatial 2D operations."""
    n, c, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)


def _volume(x, n, d):
    return x.reshape(n, d, *x.shape[1:]).permute(0, 2, 1, 3, 4)


class Conv3d(nn.Conv3d):
    def forward(self, x):
        if not _mps_26(x):
            return super().forward(x)

        # Sum one spatial convolution per depth offset.
        n, _, depth, _, _ = x.shape
        depth_pad = self.padding[0]
        depth_stride = self.stride[0]
        depth_dilation = self.dilation[0]
        out_depth = (depth + 2 * depth_pad - depth_dilation *
                     (self.kernel_size[0] - 1) - 1) // depth_stride + 1
        padded = F.pad(x, (0, 0, 0, 0, depth_pad, depth_pad)) if depth_pad else x
        result = None
        for offset in range(self.kernel_size[0]):
            start = offset * depth_dilation
            planes = _planes(padded[:, :, start:start + out_depth * depth_stride:depth_stride])
            part = F.conv2d(planes, self.weight[:, :, offset],
                            self.bias if offset == 0 else None,
                            self.stride[1:], self.padding[1:],
                            self.dilation[1:], self.groups)
            result = part if result is None else result + part
        return _volume(result, n, out_depth)


class ConvTranspose3d(nn.ConvTranspose3d):
    def forward(self, x, output_size=None):
        if not _mps_26(x):
            return super().forward(x, output_size)
        if output_size is not None:
            raise ValueError('Explicit output_size is unsupported by the MPS decoder')
        # The decoder's depth kernel and stride are both one.
        n, _, depth, _, _ = x.shape
        result = F.conv_transpose2d(
            _planes(x), self.weight[:, :, 0], self.bias, self.stride[1:],
            self.padding[1:], self.output_padding[1:], self.groups,
            self.dilation[1:])
        return _volume(result, n, depth)


class MaxPool3d(nn.MaxPool3d):
    def forward(self, x):
        if not _mps_26(x):
            return super().forward(x)
        # All four pools operate only on the spatial dimensions.
        n, _, depth, _, _ = x.shape
        result = F.max_pool2d(_planes(x), 2, 2)
        return _volume(result, n, depth)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class CoordAtt_3d(nn.Module):
    def __init__(self, inp, oup, reduction=32, threshold=8):
        super(CoordAtt_3d, self).__init__()

        class AvgPool3d_d(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=(3, 4), keepdim=True)

        class AvgPool3d_h(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=(2, 4), keepdim=True)

        class AvgPool3d_w(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=(2, 3), keepdim=True)

        self.pool_d = AvgPool3d_d()
        self.pool_h = AvgPool3d_h()
        self.pool_w = AvgPool3d_w()

        mip = max(threshold, inp // reduction)
        self.conv1 = Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()
        self.conv_d = Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, d, h, w = x.size()

        x_d = self.pool_d(x)
        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4)
        x_w = self.pool_w(x).permute(0, 1, 4, 2, 3)

        y = torch.cat([x_h, x_w, x_d], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_w = x_w.permute(0, 1, 3, 4, 2)

        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h * a_d
        return out


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):

        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel),
                                    stride=(1,stride,stride), padding=(0, padding, padding), output_padding=0, bias=True),
                        ReLU())

    def forward(self, x):

        return self.deconv(x)

class StackMFF(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self,num_channels=1,feat_channels=[16, 32, 64, 128, 256], residual='conv'):
        #[64, 128, 256, 512, 1024]

        super(StackMFF, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1,2,2))
        self.pool2 = MaxPool3d((1,2,2))
        self.pool3 = MaxPool3d((1,2,2))
        self.pool4 = MaxPool3d((1,2,2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        #3DCA
        self.att=CoordAtt_3d(feat_channels[0],feat_channels[0])

        # Output convolution
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.smooth_layer=ConvBN(kernel_size=3, padding=1, stride=1, in_channels=1, out_channels=1)
        self.act_hardact = nn.Sigmoid()


    def forward(self, x):
        # Encoder
        x1 = self.conv_blk1(x)
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)
        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        #3DCA
        d_high1=self.att(d_high1)


        out_decoder=self.one_conv(d_high1)
        # AdaptiveMaxPool3d is unsupported on MPS; reduce the full depth instead.
        out_reduce_depth = torch.amax(out_decoder, dim=2, keepdim=True)
        out_squeeze = torch.squeeze(out_reduce_depth, 2)
        out_smooth = self.smooth_layer(out_squeeze)
        out = self.act_hardact(out_smooth)

        return out
