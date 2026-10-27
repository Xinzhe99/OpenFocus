"""Vendored upstream network definition (MIT).

Source: https://github.com/Xinzhe99/StackMFF-V2
Only the architecture is kept; inference, tiling and bit-depth
handling live in OpenFocus (`fusion_methods/`).
"""
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
Network architecture for StackMFF-V2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Channel_Att_Bridge(nn.Module):
    """Channel Attention Bridge Module
    
    Args:
        c_list: List of channel dimensions
        split_att: Attention split type ('fc' or 'conv')
    """
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    """Spatial Attention Bridge Module
    Applies spatial attention using average and max pooling
    """
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    """Combined Spatial and Channel Attention Bridge
    
    Args:
        c_list: List of channel dimensions
        split_att: Attention split type
    """
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_



class ConvGRUCell(nn.Module):
    """Convolutional GRU Cell
    
    Args:
        input_dim: Input channels
        hidden_dim: Hidden state channels
        kernel_size: Conv kernel size
        bias: Whether to use bias
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv_gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=2 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)
        self.conv_can = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device)


class BidirectionalConvGRU(nn.Module):
    """Bidirectional Convolutional GRU
    
    Args:
        input_dim: Input channels
        hidden_dim: Hidden state channels
        kernel_size: Conv kernel size
        num_layers: Number of GRU layers
        batch_first: If True, batch dim is first
        bias: Whether to use bias
        return_all_layers: If True, return all layer outputs
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(BidirectionalConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Forward direction
        forward_cell_list = []
        # Backward direction
        backward_cell_list = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            forward_cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                                 hidden_dim=self.hidden_dim[i],
                                                 kernel_size=self.kernel_size[i],
                                                 bias=self.bias))
            backward_cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                                  hidden_dim=self.hidden_dim[i],
                                                  kernel_size=self.kernel_size[i],
                                                  bias=self.bias))

        self.forward_cell_list = nn.ModuleList(forward_cell_list)
        self.backward_cell_list = nn.ModuleList(backward_cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            forward_h, backward_h = hidden_state[layer_idx]
            output_inner = []

            # Forward pass
            forward_output = []
            for t in range(seq_len):
                forward_h = self.forward_cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                              h_cur=forward_h)
                forward_output.append(forward_h)

            # Backward pass
            backward_output = []
            for t in range(seq_len - 1, -1, -1):
                backward_h = self.backward_cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                                h_cur=backward_h)
                backward_output.insert(0, backward_h)

            # Combine forward and backward outputs using addition
            for t in range(seq_len):
                output_inner.append(forward_output[t] + backward_output[t])

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((forward_output[-1], backward_output[-1]))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            forward_state = self.forward_cell_list[i].init_hidden(batch_size, image_size)
            backward_state = self.backward_cell_list[i].init_hidden(batch_size, image_size)
            init_states.append((forward_state, backward_state))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# 论文地址：https://arxiv.org/pdf/2303.16900
# 论文：InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)

class InceptionDWConv2d(nn.Module):
    """Inception-style Depthwise Convolution Module
    Based on InceptionNeXt architecture (CVPR 2024)
    https://arxiv.org/pdf/2303.16900
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        square_kernel_size: Size for square kernel
        band_kernel_size: Size for band-shaped kernels
        branch_ratio: Ratio for branch channels
    """
    def __init__(self, in_channels, out_channels=None, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        self.out_channels = out_channels or in_channels

        # Add 1x1 convolution to adjust channel number if necessary
        if self.out_channels != in_channels:
            self.channel_adj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.channel_adj = nn.Identity()

        gc = max(1, int(self.out_channels * branch_ratio))  # Ensure gc is at least 1

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.split_indexes = (max(0, self.out_channels - 3 * gc), gc, gc, gc)

    def forward(self, x):
        x = self.channel_adj(x)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class FeatureExtraction(nn.Module):
    """Feature Extraction Module
    Encoder-decoder architecture with inception blocks
    
    Args:
        c_list: List of channel dimensions
        split_att: Attention split type
        bridge: Whether to use attention bridge
    """
    def __init__(self, c_list=[8, 16, 24, 32, 48, 64], split_att='fc', bridge=True):
        super(FeatureExtraction, self).__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            InceptionDWConv2d(c_list[0], c_list[1], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder3 = nn.Sequential(
            InceptionDWConv2d(c_list[1], c_list[2], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder4 = nn.Sequential(
            InceptionDWConv2d(c_list[2], c_list[3], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder5 = nn.Sequential(
            InceptionDWConv2d(c_list[3], c_list[4], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.encoder6 = nn.Sequential(
            InceptionDWConv2d(c_list[4], c_list[5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)

        self.decoder1 = nn.Sequential(
            InceptionDWConv2d(c_list[5], c_list[4], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder2 = nn.Sequential(
            InceptionDWConv2d(c_list[4], c_list[3], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder3 = nn.Sequential(
            InceptionDWConv2d(c_list[3], c_list[2], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder4 = nn.Sequential(
            InceptionDWConv2d(c_list[2], c_list[1], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )
        self.decoder5 = nn.Sequential(
            InceptionDWConv2d(c_list[1], c_list[0], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.embbed_dim=c_list[0]
        # self.embbed_dim = 1
        self.final = nn.Conv2d(c_list[0], self.embbed_dim, kernel_size=1)

    def forward(self, x):
        batch_size, num_images, height, width = x.shape
        x_reshaped = x.view(batch_size * num_images, 1, height, width)

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x_reshaped)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        focus_maps_single_layer = torch.sigmoid(out0)

        # print(focus_maps_single_layer.shape)
        # sliced_features = focus_maps_single_layer[:, :8, 541, 205]
        # average_probability = sliced_features.mean(axis=1)
        # print('Initial Probability:', average_probability)
        return focus_maps_single_layer.view(batch_size, num_images, self.embbed_dim, height, width)
class LayerInteraction(nn.Module):
    """Layer Interaction Module
    Uses bidirectional ConvGRU for temporal feature interaction
    """
    def __init__(self):
        super(LayerInteraction, self).__init__()
        self.layer_interaction_depth = BidirectionalConvGRU(input_dim=8, hidden_dim=16, kernel_size=(3, 3),
                                               num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        # self.proj_pool_depth = nn.AdaptiveMaxPool3d(output_size=(1, None, None))
        # 使用固定大小的MaxPool3d替代AdaptiveMaxPool3d
        self.proj_pool_depth = nn.MaxPool3d(kernel_size=(16, 1, 1), stride=(16, 1, 1))
    def forward(self, focus_maps):
        batch_size, num_images, _, height, width = focus_maps.shape

        lstm_out_depth, _ = self.layer_interaction_depth(focus_maps)#[B,N,C,H,W]
        # print(lstm_out_depth[0].size())
        focus_maps_depth = self.proj_pool_depth(lstm_out_depth[0])#[B,N,1,H,W]
        # print(focus_maps_depth.size())
        focus_maps_depth = focus_maps_depth.view(batch_size, num_images, height, width)#[B,N,H,W]\

        return focus_maps_depth

class DepthMapCreation(nn.Module):
    """Depth Map Creation Module
    Creates depth map from focus maps using softmax
    """
    def __init__(self):
        super(DepthMapCreation, self).__init__()

    def forward(self, focus_maps_depth, num_images):
        # # Step 1: 计算焦点概率
        # print('final likehood:', focus_maps_depth[0, :, 541, 205])
        focus_probs = F.softmax(focus_maps_depth, dim=1)
        # print('focus_probs_softmax:', focus_probs[0, :, 541, 205])
        # Step 2: 创建深度索引
        depth_indices = torch.arange(1, num_images + 1).view(1, -1, 1, 1).to(focus_maps_depth.device).float()
        # print(depth_indices)
        # Step 3: 计算深度图索引
        # tmp=focus_probs * depth_indices
        # print('index*pro:', tmp[0, :, 541, 205])
        depth_map = torch.sum(focus_probs * depth_indices, keepdim=True, dim=1)

        # Step 4: 归一化深度图
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        return depth_map

class StackMFF_V2(nn.Module):
    """Main Multi-Focus Fusion Network (Version 2)
    Combines feature extraction, layer interaction and depth map creation
    """
    def __init__(self):
        super(StackMFF_V2, self).__init__()
        self.feature_extraction = FeatureExtraction()
        self.layer_interaction = LayerInteraction()
        self.depth_map_creation = DepthMapCreation()
        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, N, H, W)
                B: batch size
                N: number of input images
                H: height
                W: width
                
        Returns:
            fused_image: Fused output image
            depth_map: Estimated depth map
            depth_map_index: Depth indices
        """
        feature_maps = self.feature_extraction(x)
        focus_maps_depth = self.layer_interaction(feature_maps)
        depth_map = self.depth_map_creation(focus_maps_depth, x.shape[1])
        fused_image, depth_map_index = self.generate_fused_image(x, depth_map)
        return fused_image, depth_map, depth_map_index

    def generate_fused_image(self, x, depth_map):
        """
        Generate fused image using depth map
        
        Args:
            x: Input image stack
            depth_map: Generated depth map
            
        Returns:
            fused_image: Final fused image
            depth_map_index: Depth indices
        """
        batch_size, num_images, height, width = x.shape

        depth_map = depth_map.squeeze(1)  # Shape: [batch_size, height, width]

        depth_map_continuous = (depth_map * (num_images - 1)).clamp(0, num_images - 1)

        depth_map_index = torch.round(depth_map_continuous).long()

        depth_map_index = torch.clamp(depth_map_index, 0, num_images - 1)

        depth_map_index_expanded = depth_map_index.unsqueeze(1)  # [batch_size, 1, height, width]
        
        fused_image = torch.gather(x, 1, depth_map_index_expanded)

        return fused_image, depth_map_index
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BidirectionalConvGRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

if __name__ == "__main__":
    # Test network
    model = StackMFF_V2()
    x = torch.ones(1, 2, 128, 128)
    fused_image, fused_depth_map, depth_map_index = model(x)
    print(fused_image.shape, fused_depth_map.shape, depth_map_index.shape)


