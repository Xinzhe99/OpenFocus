# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
Network architecture for StackMFF-V4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
except ImportError:
    FlopCountAnalysis = None
    flop_count_table = None

class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        nn.init.trunc_normal_(self.weight, std=.02)

    def forward(self, input):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(input), 
                self.weight, self.bias, padding=(self.act_num*2 + 1)//2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(input),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(bias)  # 直接使用返回的bias
        self.__delattr__('bn')
        self.deploy = True


class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        self.act = activation(dim_out, act_num, deploy=self.deploy)
 
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            
            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        # 处理conv可能没有bias的情况
        bias = conv.bias if conv.bias is not None else torch.zeros_like(running_mean)
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True

class UpBlock(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, factor=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear')
        self.act = activation(dim_out, act_num, deploy=self.deploy)
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            
            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            
            x = self.conv2(x)

        x = self.upsample(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        # 处理conv可能没有bias的情况
        bias = conv.bias if conv.bias is not None else torch.zeros_like(running_mean)
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True


class LV_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channel=1,  dims=[20*4,40*4,60*4,120*4], dims2=[80,40,24,16],
                 drop_rate=0, act_num=1, strides=[2,2,2], deploy=False):
        super().__init__()
        self.deploy = deploy
        mobile = models.mobilenet_v3_large(pretrained=True)
        # 修改第一层卷积以适应单通道输入
        self.firstconv = nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # 初始化权重
        nn.init.kaiming_normal_(self.firstconv.weight, mode='fan_out', nonlinearity='relu')
        self.encoder1 = nn.Sequential(
                mobile.features[1],
                mobile.features[2],
            )
        self.encoder2 = nn.Sequential(
             mobile.features[3],
                mobile.features[4],
                mobile.features[5],
            )
        self.encoder3 =  nn.Sequential(
                mobile.features[6],
                mobile.features[7],
                mobile.features[8],
            mobile.features[9]
            )
        self.act_learn = 0
        self.stages = nn.ModuleList()
        self.up_stages1 = nn.ModuleList()
        self.up_stages2 = nn.ModuleList()
        for i in range(len(strides)):
            stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy)
            self.stages.append(stage)
        for i in range(len(strides)):
            stage = UpBlock(dim=dims[3-i], dim_out=dims[2-i], act_num=act_num, factor=strides[2-i],deploy=deploy)
            self.up_stages1.append(stage)
        for i in range(3):
            stage = UpBlock(dim=dims2[i], dim_out=dims2[i+1], act_num=act_num, factor=2,deploy=deploy)
            self.up_stages2.append(stage)
        self.depth = len(strides)
        self.final = nn.ModuleList()  
        self.final.append(UpBlock(dim=16, dim_out=16, act_num=act_num, factor=2))
        self.final.append(nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0))
        
    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        for i in range(self.depth):
            self.up_stages1[i].act_learn = m
        for i in range(3):
            self.up_stages2[i].act_learn = m
        for i in range(len(self.final)):
            self.final[i].act_learn = m
        self.act_learn = m

    def forward(self, x):
        x = self.firstconv(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e4 = self.encoder3(e2)
        encoder=[]
        for i in range(self.depth):
            encoder.append(e4)
            e4 = self.stages[i](e4)
        for i in range(self.depth):
            e4 = self.up_stages1[i](e4)
            # 确保特征图尺寸匹配后再进行加法操作
            if e4.shape[2:] != encoder[2-i].shape[2:]:
                # 如果尺寸不匹配，对encoder特征图进行插值调整
                encoder_resized = F.interpolate(encoder[2-i], size=e4.shape[2:], mode='bilinear', align_corners=False)
                e4 = e4 + encoder_resized
            else:
                e4 = e4 + encoder[2-i]
        # 确保后续加法操作的尺寸匹配
        up_stage_0_out = self.up_stages2[0](e4)
        if up_stage_0_out.shape[2:] != e2.shape[2:]:
            e2_resized = F.interpolate(e2, size=up_stage_0_out.shape[2:], mode='bilinear', align_corners=False)
            e4 = up_stage_0_out + e2_resized
        else:
            e4 = up_stage_0_out + e2
            
        up_stage_1_out = self.up_stages2[1](e4)
        if up_stage_1_out.shape[2:] != e1.shape[2:]:
            e1_resized = F.interpolate(e1, size=up_stage_1_out.shape[2:], mode='bilinear', align_corners=False)
            e4 = up_stage_1_out + e1_resized
        else:
            e4 = up_stage_1_out + e1
            
        e4 = self.up_stages2[2](e4)
        for i in range(len(self.final)):
             e4 = self.final[i](e4)
        return e4

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        # 使用更明确的类型检查来避免类型检查工具的误报
        for i in range(self.depth):
            stage = self.stages[i]
            if hasattr(stage, 'switch_to_deploy') and callable(getattr(stage, 'switch_to_deploy', None)):
                stage.switch_to_deploy()  # type: ignore
        for i in range(self.depth):
            stage = self.up_stages1[i]
            if hasattr(stage, 'switch_to_deploy') and callable(getattr(stage, 'switch_to_deploy', None)):
                stage.switch_to_deploy()  # type: ignore
        for i in range(len(self.up_stages2)):
            stage = self.up_stages2[i]
            if hasattr(stage, 'switch_to_deploy') and callable(getattr(stage, 'switch_to_deploy', None)):
                stage.switch_to_deploy()  # type: ignore
        if len(self.final) > 0:
            final_stage = self.final[0]
            if hasattr(final_stage, 'switch_to_deploy') and callable(getattr(final_stage, 'switch_to_deploy', None)):
                final_stage.switch_to_deploy()  # type: ignore
        self.deploy = True


def lv_unet(num_classes, input_channel=1):
    model = LV_UNet(input_channel=input_channel, num_classes=num_classes)
    return model


# =====================
# Depth-wise Transformer
# =====================
class DepthTransformerLayer(nn.Module):
    """
    单层 Depth Transformer Layer
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_dim = ff_dim or embed_dim * 4
        # Feed-forward 网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B*H*W, N, C]
        """
        # We apply LayerNorm *before* attention, which is a common and stable practice (Pre-LN)
        x_norm = self.norm1(x) 

        # Self-attention 沿 depth 方向
        # x_norm is used as query, key, and value
        x_attn = self.attn(x_norm, x_norm, x_norm)[0]  # [B*H*W, N, C]
        x_attn = self.dropout(x_attn)

        # 残差连接
        x = x + x_attn  # [B*H*W, N, C]

        # Feed-forward 网络
        x_norm2 = self.norm2(x) # Pre-LN for the FFN
        x_ffn = self.ffn(x_norm2)  # [B*H*W, N, C]
        x_ffn = self.dropout(x_ffn)
        
        # 残差连接
        x = x + x_ffn  # [B*H*W, N, C]

        return x


class DepthTransformer(nn.Module):
    """
    沿 depth 方向 (num_images) 建立 Transformer，用于捕捉图层间关系
    输入输出都为 [B, N, C, H, W]
    支持自定义层数
    """
    def __init__(self, embed_dim, num_heads, num_layers=1, ff_dim=None, dropout=0.0, spatial_pool_ratio=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.spatial_pool_ratio = spatial_pool_ratio  # 空间下采样比例
        
        # 创建多层 transformer layers
        self.layers = nn.ModuleList([
            DepthTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape
        
        # 动态计算下采样尺寸
        target_h, target_w = max(1, int(H * self.spatial_pool_ratio)), max(1, int(W * self.spatial_pool_ratio))
        
        # 空间下采样以减少计算量
        x_reshaped = x.reshape(B * N, C, H, W)
        x_pooled = F.adaptive_avg_pool2d(x_reshaped, (target_h, target_w))
        x_pooled = x_pooled.reshape(B, N, C, target_h, target_w)
            
        # 在下采样后的特征图上执行注意力计算
        x_flat = x_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * target_h * target_w, N, C)
 
        for layer in self.layers:
            x_flat = layer(x_flat)

        x_processed = x_flat.reshape(B, target_h, target_w, N, C).permute(0, 3, 4, 1, 2)
        
        # 上采样恢复到原始尺寸
        x_processed_reshaped = x_processed.reshape(B * N, C, target_h, target_w)
        x_upsampled_reshaped = F.interpolate(x_processed_reshaped, size=(H, W), mode='bilinear', align_corners=True)
        x_upsampled = x_upsampled_reshaped.reshape(B, N, C, H, W)
            
        # 使用残差连接保留高频细节信息
        x_output = x + x_upsampled
        
        return x_output


class LayerInteraction(nn.Module):
    """Layer Interaction Module
    """
    def __init__(self, embed_dim, num_transformer_layers=1):
        super(LayerInteraction, self).__init__()
        self.layer_interaction_depth = DepthTransformer(embed_dim=embed_dim, num_heads=4, num_layers=num_transformer_layers)
        # 移除proj_pool_depth，不在这里降维

    def forward(self, focus_maps):
        # 只进行层间交互，不降维
        att_out = self.layer_interaction_depth(focus_maps)#[B,N,C,H,W]
        return att_out


class IntraLayerRefiner(nn.Module):
    """
    对包含了层间上下文信息的特征图进行逐层的细化。
    输入和输出的维度均为 [B, N, C, H, W]。
    """
    def __init__(self, embed_dim, num_refine_blocks=1):
        super().__init__()
        self.embed_dim = embed_dim
        
        refiner_blocks = []
        for _ in range(num_refine_blocks):
            refiner_blocks.append(
                nn.Sequential(
                    Block(embed_dim, embed_dim, stride=1) 
                )
            )
        self.refiner = nn.Sequential(*refiner_blocks)

    def forward(self, x):
        """
        x: [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape
        
        # 将 B 和 N 维度合并，以共享权重的方式处理每一层的特征
        x_reshaped = x.view(B * N, C, H, W)
        
        # 应用细化卷积层
        refined_x = self.refiner(x_reshaped)
        
        # [核心] 使用残差连接，让模块只学习修正量，训练更稳定
        output_reshaped = x_reshaped + refined_x
        
        # 恢复原始维度
        output = output_reshaped.view(B, N, C, H, W)
        return output

class FocusMapCreation(nn.Module):
    """Focus Map Creation Module
    Creates focus index map from focus maps using softmax
    """
    def __init__(self):
        super(FocusMapCreation, self).__init__()

    def forward(self, focus_maps_depth, num_images):
        # Step 1: 计算焦点概率
        focus_probs = F.softmax(focus_maps_depth, dim=1)
        
        # Step 2: 找到概率最大的图层索引（0到N-1）
        focus_map = torch.argmax(focus_probs, dim=1, keepdim=True).float()  # [B, 1, H, W]
        
        return focus_map
        
class StackMFF_V4(nn.Module):
    def __init__(self, num_transformer_layers=2, num_cycles=1): # 新增 num_cycles 参数
        super(StackMFF_V4, self).__init__()

        embed_dim = 8
        
        self.feature_extraction = lv_unet(num_classes=embed_dim, input_channel=1)
        
        # 将"层间-层内"打包成一个修正循环
        self.refinement_cycles = nn.ModuleList()
        for _ in range(num_cycles):
            self.refinement_cycles.append(nn.ModuleDict({
                'inter_model': DepthTransformer(embed_dim=embed_dim, num_heads=4, num_layers=num_transformer_layers),
                'intra_refiner': IntraLayerRefiner(embed_dim=embed_dim, num_refine_blocks=1)
            }))
        self.layer_interaction = LayerInteraction(embed_dim=embed_dim, num_transformer_layers=num_transformer_layers)
        self.proj_pool_depth = nn.MaxPool3d(kernel_size=(embed_dim, 1, 1), stride=(embed_dim, 1, 1))
        self.focus_map_creation = FocusMapCreation()
        
    def forward(self, x):
        """
        前向传播 - 包含了循环修正机制
        """
        batch_size, num_images, height, width = x.shape
        assert num_images >= 2, f"图像数量必须至少为2，当前为{num_images}"
        
        # 1. 初始层内建模
        # 将输入数据从 [B, N, H, W] 转换为 [B*N, 1, H, W] 以适配 feature_extraction 网络
        x_reshaped = x.view(batch_size * num_images, 1, height, width)
        features_single = self.feature_extraction(x_reshaped)  # Shape: [B*N, C, H, W]
        
        # 将特征重新 reshape 为 [B, N, C, H, W]
        _, channels, feat_height, feat_width = features_single.shape
        features = features_single.view(batch_size, num_images, channels, feat_height, feat_width)

        # 2. 开始循环修正
        for cycle in self.refinement_cycles:
            # 2.1 层间建模：获取上下文
            inter_context_features = getattr(cycle, 'inter_model')(features)
            # 添加残差连接
            features_with_context = features + inter_context_features
            
            # 2.2 细化层内建模：利用上下文修正自身特征
            refined_features = getattr(cycle, 'intra_refiner')(features_with_context)
            
            # 更新特征，为下一轮循环或最终输出做准备
            features = refined_features
            
        # 3. 应用层间交互模块（不降维）
        layer_interaction_features = self.layer_interaction(features)
        
        # 4. 最后才进行降维操作
        layer_interaction_features = self.proj_pool_depth(layer_interaction_features).squeeze(2)

        if self.training:
            return layer_interaction_features
        else:
            focus_map = self.focus_map_creation(layer_interaction_features, num_images)
            fused_image, focus_indices = self.generate_fused_image(x, focus_map)
            return fused_image, focus_indices

    def generate_fused_image(self, x, focus_map):
        """
        使用焦点索引图生成融合图像 - 支持可变类别数
        
        Args:
            x: 输入图像栈 [batch_size, num_images, height, width]
            focus_map: 焦点索引图 [batch_size, 1, height, width] 值范围 [0, num_images-1]
            
        Returns:
            fused_image: 最终融合图像 [batch_size, 1, height, width]
            focus_indices: 焦点索引 [batch_size, height, width] 值范围 [0, num_images-1]
        """
        batch_size, num_images, height, width = x.shape
        focus_indices = focus_map.squeeze(1).long()  # [batch_size, height, width]
        
        # 确保索引在有效范围内
        focus_indices = torch.clamp(focus_indices, 0, num_images - 1)
        
        # 使用torch.gather按索引选择像素值
        fused_image = torch.gather(x, dim=1, index=focus_indices.unsqueeze(1))
        
        return fused_image, focus_indices
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

    def switch_to_deploy(self):
        """Switch model to deploy mode"""
        if hasattr(self.feature_extraction, 'switch_to_deploy'):
            self.feature_extraction.switch_to_deploy()

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    # 创建模型并移动到GPU
    model = StackMFF_V4().to("cuda:1")
    # 创建输入数据
    x = torch.randn(1, 2, 256, 256).to("cuda:1")
    
    # 测试推理模式
    model.eval()
    with torch.no_grad():
        fused_image, focus_indices = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Fused image shape: {fused_image.shape}")
    print(f"Focus indices shape: {focus_indices.shape}")
    print(f"Focus index range: [{focus_indices.min().item():.0f}, {focus_indices.max().item():.0f}]")
    
    # 内存使用情况
    print('{:>16s} : {:<.3f} [M]'.format('Max Memory', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
    
    # 计算FLOPs和参数量
    flops = FlopCountAnalysis(model, (x,))
    print(flop_count_table(flops))
    
    # 额外测试不同图像数量
    print("\n=== Testing with different stack sizes ===")
    test_cases = [3, 5, 12]
    for num_images in test_cases:
        print(f"\n--- {num_images} images ---")
        x_test = torch.randn(1, num_images, 256, 256).to("cuda:1")
        
        model.eval()
        with torch.no_grad():
            fused_image, focus_map_index = model(x_test)
        print(f"Input: {x_test.shape} -> Output: {fused_image.shape}, Focus: {focus_map_index.shape}")
        
        model.train()
        features = model(x_test)
        print(f"Training features: {features.shape}")

