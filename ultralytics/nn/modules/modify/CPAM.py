
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ska import SKA

class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, ks=1, stride=1, pad=0, groups=1):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                           stride=stride, padding=pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1,
                                 affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(self.c(x))

class MultiScaleDynamicConv(nn.Module):
    def __init__(self, dim, sks=3, groups=8, attn_ratio=4):
        """
        dim: 输入通道数
        sks: 小核展开卷积大小  
        groups: 分组数
        attn_ratio: 通道注意力隐藏层缩放比例
        """
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.sks = sks

        # 多尺度大核感知
        # self.conv5 = Conv2d_BN(dim, dim, ks=5, stride=1, pad=3)

        # self.conv3 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1)

        self.conv7 = Conv2d_BN(dim, dim, ks=7, stride=1, pad=3)
        
        self.conv11 = Conv2d_BN(dim, dim, ks=11, stride=1, pad=5)
        
        # 修改：生成正确形状的权重用于SKA
        # SKA期望的权重形状: [batch, wc, kernel_size^2, height, width]
        # 这里我们让 wc = groups，这样可以减少参数量
        self.wc = groups
        self.proj_weight = nn.Conv2d(dim, self.wc * sks**2, kernel_size=1, stride=1)
        self.norm_weight = nn.GroupNorm(num_groups=1, num_channels=self.wc * sks**2, eps=1e-05)

        # 通道注意力
        hidden_dim = max(dim // attn_ratio, 1)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 多尺度卷积融合
        feat7 = self.conv7(x)
        feat11 = self.conv11(x)
        # feat3 = self.conv3(x)
        # feat5 = self.conv5(x)
        feat = feat7 + feat11
        # feat = feat5 + feat7

        # 生成SKA所需的权重张量
        w = self.proj_weight(self.act(feat))  # [B, wc*sks^2, H, W]
        w = self.norm_weight(w)
        w = w.view(B, self.wc, self.sks**2, H, W)  # [B, wc, K^2, H, W]

        # 通道注意力
        attn = F.adaptive_avg_pool2d(feat, 1).view(B, C)
        attn = self.fc2(self.act(self.fc1(attn)))
        attn = self.sigmoid(attn).view(B, C, 1, 1)
        
        # 应用注意力到输入特征
        x_attended = x * attn

        return x_attended, w

class CPAM(nn.Module):
    def __init__(self, dim, sks=3, groups=8):
        super(MultiScaleDynamicConv, self).__init__()
        self.lkp = MultiScaleDynamicConv(dim, sks=sks, groups=groups)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # 获取处理后的特征和权重
        x_processed, w = self.lkp(x)
        
        # 使用SKA进行特征加权
        out = self.ska(x_processed, w)
        
        # 批归一化和残差连接
        return self.bn(out) + x