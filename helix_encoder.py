import math
import torch
import torch.nn as nn


# class HelixPosEncoder (nn.Module):  #time idx to a 3d dna like vector 
#     def __init__(self,scale_init = 200.0):
#         super().__init__()
#         self.scale = nn.Parameter(torch.tensor(scale_init))
class HelixPosEncoder(nn.Module):
    def __init__(self, scale_init=200.0):
        super().__init__()
       # 可学习螺旋参数
        self.freq = nn.Parameter(torch.randn(1))   # 转速  (≈1/周期)
        self.ax   = nn.Parameter(torch.ones(1))    # x 半径
        self.ay   = nn.Parameter(torch.ones(1))    # y 半径
        self.scale = scale_init                    # 仅用于 z 归一化
    def forward(self, t_idx):               # t_idx (B,T)
        theta = 2 * math.pi * self.freq * t_idx          # (B,T)
        x = self.ax * torch.cos(theta)
        y = self.ay * torch.sin(theta)
        z = theta / (2 * math.pi * self.scale)           # 纵向递增
        return torch.stack([x, y, z], dim=-1)            # (B,T,3)