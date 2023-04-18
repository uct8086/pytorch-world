# 延后初始化

import torch
from torch import nn

"""延后初始化"""
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net[0].weight)  # 尚未初始化
print(net)

X = torch.rand(2, 20)
net(X)
print(net)