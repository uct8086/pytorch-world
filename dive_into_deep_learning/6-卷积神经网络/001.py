# 互相关运算

import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import corr2d

# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


if __name__ == '__main__':
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # print(corr2d(X, K))
    # 构造黑白图像
    X = torch.ones((6, 8))
    X[:, 2:6] = 0

    print(X)
    # 高度为1，宽度为2的卷积核
    K = torch.tensor([[1.0, -1.0]])
    # 执行互相关运算
    Y = corr2d(X, K)
    print(Y)
    print( corr2d(X.t(), K))