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

    # print(X)
    # 高度为1，宽度为2的卷积核
    K = torch.tensor([[1.0, -1.0]])
    # 执行互相关运算
    Y = corr2d(X, K)
    # print(Y)
    # print( corr2d(X.t(), K))
    # 构造一个二维卷积层，它具有1个输出通道和形状为(1，2)的卷积核 
    conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
    # 这个二维卷积层使用四维输入和输出格式(批量大小、通道、高度、宽度)， 
    # # 其中批量大小和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2 # 学习率
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad 
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')
    print( conv2d.weight.data.reshape((1, 2)))


