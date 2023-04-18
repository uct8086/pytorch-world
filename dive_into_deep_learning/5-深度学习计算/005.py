# 自定义层-不带参数的层
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean() # 减去均值

if __name__ == '__main__':

    layer = CenteredLayer()
    # print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())