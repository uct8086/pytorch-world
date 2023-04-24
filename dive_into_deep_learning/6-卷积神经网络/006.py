# 多输出

import torch
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import corr2d

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度(通道维度)，再把它们加在一起 
    return sum(corr2d(x, k) for x, k in zip(X, K)) # zip函数并排迭代两个或多个列表

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。 # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


if __name__ == '__main__':

    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    K = torch.stack((K, K + 1, K + 2), 0) # (tensor, dim) 张量，维度
    print(K.shape)
    print(corr2d_multi_in_out(X, K))