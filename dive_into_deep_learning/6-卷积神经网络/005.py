# 多输入通道

import torch
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import corr2d

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度(通道维度)，再把它们加在一起 
    return sum(corr2d(x, k) for x, k in zip(X, K)) # zip函数并排迭代两个或多个列表

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))