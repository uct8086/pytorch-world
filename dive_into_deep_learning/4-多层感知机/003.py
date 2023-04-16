import math
import numpy as np
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import evaluate_loss

if __name__ == "__main__":

    # 多项式的最⼤阶数
    max_degree = 20
    # 训练和测试数据集⼤⼩
    n_train, n_test = 100, 100
    # 分配⼤量的空间
    true_w = np.zeros(max_degree)

    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    print(np.array([5, 1.2, -3.4, 5.6]))
    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        # gamma(n)=(n-1)!
        poly_features[:, i] /= math.gamma(i + 1)

    # labels的维度:(n_train+n_test,)
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in [true_w, features, poly_features, labels]
    ]
