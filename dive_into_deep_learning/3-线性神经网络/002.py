# 正态分布
import math
import numpy as np
import sys
sys.path.append("..")
from dive_into_deep_learning.d2l.d2l_torch import plot

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 再次使用numpy进行可视化 
x = np.arange(-7, 7, 0.01)
# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
# 这里的 normal(x, mu, sigma) for mu, sigma in params 可以看作一个For循环并提取参数到normal方法中
plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


plot.show()