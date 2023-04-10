import math
import time
import numpy as np
import torch
import sys
sys.path.append("..")
from d2l.torch import Timer
n = 10000
a = torch.ones([n])
b = torch.ones([n])

c = torch.zeros(n)
timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop():.5f} sec')

timer.start()
# 使用重载的+运算符来计算按元素的和
d = a + b 
print(f'{timer.stop():.5f} sec')