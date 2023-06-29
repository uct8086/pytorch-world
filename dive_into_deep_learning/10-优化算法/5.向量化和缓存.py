import numpy as np
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import Timer



if __name__ == '__main__':
    timer = Timer()
    A = torch.zeros(256, 256)
    B = torch.randn(256, 256)
    C = torch.randn(256, 256)

    # 逐元素计算A=BC 
    timer.start()
    for i in range(256):
        for j in range(256):
            A[i, j] = torch.dot(B[i, :], C[:, j])

    print(timer.stop())

    # 逐列计算A=BC 
    timer.start()
    for j in range(256):
        A[:, j] = torch.mv(B, C[:, j])
    print(timer.stop())

    # 一次性计算A=BC
    timer.start()
    A = torch.mm(B, C) 
    print(timer.stop())

    # 乘法和加法作为单独的操作(在实践中融合)
    gigaflops = [2/i for i in timer.times]
    print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
        f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')

    # 小批量
    timer.start()
    for j in range(0, 256, 64):
        A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
    timer.stop()
    print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')