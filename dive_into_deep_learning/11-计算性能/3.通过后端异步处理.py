import os
import subprocess
import numpy
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import Benchmark, try_gpu


if __name__ == '__main__':
    # GPU计算热身
    device = try_gpu()
    a = torch.randn(size=(1000, 1000), device=device) 
    b = torch.mm(a, a)
    with Benchmark('numpy'):
        for _ in range(10):
            a = numpy.random.normal(size=(1000, 1000))
            b = numpy.dot(a, a)
    with Benchmark('torch'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)


    with Benchmark():
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)
        torch.cuda.synchronize(device)