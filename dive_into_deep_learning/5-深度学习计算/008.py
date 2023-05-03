import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import try_gpu, try_all_gpus

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())

print(try_gpu(), try_gpu(10), try_all_gpus())

if __name__ == '__main__':
    x = torch.tensor([1, 2, 3])
    print(x.device)

    X = torch.ones(2, 3, device=try_gpu())
    print(X)