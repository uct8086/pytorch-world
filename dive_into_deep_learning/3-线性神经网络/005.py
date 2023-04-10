import numpy as np
import torch
from torch.utils import data
import sys
sys.path.append("..")
from d2l.torch import load_array, synthetic_data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))