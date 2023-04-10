# 线性回归从0开始实现
import torch
import random
import sys
sys.path.append("..")
from d2l.torch import synthetic_data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
# plt.show()

def data_iter(batch_size, features, labels): 
    num_examples = len(features)
    indices = list(range(num_examples))
    # print(indices)
    # 这些样本是随机读取的，没有特定的顺序 
    random.shuffle(indices)
    # print(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
