# nn是神经网络的缩写 
import torch
from torch import nn
import sys
sys.path.append("..")
from d2l.torch import load_array, synthetic_data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


# print(next(iter(data_iter)))
net = nn.Sequential(nn.Linear(2, 1))
print(net)
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 访问线性回归的梯度
print(net[0].weight.grad)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape)) 
b = net[0].bias.data
print('b的估计误差:', true_b - b)