# 参数管理
import torch
from torch import nn

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


if __name__ == '__main__':

    # net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    # print(net(X))
    # # 第2个神经元的参数
    # print(net[2].state_dict())

    # # 第2个神经元的偏置类型
    # print(type(net[2].bias))
    # # 偏置
    # print(net[2].bias)
    # # 偏置的值
    # print(net[2].bias.data)
    # # 由于我们还没有调用反向传播，所以参数的梯度处于初始状态
    # print(net[2].weight.grad == None)

    # print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    # print(*[(name, param.shape) for name, param in net.named_parameters()])

    # print(net.state_dict()['2.bias'].data)
    # 从嵌套块收集参数
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet(X))

    print(rgnet)

    # 可以通过索引访问
    print(rgnet[0][1][0].bias.data)