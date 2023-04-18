# 参数管理 - 内置初始化
import torch
from torch import nn

# 将所有权重参数初始化为标准差为0.01的高斯随机变量，且将 偏置参数设置为0。
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

# 初始化为常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
        
# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


if __name__ == '__main__':

    # net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))

    # net.apply(init_normal)
    # print(net[0].weight.data[0], net[0].bias.data[0])

    # net.apply(init_constant)
    # print(net[0].weight.data[0], net[0].bias.data[0])

    # 对某些块应用不同的初始化方法
    # net[0].apply(init_xavier)
    # net[2].apply(init_42)
    # print(net[0].weight.data[0])
    # print(net[2].weight.data)

    # 自定义初始化
    # net.apply(my_init)
    # print(net[0].weight[:2])

    # 参数绑定
    # 我们需要给共享层一个名称，以便可以引用它的参数 
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))
    net(X)
    # 检查参数是否相同
    print(net[2].weight.data[0] == net[4].weight.data[0]) 
    net[2].weight.data[0, 0] = 100
    # 确保它们实际上是同一个对象，而不只是有相同的值 
    print(net[2].weight.data[0] == net[4].weight.data[0])
    