# 多层感知机简洁实现
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.torch import load_data_fashion_mnist, train_ch3

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == '__main__': 

    # 定义网络模型， 二层
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    # 三层
    # net = nn.Sequential(nn.Flatten(),
    #                     nn.Linear(784, 256),
    #                     nn.ReLU(),
    #                     nn.Linear(256, 128),
    #                     nn.ReLU(),
    #                     nn.Linear(128, 10))
    
    net.apply(init_weights)

    batch_size, lr, num_epochs = 256, 0.1, 10
    # 交叉熵损失
    loss = nn.CrossEntropyLoss(reduction='none')
    # 随机梯度下降，更新函数
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 加载数据集
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 训练
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
