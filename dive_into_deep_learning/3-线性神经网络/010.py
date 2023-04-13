# softmax回归的航简洁实现
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.torch import load_data_fashion_mnist, train_ch3

if __name__ == '__main__': 
    batch_size = 256
    # 每个样本都是28 × 28的图像。展平每个图像，把它们看作⻓度为784的向量,数据集有10个类别
    train_iter, test_iter = load_data_fashion_mnist(batch_size) # 加载图片数据集

    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层(flatten)，来调整网络输入的形状 
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);

    # 交叉熵损失
    loss = nn.CrossEntropyLoss(reduction='none')
    # 随机梯度下降
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    # 训练10次
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)